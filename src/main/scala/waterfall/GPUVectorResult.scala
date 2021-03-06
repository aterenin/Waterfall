/**
  *  Copyright 2016 Alexander Terenin
  *
  *  Licensed under the Apache License, Version 2.0 (the "License")
  *  you may not use this file except in compliance with the License.
  *  You may obtain a copy of the License at
  *
  *  http://www.apache.org/licenses/LICENSE-2.0
  *
  *  Unless required by applicable law or agreed to in writing, software
  *  distributed under the License is distributed on an "AS IS" BASIS,
  *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  *  See the License for the specific language governing permissions and
  *  limitations under the License.
  * /
  */

package waterfall

import jcuda.jcublas.JCublas2._
import Implicits.{DebugImplicits, DiagUnitImplicits, FillModeImplicits, TransposeImplicits}
import waterfall.Stream.GPUStream

/**
  * A not-yet-evaluated result of a computation that yields a vector
  *
  * @author Alexander Terenin
  *
  * @param computation the computation containing the needed input that will yield selected result
  */
class GPUVectorResult(computation: GPUComputation) {
  /**
    * The assignment operator: executes the GPUComputation in the GPUVectorResult and stored it in the GPUVector passed in as argument
    *
    * @author Alexander Terenin
    * @param y the GPUVector in which output will be stored
    */
  def =:(y: GPUVector): GPUVector = computation match {
    case GPUAlphaXPlusY(x1: GPUVector, x2: GPUVector) => executeSaxpy(x1, x2, y)
    case GPUGeneralMatrixVector(a: GPUMatrix, x: GPUVector) => executeSgemv(a, x, y)
    case GPULeftGeneralMatrixVector(xT: GPUVector, a: GPUMatrix) => executeSgemv(a.T, xT, y, transposeY = true)  // Ax=y is equivalent to y^T = x^T A^T
    case GPUSymmetricMatrixVector(a: GPUSymmetricMatrix, x: GPUVector) => executeSsymv(a, x, y)
    case GPULeftSymmetricMatrixVector(xT: GPUVector, a: GPUSymmetricMatrix) => executeSsymv(a, xT, y, transposeY = true) // xA=y is equivalent to y^T = A x^T since A = A^T
    case GPUTriangularMatrixVector(a: GPUTriangularMatrix, x: GPUVector) => executeStrmv(a, x, y)
    case GPULeftTriangularMatrixVector(xT: GPUVector, a: GPUTriangularMatrix) => executeStrmv(a.T, xT, y, transposeY = true)
    case GPUTriangularSolveVector(ainv: GPUInverseTriangularMatrix, x: GPUVector) => executeStrsv(ainv, x, y)
    case GPULeftTriangularSolveVector(xT: GPUVector, ainv: GPUInverseTriangularMatrix) => executeStrsv(ainv.T, xT, y, transposeY = true)
    case GPUPositiveDefiniteTriangularSolveVector(ainv: GPUInverseSymmetricMatrix, x: GPUVector) => executeSpotrs(ainv, x, y)
    case GPULeftPositiveDefiniteTriangularSolveVector(xT: GPUVector, ainv: GPUInverseSymmetricMatrix) => executeSpotrs(ainv, xT, y, transposeY = true)
    case _ => throw new Exception("wrong vector operation in =:")
  }

  /**
    * The in-place addition operator: executes the GPUComputation in the GPUVectorResult and adds it to the GPUVector passed in as argument
    *
    * @author Alexander Terenin
    * @param y the GPUVector to which output will be added
    * @return
    */
  def +=:(y: GPUVector): GPUVector  = computation match {
    case GPUGeneralMatrixVector(a: GPUMatrix, x: GPUVector) => executeSgemv(a, x, y, inplace = true)
    case GPULeftGeneralMatrixVector(xT: GPUVector, a: GPUMatrix) => executeSgemv(a.T, xT, y, inplace = true, transposeY = true)  // Ax=y is equivalent to y^T = x^T A^T
    case GPUSymmetricMatrixVector(a: GPUSymmetricMatrix, x: GPUVector) => executeSsymv(a, x, y, inplace = true)
    case GPULeftSymmetricMatrixVector(xT: GPUVector, a: GPUSymmetricMatrix) => executeSsymv(a, xT, y, inplace = true, transposeY = true) // xA=y is equivalent to y^T = A x^T since A = A^T
    case _ => throw new Exception("unsupported: cannot execute given vector operation in-place in +=:")
  }

  private def executeSaxpy(x1: GPUVector, x2: GPUVector, y: GPUVector)(implicit stream: GPUStream = Stream.default) = {
    // no need to prepare output - constant and transpose don't affect computation and are mutated at the end anyway

    // check for compatibility
    assert(x1.length == x2.length && x1.length == y.length, s"mismatched vector dimensions: got ${x1.length} != ${x2.length} != ${y.length}")
    assert(x1.isTranspose == x2.isTranspose, s"mismatched vector dimensions: tried to add row vector to column vector")
    assert(x1.constant.isEmpty || x2.constant.isEmpty, s"unsupported: both vectors being added cannot have constant")

    // determine which vector to copy, if any
    val x = if(x1.ptr eq y.ptr) { // operation: y = alpha x_2 + y
      x2
    } else if(x2.ptr eq y.ptr) { // operation: y = alpha x_1 + y
      x1
    } else {
      if(x1.constant.nonEmpty) { // operation: y = alpha x_1 + x_2, so copy x_2 to y
        x2.copyTo(y)
        x1
      } else { // operation: y = alpha x_2 + x_1, so copy x_1 to y
        x1.copyTo(y)
        x2
      }
    }

    // determine constants
    val alpha = x.constant.getOrElse(Waterfall.Constants.one)

    // set stream
    cublasSetStream(Waterfall.cublasHandle, stream.cudaStream_t).checkJCublasStatus()

    // perform in-place matrix addition using single-precision alpha x plus y
    cublasSaxpy(Waterfall.cublasHandle,
      x.length,
      alpha.ptr,
      x.ptr, x.stride,
      y.ptr, y.stride
    ).checkJCublasStatus()

    // return result
    y.mutateConstant(None).mutateTranspose(x.isTranspose)
  }

  private def executeSgemv(A: GPUMatrix, x: GPUVector, y: GPUVector, inplace: Boolean = false, transposeY: Boolean = false)(implicit stream: GPUStream = Stream.default) = {
    // prepare output
    if(!inplace) y.mutateConstant(None) else if(y.constant.isEmpty) y.mutateConstant(Some(Waterfall.Constants.one))
    y.mutateTranspose(false)

    // check for compatibility
    assert(A.numCols == x.length, s"mismatched matrix dimensions: got ${A.numCols} != ${x.length}")
    assert(A.numRows == y.length, s"mismatched vector dimensions: got ${A.numRows} != ${y.length}")
    assert(x.isTranspose == transposeY, s"mismatched vector dimensions: incorrect row/column vector")
    assert(A.constant.isEmpty || x.constant.isEmpty, s"unsupported: only one input constant can be defined")

    // determine constants
    val alpha = A.constant.getOrElse(x.constant.getOrElse(Waterfall.Constants.one))
    val beta = y.constant.getOrElse(Waterfall.Constants.zero)

    // set stream
    cublasSetStream(Waterfall.cublasHandle, stream.cudaStream_t).checkJCublasStatus()

    // perform single-precision general matrix-vector multiplication
    cublasSgemv(Waterfall.cublasHandle,
      A.isTranspose.toTransposeOpId,
      if(!A.isTranspose) A.numRows else A.numCols, if(!A.isTranspose) A.numCols else A.numRows,
      alpha.ptr,
      A.ptr, A.leadingDimension,
      x.ptr, x.stride,
      beta.ptr,
      y.ptr, y.stride
    ).checkJCublasStatus()

    // return result
    y.mutateConstant(None).mutateTranspose(transposeY)
  }

  private def executeSsymv(A: GPUSymmetricMatrix, x: GPUVector, y: GPUVector, inplace: Boolean = false, transposeY: Boolean = false)(implicit stream: GPUStream = Stream.default) = {
    // prepare output
    if(!inplace) y.mutateConstant(None) else if(y.constant.isEmpty) y.mutateConstant(Some(Waterfall.Constants.one))
    y.mutateTranspose(false)

    // check for compatibility
    assert(A.size == x.length, s"mismatched matrix dimensions: got ${A.size} != ${x.length}")
    assert(x.length == y.length, s"mismatched vector dimensions: got ${x.length} != ${y.length}")
    assert(x.isTranspose == transposeY, s"mismatched vector dimensions: incorrect row/column vector")
    assert(A.constant.isEmpty || x.constant.isEmpty, s"unsupported: only one input constant can be defined")

    // determine constants
    val alpha = A.constant.getOrElse(x.constant.getOrElse(Waterfall.Constants.one))
    val beta = y.constant.getOrElse(Waterfall.Constants.zero)

    // set stream
    cublasSetStream(Waterfall.cublasHandle, stream.cudaStream_t).checkJCublasStatus()

    // perform single-precision symmetric matrix-vector multiplication
    cublasSsymv(Waterfall.cublasHandle,
      A.fillMode.toFillModeId,
      A.size,
      alpha.ptr,
      A.ptr, A.leadingDimension,
      x.ptr, x.stride,
      beta.ptr,
      y.ptr, y.stride
    ).checkJCublasStatus()

    // return result
    y.mutateConstant(None).mutateTranspose(transposeY)
  }

  private def executeStrmv(A: GPUTriangularMatrix, x: GPUVector, y: GPUVector, transposeY: Boolean = false)(implicit stream: GPUStream = Stream.default) = {
    // prepare output
    y.mutateConstant(None).mutateTranspose(false)

    // check for compatibility
    assert(A.size == x.length, s"mismatched matrix dimensions: got ${A.size} != ${x.length}")
    assert(x.length == y.length, s"mismatched vector dimensions: got ${x.length} != ${y.length}")
    assert(x.isTranspose == transposeY, s"mismatched vector dimensions: incorrect row/column vector")
    assert(A.constant.isEmpty && x.constant.isEmpty, s"unsupported: this operation cannot be performed with constants")

    // if not in-place, copy to output
    if(!(x.ptr eq y.ptr)) x.copyTo(y)

    // set stream
    cublasSetStream(Waterfall.cublasHandle, stream.cudaStream_t).checkJCublasStatus()

    // perform single-precision triangular matrix-vector multiplication
    cublasStrmv(Waterfall.cublasHandle,
      A.fillMode.toFillModeId,
      A.isTranspose.toTransposeOpId,
      false.toDiagUnitId, // Waterfall doesn't track triangular matrices with unit diagonals, so assume that is false
      A.size, A.ptr, A.leadingDimension,
      y.ptr, y.stride
    ).checkJCublasStatus()

    // return result
    y.mutateConstant(None).mutateTranspose(transposeY)
  }

  private def executeStrsv(Ainv: GPUInverseTriangularMatrix, x: GPUVector, y: GPUVector, transposeY: Boolean = false)(implicit stream: GPUStream = Stream.default) = {
    // prepare output
    y.mutateConstant(None).mutateTranspose(false)

    // check for compatibility
    assert(Ainv.size == x.length, s"mismatched matrix dimensions: got ${Ainv.size} != ${x.length}")
    assert(x.length == y.length, s"mismatched vector dimensions: got ${x.length} != ${y.length}")
    assert(x.isTranspose == transposeY, s"mismatched vector dimensions: incorrect row/column vector")
    assert(x.constant.isEmpty, s"unsupported: this operation cannot be performed with constants")

    // if not in-place, copy to output
    if(!(x.ptr eq y.ptr)) x.copyTo(y)

    // set stream
    cublasSetStream(Waterfall.cublasHandle, stream.cudaStream_t).checkJCublasStatus()

    // perform single-precision triangular solve vector
    cublasStrsv(Waterfall.cublasHandle,
      Ainv.fillMode.toFillModeId,
      Ainv.isTranspose.toTransposeOpId,
      false.toDiagUnitId, // Waterfall doesn't track triangular matrices with unit diagonals, so assume that is false
      Ainv.size, Ainv.ptr, Ainv.leadingDimension,
      y.ptr, y.stride
    ).checkJCublasStatus()

    // return result
    y.mutateConstant(None).mutateTranspose(transposeY)
  }


  private def executeSpotrs(Ainv: GPUInverseSymmetricMatrix, x: GPUVector, y: GPUVector, transposeY: Boolean = false)(implicit stream: GPUStream = Stream.default) = {
    // prepare output
    y.mutateConstant(None).mutateTranspose(false)

    // check for compatibility
    assert(Ainv.size == x.length, s"mismatched matrix dimensions: got ${Ainv.size} != ${x.length}")
    assert(x.length == y.length, s"mismatched vector dimensions: got ${x.length} != ${y.length}")
    assert(x.isTranspose == transposeY, s"mismatched vector dimensions: incorrect row/column vector")
    assert(x.constant.isEmpty, s"unsupported: this operation cannot be performed with constants")

    // get Cholesky
    val R = Ainv.underlyingCholesky

    // if not in-place, copy to output
    if(!(x.ptr eq y.ptr)) x.copyTo(y)

    // set stream
    cublasSetStream(Waterfall.cublasHandle, stream.cudaStream_t).checkJCublasStatus()

    // perform single-precision triangular solve vector, first one on transpose of Cholesky
    cublasStrsv(Waterfall.cublasHandle,
      R.fillMode.toFillModeId,
      (!R.isTranspose).toTransposeOpId,
      false.toDiagUnitId, // Waterfall doesn't track triangular matrices with unit diagonals, so assume that is false
      R.size, R.ptr, R.leadingDimension,
      y.ptr, y.stride
    ).checkJCublasStatus()

    // perform single-precision triangular solve vector, second one not on transpose of Cholesky
    cublasStrsv(Waterfall.cublasHandle,
      R.fillMode.toFillModeId,
      R.isTranspose.toTransposeOpId,
      false.toDiagUnitId, // Waterfall doesn't track triangular matrices with unit diagonals, so assume that is false
      R.size, R.ptr, R.leadingDimension,
      y.ptr, y.stride
    ).checkJCublasStatus()

    // return result
    y.mutateConstant(None).mutateTranspose(transposeY)
  }
}