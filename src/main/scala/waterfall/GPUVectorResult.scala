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
import Implicits.{DebugImplicits, TransposeImplicits, FillModeImplicits}

/**
  * A not-yet-evaluated result of a computation that yields a vector
  *
  * @author Alexander Terenin
  *
  * @param computation the computation containing the needed input that will yield selected result
  */
class GPUVectorResult(computation: GPUComputation) {
  def =:(y: GPUVector): GPUVector = computation match {
    case GPUAlphaXPlusY(x1: GPUVector, x2: GPUVector) => executeSaxpy(x1, x2, y)
    case GPUGeneralMatrixVector(a: GPUMatrix, x: GPUVector) => executeSgemv(a, x, y)
    case GPULeftGeneralMatrixVector(xT: GPUVector, a: GPUMatrix) => executeSgemv(a.T, xT, y, transposeY = true)  // Ax=y is equivalent to y^T = x^T A^T
    case GPUSymmetricMatrixVector(a: GPUSymmetricMatrix, x: GPUVector) => executeSsymv(a, x, y)
    case GPULeftSymmetricMatrixVector(xT: GPUVector, a: GPUSymmetricMatrix) => executeSsymv(a, xT, y, transposeY = true) // xA=y is equivalent to y^T = A x^T since A = A^T
    case GPUTriangularMatrixVector(a: GPUTriangularMatrix, x: GPUVector) => ???
    case GPULeftTriangularMatrixVector(x: GPUVector, a: GPUTriangularMatrix) => ???
    case GPUTriangularSolveVector(ainv: GPUInverseTriangularMatrix, x: GPUVector) => ???
    case GPULeftTriangularSolveVector(x: GPUVector, ainv: GPUInverseTriangularMatrix) => ???
    case GPUPositiveDefiniteTriangularSolve(ainv: GPUInverseSymmetricMatrix, b: GPUMatrix) => ???
    case GPULeftPositiveDefiniteTriangularSolve(b: GPUMatrix, ainv: GPUInverseSymmetricMatrix) => ???
    case _ => throw new Exception("wrong vector operation in =:")
  }

  def +=:(y: GPUVector): GPUVector  = computation match {
    case GPUGeneralMatrixVector(a: GPUMatrix, x: GPUVector) => executeSgemv(a, x, y, inplace = true)
    case GPULeftGeneralMatrixVector(xT: GPUVector, a: GPUMatrix) => executeSgemv(a.T, xT, y, inplace = true, transposeY = true)  // Ax=y is equivalent to y^T = x^T A^T
    case GPUSymmetricMatrixVector(a: GPUSymmetricMatrix, x: GPUVector) => executeSsymv(a, x, y, inplace = true)
    case GPULeftSymmetricMatrixVector(xT: GPUVector, a: GPUSymmetricMatrix) => executeSsymv(a, xT, y, inplace = true, transposeY = true) // xA=y is equivalent to y^T = A x^T since A = A^T
    case _ => throw new Exception("unsupported: cannot execute given vector operation in-place in +=:")
  }

  private implicit class ResultImplicits(v: GPUVector) {
    def setInPlaceAdditionConstant() = if(v.constant.nonEmpty) v else v.mutateConstant(Some(Waterfall.Constants.one))
  }

  private def executeSaxpy(x1: GPUVector, x2: GPUVector, y: GPUVector) = {
    // no need to prepare output - constant and transpose don't affect computation and are mutated at the end

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

  private def executeSgemv(A: GPUMatrix, x: GPUVector, y: GPUVector, inplace: Boolean = false, transposeY: Boolean = false) = {
    // prepare output
    if(!inplace) y.mutateConstant(None) else y.setInPlaceAdditionConstant()
    y.mutateTranspose(false)

    // check for compatibility
    assert(A.numCols == x.length, s"mismatched matrix dimensions: got ${A.numCols} != ${x.length}")
    assert(A.numRows == y.length, s"mismatched vector dimensions: got ${A.numRows} != ${y.length}")
    assert(x.isTranspose == transposeY, s"mismatched vector dimensions: incorrect row/column vector")
    assert(A.constant.isEmpty || x.constant.isEmpty, s"unsupported: only one input constant can be defined")

    // determine constants
    val alpha = A.constant.getOrElse(x.constant.getOrElse(Waterfall.Constants.one))
    val beta = y.constant.getOrElse(Waterfall.Constants.zero)

    // perform single-precision general matrix-vector multiplication
    cublasSgemv(Waterfall.cublasHandle,
      A.isTranspose.toTransposeOpId,
      A.numRows, A.numCols,
      alpha.ptr,
      A.ptr, A.leadingDimension,
      x.ptr, x.stride,
      beta.ptr,
      y.ptr, y.stride
    ).checkJCublasStatus()

    // return result
    y.mutateConstant(None).mutateTranspose(transposeY)
  }

  private def executeSsymv(A: GPUSymmetricMatrix, x: GPUVector, y: GPUVector, inplace: Boolean = false, transposeY: Boolean = false) = {
    // prepare output
    if(!inplace) y.mutateConstant(None) else y.setInPlaceAdditionConstant()
    y.mutateTranspose(false)

    // check for compatibility
    assert(A.size == x.length, s"mismatched matrix dimensions: got ${A.numCols} != ${x.length}")
    assert(x.length == y.length, s"mismatched vector dimensions: got ${x.length} != ${y.length}")
    assert(x.isTranspose == transposeY, s"mismatched vector dimensions: incorrect row/column vector")
    assert(A.constant.isEmpty || x.constant.isEmpty, s"unsupported: only one input constant can be defined")

    // determine constants
    val alpha = A.constant.getOrElse(x.constant.getOrElse(Waterfall.Constants.one))
    val beta = y.constant.getOrElse(Waterfall.Constants.zero)

    // perform single-precision general matrix-vector multiplication
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
}