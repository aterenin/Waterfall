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
  def :=>(y: GPUVector): GPUVector = execute(y.withoutConstant)
  def =:(y: GPUVector): GPUVector = execute(y.withoutConstant)

  def :+=>(y: GPUVector): GPUVector = execute(y.withConstant(Waterfall.Constants.one))
  def +=:(y: GPUVector): GPUVector  = execute(y.withConstant(Waterfall.Constants.one))

  def :++=>(y: GPUVector): GPUVector  = execute(y)
  def ++=:(y: GPUVector): GPUVector  = execute(y)

  private def execute(y: GPUVector) = computation match {
    case GPUAlphaXPlusY(x: GPUVector) => executeSaxpy(x, y)
    case GPUGeneralMatrixVector(a: GPUMatrix, x: GPUVector) => executeSgemv(a, x, y)
    case GPULeftGeneralMatrixVector(x: GPUVector, a: GPUMatrix) => executeSgemv(a.T, x.T, y.T)  // Ax=y is equivalent to y^T = x^T A^T
    case GPUSymmetricMatrixVector(a: GPUSymmetricMatrix, x: GPUVector) => executeSsymv(a, x, y)
    case GPULeftSymmetricMatrixVector(x: GPUVector, a: GPUSymmetricMatrix) => executeSsymv(a, x.T, y.T) // see lgemv, and note A^T = A
    case _ => throw new Exception("wrong vector operation in execute()")
  }

  private def executeSaxpy(x: GPUVector, y: GPUVector) = {
    // check for compatibility
    assert(x.length == y.length, s"mismatched vector dimensions: got ${x.length} != ${y.length}")
    assert(x.isTranspose == y.isTranspose, s"mismatched vector dimensions: tried to add row vector to column vector")
    assert(y.constant.isEmpty, s"unsupported: output vector must not have constant")

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
    y
  }

  private def executeSgemv(A: GPUMatrix, x: GPUVector, y: GPUVector) = {
    // check for compatibility
    assert(A.numCols == x.length, s"mismatched matrix dimensions: got ${A.numCols} != ${x.length}")
    assert(x.length == y.length, s"mismatched vector dimensions: got ${x.length} != ${y.length}")
    assert(x.isTranspose == y.isTranspose, s"mismatched vector dimensions: incorrect row/column vector")
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
    y
  }

  private def executeSsymv(A: GPUSymmetricMatrix, x: GPUVector, y: GPUVector) = {
    // check for compatibility
    assert(A.size == x.length, s"mismatched matrix dimensions: got ${A.numCols} != ${x.length}")
    assert(x.length == y.length, s"mismatched vector dimensions: got ${x.length} != ${y.length}")
    assert(x.isTranspose == y.isTranspose, s"mismatched vector dimensions: incorrect row/column vector")
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
    y
  }

  private def executeStrmv(A: GPUTriangularMatrix, x: GPUVector) = {
    // check for compatibility
    assert(A.size == x.length, s"mismatched matrix dimensions: got ${A.numCols} != ${x.length}")
    assert(x.length == y.length, s"mismatched vector dimensions: got ${x.length} != ${y.length}")
    assert(x.isTranspose == y.isTranspose, s"mismatched vector dimensions: incorrect row/column vector")
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
    y
  }
}