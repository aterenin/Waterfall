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
import Implicits.{DebugImplicits, TransposeImplicits, FillModeImplicits, SideImplicits}
import MatrixProperties.{Left, Right}

/**
  * A not-yet-evaluated result of a computation that yields a matrix
  *
  * @author Alexander Terenin
  *
  * @param computation the computation containing the needed input that will yield selected result
  */
class GPUMatrixResult(computation: GPUComputation) {
  def :=>(C: GPUMatrix): GPUMatrix = execute(C.withoutConstant)
  def =:(C: GPUMatrix): GPUMatrix = execute(C.withoutConstant)

  def :+=>(C: GPUMatrix): GPUMatrix = execute(C.withConstant(Waterfall.Constants.one))
  def +=:(C: GPUMatrix): GPUMatrix  = execute(C.withConstant(Waterfall.Constants.one))

  def :++=>(C: GPUMatrix): GPUMatrix  = execute(C)
  def ++=:(C: GPUMatrix): GPUMatrix  = execute(C)

  private implicit class Checks(M: GPUMatrix) {
    def checkNoTranspose = {
      assert(!M.isTranspose, s"unsupported: output matrix cannot already be transposed")
      M
    }
  }

  private def execute(C: GPUMatrix) = computation match {
    case GPUMatrixAlphaXPlusY(a) => executeSaxpy(a,C) // somehow, capitals cause an error at compile time - Scala bug?
    case GPUGeneralAddMatrix(a,b) => executeSgeam(a,b,C)
    case GPUGeneralMatrixMatrix(a,b) => executeSgemm(a,b,C)
    case GPUSymmetricMatrixMatrix(a,b) if !b.isTranspose => executeSsymm(a,b,C.checkNoTranspose)
    case GPUSymmetricMatrixMatrix(a,b) if b.isTranspose => executeSsymm(b.T,a,C.checkNoTranspose.T)
    case GPULeftSymmetricMatrixMatrix(b,a) if !b.isTranspose => executeSsymm(b,a,C.checkNoTranspose)
    case GPULeftSymmetricMatrixMatrix(b,a) if b.isTranspose => executeSsymm(a,b.T,C.checkNoTranspose.T)
    case _ => throw new Exception("wrong matrix operation in execute(C)")
  }

  private def executeSgeam(A: GPUMatrix, B: GPUMatrix, C: GPUMatrix) = {
    // check for compatibility
    assert(A.numRows == B.numRows, s"mismatched matrix dimensions: got ${A.numRows} != ${B.numRows}")
    assert(B.numRows == C.numRows, s"mismatched matrix dimensions: got ${B.numRows} != ${C.numRows}")
    assert(A.numCols == B.numCols, s"mismatched matrix dimensions: got ${A.numCols} != ${B.numCols}")
    assert(B.numCols == C.numCols, s"mismatched matrix dimensions: got ${B.numCols} != ${C.numCols}")
    assert(!C.isTranspose, s"unsupported: output matrix cannot be transposed, transpose input matrices instead")

    // determine constants
    val alpha = A.constant.getOrElse(Waterfall.Constants.one)
    val beta = B.constant.getOrElse(Waterfall.Constants.one)

    // perform single-precision general matrix-matrix addition
    cublasSgeam(Waterfall.cublasHandle,
      A.isTranspose.toTransposeOpId, B.isTranspose.toTransposeOpId,
      C.numRows, C.numCols,
      alpha.ptr,
      A.ptr, A.leadingDimension,
      beta.ptr,
      B.ptr, B.leadingDimension,
      C.ptr, C.leadingDimension
    ).checkJCublasStatus()

    // return result
    C
  }

  private def executeSgemm(A: GPUMatrix, B: GPUMatrix, C: GPUMatrix) = {
    // check for compatibility
    assert(A.numRows == C.numRows, s"mismatched matrix dimensions: got ${A.numRows} != ${C.numRows}")
    assert(B.numCols == C.numCols, s"mismatched matrix dimensions: got ${B.numCols} != ${C.numCols}")
    assert(A.numRows == B.numCols, s"mismatched matrix dimensions: got ${A.numRows} != ${B.numCols}")
    assert(!C.isTranspose, s"unsupported: output matrix cannot be transposed, transpose input matrices instead")
    assert(A.constant.isEmpty || B.constant.isEmpty, s"unsupported: only one input constant can be defined")

    // determine constants
    val alpha = A.constant.getOrElse(B.constant.getOrElse(Waterfall.Constants.one))
    val beta = C.constant.getOrElse(Waterfall.Constants.zero)

    // perform single-precision general matrix-matrix multiplication
    cublasSgemm(Waterfall.cublasHandle,
      A.isTranspose.toTransposeOpId, B.isTranspose.toTransposeOpId,
      C.numRows, C.numCols, A.numCols,
      alpha.ptr,
      A.ptr, A.leadingDimension,
      B.ptr, B.leadingDimension,
      beta.ptr,
      C.ptr, C.leadingDimension
    ).checkJCublasStatus()

    // return result
    C
  }

  private def executeSaxpy(A: GPUMatrix, C: GPUMatrix) = {
    // check for compatibility
    assert(A.numRows == C.numRows, s"mismatched matrix dimensions: got ${A.numRows} != ${C.numRows}")
    assert(A.numCols == C.numCols, s"mismatched matrix dimensions: got ${A.numCols} != ${C.numCols}")
    assert(A.isTranspose == C.isTranspose, s"unsupported: A,B must have same transpose flag for in-place addition")
    assert(A.numElements < Int.MaxValue.toLong, s"unsupported: array size bigger than Int.MaxValue")
    assert(C.constant.isEmpty, s"unsupported: output matrix must not have constant")

    // determine constants
    val alpha = A.constant.getOrElse(Waterfall.Constants.one)

    // perform in-place matrix addition using single-precision alpha x plus y
    cublasSaxpy(Waterfall.cublasHandle,
      A.numElements.toInt,
      alpha.ptr,
      A.ptr, 1,
      C.ptr, 1
    ).checkJCublasStatus()

    // return result
    C
  }

  private def executeSsymm(M: GPUMatrix, N: GPUMatrix, C: GPUMatrix) = {
    //determine parameters and check for compatibility
    val (a, b, side) = (M,N) match {
      case (b: GPUMatrix, a: GPUSymmetricMatrix) =>
        assert(a.size == b.numRows, s"mismatched matrix dimensions: got ${a.size} != ${b.numRows}")
        assert(a.size == C.numRows, s"mismatched matrix dimensions: got ${a.numRows} != ${C.numRows}")
        assert(b.numCols == C.numCols, s"mismatched matrix dimensions: got ${b.numCols} != ${C.numCols}")
        (a, b, Right)
      case (a: GPUSymmetricMatrix, b: GPUMatrix) =>
        assert(b.numCols == a.size, s"mismatched matrix dimensions: got ${b.numCols} != ${a.size}")
        assert(a.size == C.numCols, s"mismatched matrix dimensions: got ${a.size} != ${C.numCols}")
        assert(b.numRows == C.numRows, s"mismatched matrix dimensions: got ${b.numRows} != ${C.numRows}")
        (a, b, Left)
    }
    assert(a.constant.isEmpty || b.constant.isEmpty, s"unsupported: only one input constant can be defined")

    // determine constants
    val alpha = a.constant.getOrElse(b.constant.getOrElse(Waterfall.Constants.one))
    val beta = C.constant.getOrElse(Waterfall.Constants.zero)

    // perform single-precision general matrix-matrix multiplication
    cublasSsymm(Waterfall.cublasHandle,
      side.toSideId, a.fillMode.toFillModeId,
      C.numRows, C.numCols,
      alpha.ptr,
      a.ptr, a.leadingDimension,
      b.ptr, b.leadingDimension,
      beta.ptr,
      C.ptr, C.leadingDimension
    ).checkJCublasStatus()

    // return result
    C
  }
}
