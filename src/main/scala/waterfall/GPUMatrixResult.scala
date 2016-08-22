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
import Implicits.TransposeImplicits
import Implicits.DebugImplicits

/**
  * A not-yet-evaluated result of a computation that yields a matrix
  *
  * @author Alexander Terenin
  *
  * @param computation the computation containing the needed input that will yield selected result
  */
class GPUMatrixResult(computation: GPUComputation) {
  def :=>(C: GPUMatrix): GPUMatrix = execute(C)
  def =:(C: GPUMatrix): GPUMatrix = execute(C)
//  def :+=>(C: GPUMatrix) = ???
//  def +=:(C: GPUMatrix) = execute(C)


  private def execute(C: GPUMatrix) = computation match {
    case GPUmaxpy(a) => executeSaxpy(a,C) // somehow, capitals cause an error at compile time - Scala bug?
    case GPUgeam(a,b) => executeSgeam(a,b,C)
    case GPUgemm(a,b) => executeSgemm(a,b,C)
    case _ => throw new Exception("wrong matrix operation in execute(C)")
  }

  private def executeSgeam(A: GPUMatrix, B: GPUMatrix, C: GPUMatrix) = {
    // check for compatibility
    assert(A.numRows == B.numRows, s"mismatched matrix dimensions: got ${A.numRows} != ${B.numRows}")
    assert(B.numRows == C.numRows, s"mismatched matrix dimensions: got ${B.numRows} != ${C.numRows}")
    assert(A.numCols == B.numCols, s"mismatched matrix dimensions: got ${A.numCols} != ${B.numCols}")
    assert(B.numCols == C.numCols, s"mismatched matrix dimensions: got ${B.numCols} != ${C.numCols}")
    assert(!C.isTranspose, s"unsupported: output matrix cannot be transposed, transpose input matrices instead")

    // perform single-precision general matrix-matrix addition
    cublasSgeam(Waterfall.cublasHandle,
      A.isTranspose.toTransposeOp, B.isTranspose.toTransposeOp,
      C.numRows, C.numCols,
      Waterfall.ptrOne,
      A.ptr, A.leadingDimension,
      Waterfall.ptrOne,
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

    // perform single-precision general matrix-matrix multiplication
    cublasSgemm(Waterfall.cublasHandle,
      A.isTranspose.toTransposeOp, B.isTranspose.toTransposeOp,
      C.numRows, C.numCols, A.numCols,
      Waterfall.ptrOne,
      A.ptr, A.leadingDimension,
      B.ptr, B.leadingDimension,
      Waterfall.ptrZero,
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

    // perform in-place matrix addition using single-precision alpha x plus y
    cublasSaxpy(Waterfall.cublasHandle,
      A.size.toInt,
      Waterfall.ptrOne,
      A.ptr, 1,
      C.ptr, 1
    ).checkJCublasStatus()

    // return result
    C
  }
}
