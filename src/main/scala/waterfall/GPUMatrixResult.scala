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
  * Created by Admin on 8/19/16.
  */
class GPUMatrixResult(A: GPUMatrix, B: GPUMatrix, computation: GPUComputation) {
  def :=>(C: GPUMatrix) = execute(C)
  def =:(C: GPUMatrix) = execute(C)
//  def :+=>(C: GPUMatrix) = ???
//  def +=:(C: GPUMatrix) = execute(C)

  private def execute(C: GPUMatrix) = computation match {
    case GPUgemm => executeSgemm(C: GPUMatrix)
    case GPUaxpy => executeSaxpy(C: GPUMatrix)
    case _ => throw new Exception("unsupported matrix operation")
  }

  private def executeSgemm(C: GPUMatrix) = {
    // check for compatibility
    assert(A.numRows == C.numRows, s"mismatched matrix dimensions in *: got ${A.numRows} != ${C.numRows}")
    assert(B.numCols == C.numCols, s"mismatched matrix dimensions in *: got ${B.numCols} != ${C.numCols}")
    assert(A.numRows == B.numCols, s"mismatched matrix dimensions in *: got ${A.numRows} != ${B.numCols}")

    println(A.isTranspose)
    println(A.numRows)
    println(A.numCols)
    println(B.isTranspose)
    println(B.numRows)
    println(B.numCols)

    // perform single-precision general matrix-matrix multiplication
    cublasSgemm(Waterfall.cublasHandle,
      A.isTranspose.toTransposeOp, B.isTranspose.toTransposeOp,
      C.numRows, C.numCols, A.numCols,
      Waterfall.ptrOne,
      A.ptr, A.leadingDimension,
      B.ptr, B.leadingDimension,
      Waterfall.ptrZero,
      C.ptr, C.leadingDimension)
      .checkJCublasStatus()

    // return result
    C
  }

  private def executeSaxpy(C: GPUMatrix) = {
    C
  }
}
