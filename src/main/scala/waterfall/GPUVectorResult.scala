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
  * A not-yet-evaluated result of a computation that yields a vector
  *
  * @author Alexander Terenin
  *
  * @param computation the computation containing the needed input that will yield selected result
  */
class GPUVectorResult(computation: GPUComputation) {
  def :=>(y: GPUVector): GPUVector = execute(y)
  def =:(y: GPUVector): GPUVector = execute(y)
  //  def :+=>(C: GPUMatrix) = ???
  //  def +=:(C: GPUMatrix) = execute(C)

  private def execute(y: GPUVector): GPUVector = computation match {
    case GPUaxpy(x: GPUVector) => executeSaxpy(x, y)
    case GPUgemv(a: GPUMatrix, x: GPUVector) => executeSgemv(a, x, y)
    case GPUlgemv(x: GPUVector, a: GPUMatrix) => executeSgemv(a, x, y, yT = true)
    case _ => throw new Exception("wrong vector operation in execute()")
  }

  private def executeSaxpy(x: GPUVector, y: GPUVector) = {
    // check for compatibility
    assert(x.length == y.length, s"mismatched matrix dimensions: got ${x.length} != ${y.length}")
    assert(x.isTranspose == y.isTranspose, s"mismatched vector dimensions: tried to add row vector to column vector")

    // perform in-place matrix addition using single-precision alpha x plus y
    cublasSaxpy(Waterfall.cublasHandle,
      x.length,
      Waterfall.ptrOne,
      x.ptr, x.stride,
      y.ptr, y.stride
    ).checkJCublasStatus()

    // return result
    y
  }

  private def executeSgemv(A: GPUMatrix, x: GPUVector, y: GPUVector, yT: Boolean = false) = {
    // check for compatibility
    assert(A.numCols == x.length, s"mismatched matrix dimensions: got ${A.numCols} != ${x.length}")
    assert(x.length == y.length, s"mismatched vector dimensions: got ${x.length} != ${y.length}")
    assert(!x.isTranspose, s"mismatched vector dimensions: must be column vectors")
    assert(!y.isTranspose, s"mismatched vector dimensions: must be column vectors")

    // perform single-precision general matrix-vector multiplication
    cublasSgemv(Waterfall.cublasHandle,
      A.isTranspose.toTransposeOp,
      A.numRows, A.numCols,
      Waterfall.ptrOne,
      A.ptr, A.leadingDimension,
      x.ptr, x.stride,
      Waterfall.ptrZero,
      y.ptr, y.stride
    ).checkJCublasStatus()

    // return result
    if(yT) y.T else y
  }
}