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
import Implicits.DebugImplicits
import waterfall.Stream.GPUStream

/**
  * A not-yet-evaluated result of a computation that yields a constant
  *
  * @author Alexander Terenin
  *
  * @param computation the computation containing the needed input that will yield selected result
  */
class GPUConstantResult(computation: GPUComputation) {
  def :=>(c: GPUConstant): GPUConstant = execute(c)
  def =:(c: GPUConstant): GPUConstant = execute(c)

  private def execute(c: GPUConstant) = computation match {
    case GPUDot(x: GPUVector, y: GPUVector) => executeSdot(x, y, c)
    case _ => throw new Exception("wrong constant operation in execute()")
  }

  private def executeSdot(x: GPUVector, y: GPUVector, c: GPUConstant)(implicit stream: GPUStream = Stream.default) = {
    // check for compatibility
    assert(x.length == y.length, s"mismatched vector dimensions: got ${x.length} != ${y.length}")
    assert(x.isTranspose == y.isTranspose, s"mismatched vector dimensions: tried to dot row vector with column vector")
    assert(x.constant.isEmpty && y.constant.isEmpty, s"unsupported: input must not have constants")

    // set stream
    cublasSetStream(Waterfall.cublasHandle, stream.cudaStream_t).checkJCublasStatus()

    // perform dot product
    cublasSdot(Waterfall.cublasHandle,
      x.length,
      x.ptr, x.stride,
      y.ptr, y.stride,
      c.ptr
    ).checkJCublasStatus()

    // return result
    c
  }
}