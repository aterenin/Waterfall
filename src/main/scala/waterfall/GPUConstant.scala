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


import jcuda.runtime.JCuda.{cudaMalloc, cudaMemcpyAsync}
import jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice
import jcuda.{Pointer, Sizeof}
import Implicits.DebugImplicits
import waterfall.Stream.GPUStream

/**
  * A GPU constant
  *
  * @author Alexander Terenin
  *
  * @param ptr JCuda pointer to the GPU's array
  */
class GPUConstant(ptr: Pointer) extends GPUArray(ptr, 1L) {
  def *(that: GPUMatrix) = that.withConstant(this)
  def *(that: GPUVector) = that.withConstant(this)
}

object GPUConstant {
  def create(v: Float)(implicit stream: GPUStream = Stream.default): GPUConstant = {
    val ptr = new Pointer
    cudaMalloc(ptr, Sizeof.FLOAT).checkJCudaStatus()
    cudaMemcpyAsync(ptr, Pointer.to(Array(v)), Sizeof.FLOAT, cudaMemcpyHostToDevice, stream.cudaStream_t).checkJCudaStatus()
    stream.synchronize()
    new GPUConstant(ptr)
  }
}