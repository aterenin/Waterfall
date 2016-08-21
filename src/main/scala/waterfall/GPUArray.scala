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

import jcuda.runtime.JCuda._
import jcuda.runtime.cudaMemcpyKind._
import jcuda.{Pointer, Sizeof}
import Implicits.DebugImplicits

/**
  * A GPU array
  *
  * @author Alexander Terenin
  * @param ptr JCuda pointer to the GPU's array
  * @param length length of array
  */
class GPUArray(val ptr: Pointer,
               val length: Long
              ) {
  val numBytes = length * Sizeof.FLOAT.toLong

  def copyTo(that: GPUArray) = {
    assert(this.length == that.length, "tried to copy into array of non-matching length")
    cudaMemcpy(that.ptr, ptr, numBytes, cudaMemcpyDeviceToDevice).checkJCudaStatus()
  }

  def copyToHost: Array[Float] = {
    assert(length < Int.MaxValue, "array too big to store on host, length > Int.MaxValue")
    val result = Array.ofDim[Float](length.toInt)
    cudaMemcpy(Pointer.to(result), ptr, numBytes, cudaMemcpyDeviceToHost).checkJCudaStatus()
    result
  }

  def copyToHostBuffer(b: java.nio.FloatBuffer): Unit = {
    cudaMemcpy(Pointer.toBuffer(b), ptr, numBytes, cudaMemcpyDeviceToHost).checkJCudaStatus()
  }
}
