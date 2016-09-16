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

import jcuda.driver.CUstream
import jcuda.runtime.cudaStream_t
import jcuda.runtime.JCuda.{cudaStreamCreate, cudaStreamSynchronize}
import Implicits.DebugImplicits

object Stream {
  val default = create

  def create = {
    val cudaStream_t = new cudaStream_t
    val CUstream = new CUstream(cudaStream_t)
    cudaStreamCreate(cudaStream_t).checkJCudaStatus()
    GPUStream(cudaStream_t, CUstream)
  }

  case class GPUStream(cudaStream_t: cudaStream_t, CUstream: CUstream) {
    def synchronize() = cudaStreamSynchronize(cudaStream_t).checkJCudaStatus()
  }
}