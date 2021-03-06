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

import jcuda.jcusolver.JCusolverDn.cusolverDnSpotrf_bufferSize
import jcuda.runtime.JCuda.{cudaMalloc, cudaMemcpyAsync}
import jcuda.{Pointer, Sizeof}
import Implicits.{DebugImplicits, FillModeImplicits}
import jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost
import waterfall.Stream.GPUStream

/**
  * An object containing various matrix properties
  *
  * @author Alexander Terenin
  */
object MatrixProperties {
  sealed trait FillMode
  case object Upper extends FillMode
  case object Lower extends FillMode

  sealed trait Side
  case object Left extends Side
  case object Right extends Side

  case class CholeskyWorkspace(workspace: Pointer, workspaceNumBytes: Int, devInfo: Pointer)
  def createCholeskyWorkspace(A: GPUSymmetricMatrix) = {
    // calculate buffer size
    val workspaceSize = Array.ofDim[Int](1)
    cusolverDnSpotrf_bufferSize(Waterfall.cusolverDnHandle,
      A.fillMode.toFillModeId, A.size, A.ptr, A.leadingDimension,
      workspaceSize).checkJCusolverStatus()
    val workspaceNumBytes = workspaceSize.head * Sizeof.FLOAT

    // allocate workspace
    val workspace = new Pointer
    cudaMalloc(workspace, workspaceNumBytes)
    val devInfo = new Pointer
    cudaMalloc(devInfo, Sizeof.INT)

    // return container with necessary pointers
    CholeskyWorkspace(workspace, workspaceNumBytes, devInfo)
  }
  def checkDevInfo(devInfo: Pointer)(implicit stream: GPUStream = Stream.default): Unit = {
    val result = Array.ofDim[Int](1)
    cudaMemcpyAsync(Pointer.to(result), devInfo, Sizeof.FLOAT, cudaMemcpyDeviceToHost, stream.cudaStream_t).checkJCudaStatus()
    stream.synchronize()
    assert(result.head == 0, s"cuSOLVER devInfo was not zero, got ${result.head}")
  }
}