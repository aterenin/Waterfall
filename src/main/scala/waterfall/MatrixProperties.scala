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
import jcuda.runtime.JCuda.cudaMalloc
import jcuda.{Pointer, Sizeof}
import Implicits.FillModeImplicits

object MatrixProperties {
  sealed trait FillMode
  case object Upper extends FillMode
  case object Lower extends FillMode

  sealed trait Side
  case object Left extends Side
  case object Right extends Side

  case class CholeskyWorkspace(workspace: Pointer, workspaceSize: Int, devInfo: Pointer)
  def createCholeskyWorkspace(A: GPUSymmetricMatrix) = {
    // calculate buffer size
    val workspaceSize = Array.ofDim[Int](1)
    cusolverDnSpotrf_bufferSize(Waterfall.cusolverDnHandle,
      A.fillMode.toFillModeId, A.size, A.ptr, A.leadingDimension,
      workspaceSize)

    // allocate workspace
    val workspace = new Pointer
    cudaMalloc(workspace, workspaceSize.head)
    val devInfo = new Pointer
    cudaMalloc(devInfo, Sizeof.INT)

    // return container with necessary pointers
    CholeskyWorkspace(workspace, workspaceSize.head, devInfo)
  }
}