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

trait Cholesky extends GPUSymmetricMatrix {
  def inv = ???
  def chol = ???
}

object Cholesky {
  case class CholeskyWorkspace(workspace: Pointer, devInfo: Pointer)
  def createWorkspace(A: GPUSymmetricMatrix) = {
    // calculate buffer size
    val workspaceSize = Array.ofDim[Int](1)
    cusolverDnSpotrf_bufferSize(Waterfall.cusolverHandle,
      A.fillMode.toFillModeId, A.size, A.ptr, A.leadingDimension,
      workspaceSize)

    // allocate workspace
    val workspace = new Pointer
    cudaMalloc(workspace, workspaceSize.head)
    val devInfo = new Pointer
    cudaMalloc(devInfo, Sizeof.INT)

    // return container with necessary pointers
    CholeskyWorkspace(workspace, devInfo)
  }
}
