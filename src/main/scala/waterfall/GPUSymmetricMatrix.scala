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

import jcuda.Pointer
import Cholesky.CholeskyWorkspace
import MatrixProperties.FillMode

class GPUSymmetricMatrix(ptr: Pointer,
                         val size: Int,
                         val fillMode: FillMode,
                         constant: Option[GPUConstant] = None
                        ) extends GPUMatrix(ptr, numRows=size, numCols=size, isTranspose=false, constant) {
  override val isTranspose = false
  override def T = this

  override def *(that: GPUMatrix) = new GPUMatrixResult(GPUSymmetricMatrixMatrix(this, that))
  override def *(that: GPUVector) = new GPUVectorResult(GPUSymmetricMatrixVector(this, that))

  override def withConstant(that: GPUConstant) = super.withConstant(that).declareSymmetric(fillMode)
  override def withoutConstant = super.withoutConstant.declareSymmetric(fillMode)

  def computeCholesky(workspace: CholeskyWorkspace) = ???
  def computeInPlaceCholesky(workspace: CholeskyWorkspace) = ???
}

