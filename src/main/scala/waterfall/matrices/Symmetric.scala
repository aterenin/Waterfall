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

package waterfall.matrices

import waterfall._
import waterfall.matrices.Cholesky.CholeskyWorkspace
import waterfall.matrices.MatrixProperties.{Lower, Upper}

trait Symmetric extends GPUMatrix {
  assert(numRows == numCols, "mismatched matrix dimensions: tried to create a non-square symmetric matrix")
  val fillMode = super.isTranspose match {case true => Lower; case false => Upper}
  val size = numRows
  override val isTranspose = false
  override def T = this
  override def *(that: GPUMatrix) = new GPUMatrixResult(GPUsymm(this, that))
  override def *(that: GPUMatrix with Symmetric) = *(that.asInstanceOf[GPUMatrix])
  override def *(that: GPUVector) = new GPUVectorResult(GPUsymv(this, that))
  def computeCholesky(workspace: CholeskyWorkspace) = ???
  def computeInPlaceCholesky(workspace: CholeskyWorkspace) = ???
}

