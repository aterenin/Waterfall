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
import MatrixProperties.{CholeskyWorkspace, FillMode}

class GPUSymmetricMatrix(ptr: Pointer,
                         val size: Int,
                         val fillMode: FillMode,
                         const: Option[GPUConstant] = None,
                         private var iCholesky: Option[GPUTriangularMatrix] = None
                        ) extends GPUMatrix(ptr, iNumRows=size, iNumCols=size, iIsTranspose=false, const) {
  override def mutateTranspose(newTranspose: Boolean) = this
  override val isTranspose = false
  override val T = this

  override def withConstant(that: GPUConstant) = super.withConstant(that).declareSymmetric(fillMode)
  override def withoutConstant = super.withoutConstant.declareSymmetric(fillMode)

  override def *(that: GPUMatrix) = new GPUMatrixResult(GPUSymmetricMatrixMatrix(this, that))
  override def *(that: GPUVector) = new GPUVectorResult(GPUSymmetricMatrixVector(this, that))

  def chol = iCholesky.getOrElse(throw new Exception(s"tried to get Cholesky decomposition, but none attached"))
  def inv = ???

  def hasCholesky = iCholesky.nonEmpty

  def computeCholesky(workspace: CholeskyWorkspace) = {
    assert(constant.isEmpty, s"unsupported: cannot compute Cholesky for matrix with attached constant")
    new GPUMatrixResult(GPUPositiveDefiniteTriangularFactorize(this, workspace))
  }
  def attachCholesky(chol: GPUTriangularMatrix, workspace: CholeskyWorkspace) = { iCholesky = Some(chol); this }
}

