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
import MatrixProperties._

/**
  * A GPU triangular matrix
  *
  * @author Alexander Terenin
  *
  * @param ptr JCuda pointer to the GPU's array
  * @param size number of rows and columns
  * @param fillMode whether the matrix has upper or lower fill mode
  * @param isTranspose whether or not the matrix is transposed
  * @param const the optional constant this vector is multiplied by, default none
  */
class GPUTriangularMatrix(ptr: Pointer,
                          val size: Int,
                          val fillMode: FillMode,
                          isTranspose: Boolean = false,
                          const: Option[GPUConstant] = None
                         ) extends GPUMatrix(ptr, iNumRows=size, iNumCols=size, iIsTranspose=isTranspose, const) {
  override def T = super.T.declareTriangular(fillMode)

  def inv = {
    assert(constant.isEmpty, s"unsupported: cannot invert matrix with attached constant")
    new GPUInverseTriangularMatrix(this, isTranspose)
  }

  override def +(that: GPUMatrix) = throw new Exception("warning: the other side of a triangular matrix may contain garbage rather than zeros - if you really need addition, cast to GPUMatrix first")
  override def *(that: GPUMatrix) = new GPUMatrixResult(GPUTriangularMatrixMatrix(this, that))
  override def *(that: GPUSymmetricMatrix) = new GPUMatrixResult(GPUTriangularMatrixMatrix(this, that))
  override def *(that: GPUVector) = new GPUVectorResult(GPUTriangularMatrixVector(this, that))

  override def withConstant(that: GPUConstant) = super.withConstant(that).declareTriangular(fillMode)
  override def withoutConstant = super.withoutConstant.declareTriangular(fillMode)
}