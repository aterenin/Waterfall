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

/**
  * A GPU inverse triangular matrix
  *
  * @author Alexander Terenin
  *
  * @param inv matrix from which this inverse was created
  * @param isTranspose indicates whether transpose or not (no default to ensure it is set to same as inv on creation)
  */
class GPUInverseTriangularMatrix(val inv: GPUTriangularMatrix,
                                 isTranspose: Boolean
                                ) extends GPUInverseMatrix(inv.ptr, inv.size, isTranspose) {
  val fillMode = inv.fillMode
  def *(that: GPUMatrix) = new GPUMatrixResult(GPUTriangularSolveMatrix(this, that))
  def *(that: GPUVector) = new GPUVectorResult(GPUTriangularSolveVector(this, that))
  val T = new GPUInverseTriangularMatrix(inv, isTranspose = !isTranspose)
}