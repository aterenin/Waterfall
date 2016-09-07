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
  * A GPU inverse symmetric matrix (evaluated via Cholesky)
  *
  * @author Alexander Terenin
  *
  * @param inv matrix from which this inverse was created
  */
class GPUInverseSymmetricMatrix(val inv: GPUSymmetricMatrix
                               ) extends GPUInverseMatrix(inv.ptr, inv.size, isTranspose = false) {
  val fillMode = inv.fillMode
  val underlyingCholesky = inv.chol
  def *(that: GPUMatrix) = new GPUMatrixResult(GPUPositiveDefiniteTriangularSolve(this, that))
  def *(that: GPUVector) = new GPUVectorResult(GPUPositiveDefiniteTriangularSolveVector(this, that))
  val T = this
}