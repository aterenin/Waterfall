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

import jcuda.runtime.JCuda.{cudaMalloc, cudaMemcpy}
import jcuda.runtime.cudaMemcpyKind.{cudaMemcpyDeviceToDevice, cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice}
import jcuda.{Pointer, Sizeof}
import Implicits.DebugImplicits
import MatrixProperties.{FillMode, Lower}

/**
  * A GPU matrix
  *
  * @author Alexander Terenin
  *
  * @param ptr JCuda pointer to the GPU's array
  * @param size number of rows and number of columns
  * @param isTranspose whether or not matrix is transpose, default false
  */
abstract class GPUInverseMatrix(ptr: Pointer,
                                val size: Int,
                                val isTranspose: Boolean = false
                               ) extends GPUArray(ptr, size.toLong * size.toLong) {
  val leadingDimension = size

  def *(that: GPUMatrix): GPUMatrixResult //= new GPUMatrixResult(GPUGeneralMatrixMatrix(this, that))
  def *(that: GPUVector): GPUVectorResult
  def T: GPUInverseMatrix //= new GPUInverseMatrix(ptr, numRows = numCols, numCols = numRows, isTranspose = !isTranspose)
}