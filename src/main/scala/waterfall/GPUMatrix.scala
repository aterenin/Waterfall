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

import java.nio.FloatBuffer

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
  * @param numRows number of rows
  * @param numCols number of columns
  * @param isTranspose whether or not matrix is transpose, default false
  */
class GPUMatrix(ptr: Pointer,
                val numRows: Int,
                val numCols: Int,
                val isTranspose: Boolean = false,
                val constant: Option[GPUConstant] = None
               ) extends GPUArray(ptr, numRows.toLong * numCols.toLong) {
  val leadingDimension = if(isTranspose) numCols else numRows

  def *(that: GPUMatrix) = new GPUMatrixResult(GPUGeneralMatrixMatrix(this, that))
  def +(that: GPUMatrix) = new GPUMatrixResult(GPUGeneralAddMatrix(this, that))

  def *(that: GPUVector) = new GPUVectorResult(GPUGeneralMatrixVector(this, that))
  def *(that: GPUSymmetricMatrix) = new GPUMatrixResult(GPULeftSymmetricMatrixMatrix(this, that))
  def *(that: GPUTriangularMatrix) = new GPUMatrixResult(GPULeftTriangularMatrixMatrix(this, that))

  // switch order of arguments due to Scala operator order reversal for operators with : in them
  def +=:(that: GPUMatrix) = new GPUMatrixResult(GPUMatrixAlphaXPlusY(this)) :=> that

  def T = new GPUMatrix(ptr, numRows = numCols, numCols = numRows, isTranspose = !isTranspose, constant)

  def withConstant(that: GPUConstant) = new GPUMatrix(ptr, numRows, numCols, isTranspose, constant = Option(that))
  def withoutConstant = new GPUMatrix(ptr, numRows, numCols, isTranspose, constant = None)

  def declareSymmetric = new GPUSymmetricMatrix(ptr, numRows, Lower, constant)
  def declareSymmetric(fillMode: FillMode) = new GPUSymmetricMatrix(ptr, numCols, fillMode, constant)

  def declareTriangular = new GPUTriangularMatrix(ptr, numRows, Lower, isTranspose, constant)
  def declareTriangular(fillMode: FillMode) = new GPUTriangularMatrix(ptr, numCols, fillMode, isTranspose, constant)
}

object GPUMatrix {
  def create(numRows: Int, numCols: Int): GPUMatrix = {
    val ptr = new Pointer
    val numBytes = numRows.toLong * numCols.toLong * Sizeof.FLOAT.toLong
    cudaMalloc(ptr, numBytes).checkJCudaStatus()
    new GPUMatrix(ptr, numRows, numCols)
  }


  def createFromColumnMajorArray(data: Array[Array[Float]]) = {
    val numRows = data.head.length
    val numCols = data.length
    val M = create(numRows, numCols)
    cudaMemcpy(M.ptr, Pointer.to(data.flatten), M.numBytes, cudaMemcpyHostToDevice).checkJCudaStatus()
    M
  }

  def createFromBuffer(data: FloatBuffer, numRows: Int, numCols: Int) = {

  }

}