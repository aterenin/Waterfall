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
  * @param iNumRows number of rows (internally mutable)
  * @param iNumCols number of columns (internally mutable)
  * @param iIsTranspose whether or not matrix is transpose, default false (internally mutable)
  * @param iConstant the optional constant this matrix is multiplied by, default none (internally mutable)
  */
class GPUMatrix(ptr: Pointer,
                private var iNumRows: Int,
                private var iNumCols: Int,
                private var iIsTranspose: Boolean = false,
                private var iConstant: Option[GPUConstant] = None
               ) extends GPUArray(ptr, iNumRows.toLong * iNumCols.toLong) {
  def numRows = iNumRows
  def numCols = iNumCols
  def isTranspose = iIsTranspose
  def constant = iConstant
  def leadingDimension = if(isTranspose) numCols else numRows

  private[waterfall] def mutateConstant(newConstant: Option[GPUConstant]) = { if(constant.nonEmpty || newConstant.nonEmpty) { iConstant = newConstant }; this }
  private[waterfall] def mutateTranspose(newTranspose: Boolean) = { if(newTranspose != isTranspose) { iIsTranspose = newTranspose; iNumRows = numCols; iNumCols = numRows }; this }

  def *(that: GPUMatrix) = new GPUMatrixResult(GPUGeneralMatrixMatrix(this, that))
  def +(that: GPUMatrix) = new GPUMatrixResult(GPUGeneralAddMatrix(this, that))

  def *(that: GPUVector) = new GPUVectorResult(GPUGeneralMatrixVector(this, that))
  def *(that: GPUSymmetricMatrix) = new GPUMatrixResult(GPULeftSymmetricMatrixMatrix(this, that))
  def *(that: GPUTriangularMatrix) = new GPUMatrixResult(GPULeftTriangularMatrixMatrix(this, that))

  def T = new GPUMatrix(ptr, iNumRows = numCols, iNumCols = numRows, iIsTranspose = !isTranspose, constant)

  def withConstant(that: GPUConstant) = new GPUMatrix(ptr, numRows, numCols, isTranspose, iConstant = Option(that))
  def withoutConstant = if(constant.nonEmpty) new GPUMatrix(ptr, numRows, numCols, isTranspose, iConstant = None) else this

  def declareSymmetric = new GPUSymmetricMatrix(ptr, numRows, fillMode = Lower, constant)
  def declareSymmetric(fillMode: FillMode) = new GPUSymmetricMatrix(ptr, numCols, fillMode, constant)

  def declareTriangular = new GPUTriangularMatrix(ptr, numRows, fillMode = Lower, isTranspose, constant)
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