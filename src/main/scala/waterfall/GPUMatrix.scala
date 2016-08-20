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
class GPUMatrix(
                 val ptr: Pointer,
                 val numRows: Int,
                 val numCols: Int,
                 val isTranspose: Boolean = false
               ) {
  val numBytes = numRows.toLong * numCols.toLong * Sizeof.FLOAT.toLong
  val leadingDimension = if(isTranspose) numCols else numRows

  def *(that: GPUMatrix) = new GPUMatrixResult(this, that, GPUgemm)


//  def *(that: GPUVector) = new GPUVectorResult(this, that, gemv)

  def T = new GPUMatrix(ptr, numRows = numCols, numCols = numRows, isTranspose = !isTranspose)

//  def inv = ???

  def copyTo(that: GPUMatrix) = {
    cudaMemcpy(that.ptr, ptr, numBytes, cudaMemcpyDeviceToDevice).checkJCudaStatus()
  }

  def copyToHost: Array[Float] = {
    val result = Array.ofDim[Float](numRows*numCols)
    cudaMemcpy(Pointer.to(result), ptr, numBytes, cudaMemcpyDeviceToHost).checkJCudaStatus()
    result
  }

  def copyToHostBuffer(b: java.nio.FloatBuffer): Unit = {
    cudaMemcpy(Pointer.toBuffer(b), ptr, numBytes, cudaMemcpyDeviceToHost).checkJCudaStatus()
  }
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