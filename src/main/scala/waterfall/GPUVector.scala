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
  * @param length number of rows
  * @param isTranspose whether or not vector is transposed, default false (column vector)
  */
class GPUVector(ptr: Pointer,
                val length: Int,
                val isTranspose: Boolean = false,
                val stride: Int = 1
               ) extends GPUArray(ptr, length.toLong) {


  // due to Scala operator order reversal for operators with : in them, that needs to be mutated, and this doesn't
  def +=:(that: GPUVector) = new GPUVectorResult(GPUaxpy(this)) :=> that

  def *(that: GPUMatrix) = {
    // Ax=y is equivalent to y^T = x^T A^T
    assert(isTranspose, s"mismatched vector dimensions: must be column vectors")
    new GPUVectorResult(GPUlgemv(this.T, that.T))
  }

  def T = new GPUVector(ptr, length = length, isTranspose = !isTranspose)

  def toGPUMatrix = new GPUMatrix(ptr, length, 1, isTranspose)

  //  def performTranspose = ???
  //  def inv = ???
}

object GPUVector {
  def create(length: Int): GPUVector = {
    val ptr = new Pointer
    val numBytes = length.toLong * Sizeof.FLOAT.toLong
    cudaMalloc(ptr, numBytes).checkJCudaStatus()
    new GPUVector(ptr, length)
  }


  def createFromArray(data: Array[Float]) = {
    val M = create(data.length)
    cudaMemcpy(M.ptr, Pointer.to(data), M.numBytes, cudaMemcpyHostToDevice).checkJCudaStatus()
    M
  }

  def createFromBuffer(data: FloatBuffer, numRows: Int, numCols: Int) = {

  }

}