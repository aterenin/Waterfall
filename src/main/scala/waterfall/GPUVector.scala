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

import jcuda.runtime.JCuda.{cudaMalloc, cudaMemcpyAsync}
import jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice
import jcuda.{Pointer, Sizeof}
import Implicits.DebugImplicits
import waterfall.Stream.GPUStream

/**
  * A GPU matrix
  *
  * @author Alexander Terenin
  *
  * @param ptr JCuda pointer to the GPU's array
  * @param length number of rows
  * @param iIsTranspose whether or not vector is transposed, default false yielding column vectors (internally mutable)
  * @param iConstant the optional constant this vector is multiplied by, default none (internally mutable)
  */
class GPUVector(ptr: Pointer,
                val length: Int,
                private var iIsTranspose: Boolean = false,
                private var iConstant: Option[GPUConstant] = None,
                val stride: Int = 1
               ) extends GPUArray(ptr, length.toLong) {
  def isTranspose = iIsTranspose
  def constant = iConstant

  private[waterfall] def mutateConstant(newConstant: Option[GPUConstant]) = { iConstant = newConstant; this }
  private[waterfall] def mutateTranspose(newTranspose: Boolean) = { iIsTranspose = newTranspose; this }

  def +(that: GPUVector) = new GPUVectorResult(GPUAlphaXPlusY(this, that))
  def *(that: GPUMatrix) = new GPUVectorResult(GPULeftGeneralMatrixVector(this, that))
  def *(that: GPUSymmetricMatrix) = new GPUVectorResult(GPULeftSymmetricMatrixVector(this, that))
  def *(that: GPUTriangularMatrix) = new GPUVectorResult(GPULeftTriangularMatrixVector(this, that))
  def *(that: GPUInverseSymmetricMatrix) = new GPUVectorResult(GPULeftPositiveDefiniteTriangularSolveVector(this, that))
  def *(that: GPUInverseTriangularMatrix) = new GPUVectorResult(GPULeftTriangularSolveVector(this, that))

  // won't have desired order of operations - Scala limitation
  def dot(that: GPUVector) = new GPUConstantResult(GPUDot(this, that))
  def outer(that: GPUVector) = this.asColumnVector.asGPUMatrix * that.asRowVector.asGPUMatrix

  def T = new GPUVector(ptr,  length, iIsTranspose = !isTranspose, constant, stride)
  def asColumnVector = new GPUVector(ptr,  length, iIsTranspose = false, constant, stride)
  def asRowVector = new GPUVector(ptr, length, iIsTranspose = true, constant, stride)

  def withConstant(that: GPUConstant) = new GPUVector(ptr, length, isTranspose, iConstant = Option(that), stride)
  def withoutConstant = if(constant.nonEmpty) new GPUVector(ptr, length, isTranspose, iConstant = None, stride) else this

  def asGPUMatrix = new GPUMatrix(ptr, if(isTranspose) 1 else length, if(isTranspose) length else 1)
}

object GPUVector {
  def create(length: Int): GPUVector = {
    val ptr = new Pointer
    val numBytes = length.toLong * Sizeof.FLOAT.toLong
    cudaMalloc(ptr, numBytes).checkJCudaStatus()
    new GPUVector(ptr, length)
  }


  def createFromArray(data: Array[Float], async: Boolean = false)(implicit stream: GPUStream = Stream.default) = {
    val M = create(data.length)
    cudaMemcpyAsync(M.ptr, Pointer.to(data), M.numBytes, cudaMemcpyHostToDevice, stream.cudaStream_t).checkJCudaStatus()
    if(!async) stream.synchronize()
    M
  }

  def createFromBuffer(data: FloatBuffer, numRows: Int, numCols: Int)(implicit stream: GPUStream = Stream.default) = ???
}