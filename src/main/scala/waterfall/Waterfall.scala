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

import jcuda.jcublas.JCublas2.{cublasCreate, cublasSetPointerMode}
import jcuda.jcublas.cublasHandle
import jcuda.jcublas.cublasPointerMode.CUBLAS_POINTER_MODE_DEVICE
import jcuda.jcurand.JCurand.curandCreateGenerator
import jcuda.jcurand.curandGenerator
import jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_PHILOX4_32_10
import jcuda.jcusolver.JCusolverDn.cusolverDnCreate
import jcuda.jcusolver.cusolverDnHandle
import jcuda.runtime.JCuda.{cudaMalloc, cudaMemcpy}
import jcuda.runtime.cudaMemcpyKind.{cudaMemcpyHostToDevice}
import jcuda.{Pointer, Sizeof}
import Implicits.DebugImplicits

/**
  * The main Waterfall object - stores handles for CUDA libraries, initialization and cleanup code, utilities, etc.
  *
  * Mutable variables: isInitialized
  *
  * @author Alexander Terenin
  */
object Waterfall {
  val curandGenerator = new curandGenerator()
  val cublasHandle = new cublasHandle()
  val cusolverDnHandle = new cusolverDnHandle


  private var initialized = false
  def isInitialized = initialized

  def init() = if(!initialized) {
    curandCreateGenerator(curandGenerator, CURAND_RNG_PSEUDO_PHILOX4_32_10).checkJCurandStatus()

    cublasCreate(cublasHandle).checkJCublasStatus()
    cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE).checkJCublasStatus()

    cusolverDnCreate(cusolverDnHandle).checkJCusolverStatus()

    initialized = true
  } else throw new Exception("Waterfall.init() called, but was already previously initialized")

  object Constants {
    lazy val one = GPUConstant.create(1.0f)
    lazy val zero = GPUConstant.create(0.0f)
    lazy val minusOne = GPUConstant.create(-1.0f)
  }

  object SizeOf {
    val sizeOfCurandStateXORWOW = 48L //somehow, nowhere to be found in JCuda
    val sizeOfCurandStatePhilox4_32_10_t = 64L
    val sizeOfCurandStateMRG32k3a = 72L
  }
}