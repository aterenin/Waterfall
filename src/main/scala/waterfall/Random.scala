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
import jcuda.runtime.JCuda.cudaMalloc
import jcuda.jcurand.JCurand.curandSetPseudoRandomGeneratorSeed

/**
  * An object that contains methods for generating random numbers
  *
  * @author Alexander Terenin
  */
object Random {
  def normal = new GPUArrayResult(GPUGenerateNormal(0.0f,1.0f))
  def normal(mu: Float, sigmasq: Float) = new GPUArrayResult(GPUGenerateNormal(mu, math.sqrt(sigmasq.toDouble).toFloat))
  def normal(mu: Double, sigmasq: Double) = new GPUArrayResult(GPUGenerateNormal(mu.toFloat, math.sqrt(sigmasq).toFloat))
  def normalByStddev(mu: Float, sigma: Float) = new GPUArrayResult(GPUGenerateNormal(mu, sigma))
  def uniform = new GPUArrayResult(GPUGenerateUniform)

  def setSeed(seed: Int) = curandSetPseudoRandomGeneratorSeed(Waterfall.curandGenerator, seed)

  def allocateDeviceRNGState(state: RNGState) = {
    val p = new Pointer
    cudaMalloc(p, state.size)
    GPURNGState(p, state)
  }

  sealed trait RNGState { val size: Int }
  case object PhiloxState extends RNGState { val size = 64 } //somehow, sizes are nowhere to be found in JCuda
  case object MRG32k3aState extends RNGState {val size = 72 }
  case object XORWOWState extends RNGState { val size = 48 }
  case class GPURNGState(ptr: Pointer, state: RNGState) { val numBytes = state.size }
}