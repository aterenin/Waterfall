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

import Implicits.DebugImplicits
import jcuda.jcurand.JCurand.{curandGenerateNormal, curandGenerateUniform}

/**
  * A not-yet-evaluated result of a computation that yields an array
  *
  * @author Alexander Terenin
  *
  * @param computation the computation containing the needed input that will yield selected result
  */
class GPUArrayResult(computation: GPUComputation) {
  def =:(a: GPUArray): GPUArray = computation match {
    case GPUGenerateNormal(mu: Float, sigma: Float) => executeGenerateNormal(a, mu, sigma)
    case GPUGenerateUniform => executeGenerateUniform(a)
    case _ => throw new Exception("wrong array operation in =:")
  }

  private def executeGenerateNormal(a: GPUArray, mu: Float, sigma: Float) = {
    assert(a.numElements % 2 == 0, s"unsupported: for some reason, cuRAND can only generate normals for arrays with an even number of elements - try padding your array by 1 and ignoring the last element")
    curandGenerateNormal(Waterfall.curandGenerator,
      a.ptr, a.numElements,
      mu, sigma
    ).checkJCurandStatus()
    a
  }

  private def executeGenerateUniform(a: GPUArray) = {
    curandGenerateUniform(Waterfall.curandGenerator,
      a.ptr, a.numElements
    ).checkJCurandStatus()
    a
  }
}