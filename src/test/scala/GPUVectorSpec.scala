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

import org.scalatest.{Assertions, FlatSpec, Matchers}
import waterfall.{GPUMatrix}

class GPUVectorSpec extends FlatSpec with Assertions with Matchers {

  val hostX =  Array(
    Array(-1.3005037f, 0.2967214f, -2.2460886f, 0.5507635f, -0.2778561f),
    Array( 0.7178072f, 0.3204358f, -0.4091498f, 0.1456138f, -0.1742799f),
    Array(-1.3635131f, 0.7393017f, -0.2307020f, 0.5801271f, -0.2263644f)
  ).transpose // transpose to change to column major format
  val hostY = Array(0.6998572f, -1.0195756f, 1.0799649f, -0.6968716f, 0.4279191f)
  val hostZ = Array(0.83497682f, 0.85264958f, -1.77764342f, -0.09401397f, 0.15499778f)


  override def withFixture(test: NoArgTest) = {
    assume(GPUTestSetup.initialized)
    test()
  }


  "GPUVector" should "perform vector-matrix multiplication" in {
    cancel()
  }

  it should "perform in-place vector addition" in {
    cancel()
  }

  it should "perform a dot product computations" in {
    cancel()
  }
}
