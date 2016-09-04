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
import waterfall.{GPUArray, GPUConstant, GPUMatrix, GPUVector}

class GPUVectorSpec extends FlatSpec with Assertions with Matchers {
  import GPUTestInit._

  override def withFixture(test: NoArgTest) = {
    assume(initialized)
    test()
  }


  "GPUVector" should "perform vector-matrix multiplication" in {
    val X = GPUMatrix.createFromColumnMajorArray(hostX)
    val v = GPUVector.createFromArray(hostV)
    val vtXt = GPUVector.create(v.length)

    vtXt =: v.T * X.T

    testGPUEquality(vtXt, hostvtXt)
    testGPUEquality(X, hostX.flatten)
    testGPUEquality(v, hostV)
  }

  it should "perform in-place vector addition" in {
    val v = GPUVector.createFromArray(hostV)
    val twoV = GPUVector.create(v.length)

    v.copyTo(twoV)

    twoV =: v + twoV

    testGPUEquality(twoV, hostV.map(_ * 2.0f))
    testGPUEquality(v, hostV)
  }

  it should "perform a dot product computations" in {
    val v = GPUVector.createFromArray(hostV)
    val c = GPUConstant.create(0.0f)

    c =: (v dot v)

    testGPUEquality(c, Array(hostV.zip(hostV).map(p => p._1 * p._2).sum))
  }
}
