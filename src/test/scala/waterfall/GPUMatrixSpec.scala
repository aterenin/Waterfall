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

import org.scalatest.{Assertions, FlatSpec, Matchers}

class GPUMatrixSpec extends FlatSpec with Assertions with Matchers {
  import GPUTestUtils._

  override def withFixture(test: NoArgTest) = { assume(initialized); test() }


  it should "perform matrix-matrix multiplication" in {
    val X = GPUMatrix.createFromColumnMajorArray(hostX)
    val XtX = GPUMatrix.create(hostXtX.length, hostXtX.length)

    XtX =: X.T * X

    testGPUEquality(XtX, hostXtX)
    testGPUEquality(X, hostX)
  }

  it should "perform out-of-place matrix-matrix addition" in {
    val X = GPUMatrix.createFromColumnMajorArray(hostX)
    val twoX = GPUMatrix.create(hostXnumRows, hostXnumCols)

    twoX =: X + X

    testGPUEquality(twoX, hostX.map(_.map(_*2.0f)))
    testGPUEquality(X, hostX)
  }

  it should "perform in-place matrix-matrix addition" in {
    val X = GPUMatrix.createFromColumnMajorArray(hostX)
    val twoX = GPUMatrix.create(hostXnumRows, hostXnumCols)
    X.copyTo(twoX)

    twoX =: X + twoX

    testGPUEquality(twoX, hostX.map(_.map(_*2.0f)))
    testGPUEquality(X, hostX)
  }

  it should "perform matrix-vector multiplication" in {
    val X = GPUMatrix.createFromColumnMajorArray(hostX)
    val v = GPUVector.createFromArray(hostV)
    val Xv = GPUVector.create(v.length)
    val vtXt = GPUVector.create(v.length)

    Xv =: X * v
    vtXt =: v.T * X.T

    testGPUEquality(Xv, hostXv)
    testGPUEquality(vtXt, hostvtXt)
    testGPUEquality(X, hostX)
    testGPUEquality(v, hostV)
  }
}
