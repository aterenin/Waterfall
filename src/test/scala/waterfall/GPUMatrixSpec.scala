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


  "GPUMatrix" should "perform matrix-matrix multiplication" in {
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
    val XT = GPUMatrix.createFromColumnMajorArray(hostX.transpose)
    val v = GPUVector.createFromArray(hostV)
    val Xv = GPUVector.create(X.numRows)
    val Xv2 = GPUVector.create(X.numRows)
    val vtXt = GPUVector.create(X.numRows)
    val vtXt2 = GPUVector.create(X.numRows)

    Xv =: X * v
    Xv2 =: XT.T * v
    vtXt =: v.T * X.T
    vtXt2 =: v.T * XT

    testGPUEquality(Xv, hostXv)
    testGPUEquality(Xv2, hostXv)
    testGPUEquality(vtXt, hostvtXt, transpose = true)
    testGPUEquality(vtXt2, hostvtXt, transpose = true)
    testGPUEquality(X, hostX)
    testGPUEquality(v, hostV)
  }
}
