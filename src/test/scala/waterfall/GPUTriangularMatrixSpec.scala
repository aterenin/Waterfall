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

class GPUTriangularMatrixSpec extends FlatSpec with Assertions with Matchers {
  import GPUTestUtils._

  override def withFixture(test: NoArgTest) = { assume(initialized); test() }


  "GPUTriangularMatrix" should "perform matrix-matrix multiplication" in {
    val X = GPUMatrix.createFromColumnMajorArray(hostX)
    val XT = GPUMatrix.createFromColumnMajorArray(hostX.transpose)
    val R = GPUMatrix.createFromColumnMajorArray(hostR).declareTriangular
    val XR = GPUMatrix.create(hostXnumRows, hostXnumCols)
    val XR2 = GPUMatrix.create(hostXnumRows, hostXnumCols)
    val XRt = GPUMatrix.create(hostXnumRows, hostXnumCols)
    val XRt2 = GPUMatrix.create(hostXnumRows, hostXnumCols)
    val RXt = GPUMatrix.create(hostXnumCols, hostXnumRows)
    val RXt2 = GPUMatrix.create(hostXnumCols, hostXnumRows)
    val RtXt = GPUMatrix.create(hostXnumCols, hostXnumRows)
    val RtXt2 = GPUMatrix.create(hostXnumCols, hostXnumRows)

    XR =: X * R
    XR2 =: XT.T * R
    XRt =: X * R.T
    XRt2 =: XT.T * R.T
    RXt =: R * X.T
    RXt2 =: R * XT
    RtXt =: R.T * X.T
    RtXt2 =: R.T * XT

    testGPUEquality(XR, hostXR)
    testGPUEquality(XR2, hostXR)
    testGPUEquality(XRt, hostXRt)
    testGPUEquality(XRt2, hostXRt)
    testGPUEquality(RXt, hostRXt)
    testGPUEquality(RXt2, hostRXt)
    testGPUEquality(RtXt, hostRtXt)
    testGPUEquality(RtXt2, hostRtXt)
    testGPUEquality(R, hostR)
    testGPUEquality(X, hostX)
  }

  it should "perform matrix-vector multiplication" in {
    val R = GPUMatrix.createFromColumnMajorArray(hostR).declareTriangular
    val v = GPUVector.createFromArray(hostV)
    val Rv = GPUVector.create(v.length)
    val vtRt = GPUVector.create(v.length)

    Rv =: R * v
    vtRt =: v.T * R.T

    testGPUEquality(Rv, hostRv)
    testGPUEquality(vtRt, hostvtRt, transpose = true)
    testGPUEquality(v, hostV)
    testGPUEquality(R, hostR)
  }

  "GPUInverseTriangularMatrix" should "solve a triangular system" in {
    val R = GPUMatrix.createFromColumnMajorArray(hostR).declareTriangular
    val v = GPUVector.createFromArray(hostV)
    val RinvV = GPUVector.create(v.length)
    val vtRinvt = GPUVector.create(v.length)
    val vtRinvt2 = GPUVector.create(v.length)

    RinvV =: R.inv * v
    vtRinvt =: v.T * R.inv.T
    vtRinvt2 =: v.T * R.T.inv

    testGPUEquality(RinvV, hostRinvV)
    testGPUEquality(vtRinvt, hostVtRinvt, transpose = true)
    testGPUEquality(vtRinvt2, hostVtRinvt, transpose = true)
    testGPUEquality(R, hostR)
    testGPUEquality(v, hostV)
  }

  it should "solve a triangular matrix equation" in {
    val X = GPUMatrix.createFromColumnMajorArray(hostX)
    val XT = GPUMatrix.createFromColumnMajorArray(hostX.transpose)
    val R = GPUMatrix.createFromColumnMajorArray(hostR).declareTriangular
    val XRinv = GPUMatrix.create(hostXnumRows, hostXnumCols)
    val XRinv2 = GPUMatrix.create(hostXnumRows, hostXnumCols)
    val XRinvt = GPUMatrix.create(hostXnumRows, hostXnumCols)
    val XRinvt2 = GPUMatrix.create(hostXnumRows, hostXnumCols)
    val XRinvt3 = GPUMatrix.create(hostXnumRows, hostXnumCols)
    val XRinvt4 = GPUMatrix.create(hostXnumRows, hostXnumCols)
    val RinvXt = GPUMatrix.create(hostXnumCols, hostXnumRows)
    val RinvXt2 = GPUMatrix.create(hostXnumCols, hostXnumRows)
    val RinvtXt = GPUMatrix.create(hostXnumCols, hostXnumRows)
    val RinvtXt2 = GPUMatrix.create(hostXnumCols, hostXnumRows)
    val RinvtXt3 = GPUMatrix.create(hostXnumCols, hostXnumRows)
    val RinvtXt4 = GPUMatrix.create(hostXnumCols, hostXnumRows)

    XRinv =: X * R.inv
    XRinv2 =: XT.T * R.inv
    XRinvt =: X * R.inv.T
    XRinvt2 =: X * R.T.inv
    XRinvt3 =: XT.T * R.inv.T
    XRinvt4 =: XT.T * R.T.inv
    RinvXt =: R.inv * X.T
    RinvXt2 =: R.inv * XT
    RinvtXt =: R.inv.T * X.T
    RinvtXt2 =: R.T.inv * X.T
    RinvtXt3 =: R.inv.T * XT
    RinvtXt4 =: R.T.inv * XT

    testGPUEquality(XRinv, hostXRinv)
    testGPUEquality(XRinv2, hostXRinv)
    testGPUEquality(XRinvt, hostXRinvt)
    testGPUEquality(XRinvt2, hostXRinvt)
    testGPUEquality(XRinvt3, hostXRinvt)
    testGPUEquality(XRinvt4, hostXRinvt)
    testGPUEquality(RinvXt, hostRinvXt)
    testGPUEquality(RinvXt2, hostRinvXt)
    testGPUEquality(RinvtXt, hostRinvtXt)
    testGPUEquality(RinvtXt2, hostRinvtXt)
    testGPUEquality(RinvtXt3, hostRinvtXt)
    testGPUEquality(RinvtXt4, hostRinvtXt)
    testGPUEquality(X, hostX)
    testGPUEquality(R, hostR)
  }
}
