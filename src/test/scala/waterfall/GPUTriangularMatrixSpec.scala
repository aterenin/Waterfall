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
import waterfall.MatrixProperties.createCholeskyWorkspace

class GPUTriangularMatrixSpec extends FlatSpec with Assertions with Matchers {
  import GPUTestUtils._

  override def withFixture(test: NoArgTest) = { assume(initialized); test() }


  "GPUTriangularMatrix" should "perform matrix-matrix multiplication" in {
    cancel()
    val X = GPUMatrix.createFromColumnMajorArray(hostX)
    val R = GPUMatrix.createFromColumnMajorArray(hostR).declareTriangular
    val XR = GPUMatrix.create(hostXnumRows, hostXnumCols)
    val RXt = GPUMatrix.create(hostXnumCols, hostXnumRows)

    XR =: X * R
    RXt =: R * X.T

    testGPUEquality(XR, hostXR)
    testGPUEquality(RXt, hostRXt)
    testGPUEquality(R, hostR)
    testGPUEquality(X, hostX)
  }

  it should "perform matrix-vector multiplication" in {
    cancel()
    val R = GPUMatrix.createFromColumnMajorArray(hostR).declareTriangular
    val v = GPUVector.createFromArray(hostV)
    val Rv = GPUVector.create(v.length)
    val vtRt = GPUVector.create(v.length)

    Rv =: R * v
    vtRt =: v * R.T

    testGPUEquality(Rv, hostRv)
    testGPUEquality(vtRt, hostvtRt)
    testGPUEquality(v, hostV)
    testGPUEquality(R, hostR)
  }

  "GPUInverseTriangularMatrix" should "solve a triangular system" in {
    cancel()
    val R = GPUMatrix.createFromColumnMajorArray(hostR).declareTriangular
    val v = GPUVector.createFromArray(hostV)
    val RinvV = GPUVector.create(v.length)
    val vtRinvt = GPUVector.create(v.length)
    val vtRinvt2 = GPUVector.create(v.length)

    RinvV =: R.inv * v
    vtRinvt =: v.T * R.inv.T
    vtRinvt2 =: v.T * R.T.inv

    testGPUEquality(RinvV, hostRinvV)
    testGPUEquality(vtRinvt, hostVtRinvt)
    testGPUEquality(vtRinvt2, hostVtRinvt)
    testGPUEquality(R, hostR)
    testGPUEquality(v, hostV)
  }

  it should "solve a triangular matrix equation" in {
    cancel()
    val X = GPUMatrix.createFromColumnMajorArray(hostX)
    val R = GPUMatrix.createFromColumnMajorArray(hostR).declareTriangular
    val XRinv = GPUMatrix.create(hostXnumRows, hostXnumCols)
    val RinvXt = GPUMatrix.create(hostXnumCols, hostXnumRows)

    XRinv =: X * R.inv
    RinvXt =: R.inv * X.T

    testGPUEquality(XRinv, hostXRinv)
    testGPUEquality(RinvXt, hostRinvXt)
    testGPUEquality(X, hostX)
    testGPUEquality(R, hostR)
  }
}
