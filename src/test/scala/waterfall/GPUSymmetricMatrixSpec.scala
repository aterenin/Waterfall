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
import waterfall.MatrixProperties.{createCholeskyWorkspace, checkDevInfo}

class GPUSymmetricMatrixSpec extends FlatSpec with Assertions with Matchers {
  import GPUTestUtils._

  override def withFixture(test: NoArgTest) = { assume(initialized); test() }


  "GPUSymmetricMatrix" should "perform matrix-matrix multiplication" in {
    val X = GPUMatrix.createFromColumnMajorArray(hostX)
    val XT = GPUMatrix.createFromColumnMajorArray(hostX.transpose)
    val XtX = GPUMatrix.createFromColumnMajorArray(hostXtX).declareSymmetric
    val XXtX = GPUMatrix.create(hostXnumRows, hostXnumCols)
    val XtXXt = GPUMatrix.create(hostXnumCols, hostXnumRows)
    val XXtX2 = GPUMatrix.create(hostXnumRows, hostXnumCols)
    val XtXXt2 = GPUMatrix.create(hostXnumCols, hostXnumRows)

    XXtX =: X * XtX
    XtXXt =: XtX * X.T
    XXtX2 =: XT.T * XtX
    XtXXt2 =: XtX * XT

    testGPUEquality(XXtX, hostXXtX)
    testGPUEquality(XtXXt, hostXtXXt)
    testGPUEquality(XXtX2, hostXXtX)
    testGPUEquality(XtXXt2, hostXtXXt)
    testGPUEquality(XtX, hostXtX)
    testGPUEquality(X, hostX)
  }

  it should "perform matrix-vector multiplication" in {
    val XtX = GPUMatrix.createFromColumnMajorArray(hostXtX).declareSymmetric
    val v = GPUVector.createFromArray(hostV)
    val XtXv = GPUVector.create(v.length)
    val vtXtX = GPUVector.create(v.length)

    XtXv =: XtX * v
    vtXtX =: v.T * XtX

    testGPUEquality(XtXv, hostXtXv)
    testGPUEquality(vtXtX, hostvtXtX, transpose = true)
    testGPUEquality(XtX, hostXtX)
    testGPUEquality(v, hostV)
  }

  it should "perform a Cholesky decomposition" in {
    val XtX = GPUMatrix.createFromColumnMajorArray(hostXtX).declareSymmetric
    val R = GPUMatrix.create(XtX.size, XtX.size).declareTriangular
    val ws = createCholeskyWorkspace(XtX)

    R =: XtX.computeCholesky(ws)

    checkDevInfo(ws.devInfo)

    R shouldEqual XtX.chol
    testGPUEquality(R, hostR.map(_.map(f => if(f == 0.0f) Float.NaN else f)))
  }

  "GPUInverseSymmetricMatrix" should "solve a linear system using provided Cholesky decomposition" in {
    cancel()
    val XtX = GPUMatrix.createFromColumnMajorArray(hostXtX).declareSymmetric
    val v = GPUVector.createFromArray(hostV)
    val R = GPUMatrix.createFromColumnMajorArray(hostR).declareTriangular
    val ws = createCholeskyWorkspace(XtX)
    val XtXinvV = GPUVector.create(v.length)
    val vtXtXinv = GPUVector.create(v.length)

    XtX.attachCholesky(R, ws)

    XtXinvV =: XtX.inv * v
    vtXtXinv =: v.T * XtX.inv

    testGPUEquality(XtXinvV, hostXtXinvV)
    testGPUEquality(vtXtXinv, hostvtXtXinv, transpose = true)
    testGPUEquality(XtX, hostXtX)
    testGPUEquality(R, hostR)
    testGPUEquality(v, hostV)
  }

  it should "solve a matrix equation using provided Cholesky decomposition" in {
    cancel()
    val X = GPUMatrix.createFromColumnMajorArray(hostX)
    val XtX = GPUMatrix.createFromColumnMajorArray(hostXtX).declareSymmetric
    val R = GPUMatrix.createFromColumnMajorArray(hostR).declareTriangular
    val ws = createCholeskyWorkspace(XtX)
    val XXtXinv = GPUMatrix.create(hostXnumRows, hostXnumCols)
    val XtXinvXt = GPUMatrix.create(hostXnumCols, hostXnumRows)

    XtX.attachCholesky(R, ws)

    XXtXinv =: X * XtX.inv
    XtXinvXt =: XtX.inv * X.T

    testGPUEquality(XXtXinv, hostXXtXinv)
    testGPUEquality(XtXinvXt, hostXtXinvXt)
    testGPUEquality(X, hostX)
    testGPUEquality(XtX, hostXtX)
    testGPUEquality(R, hostR)
  }
}
