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
import waterfall.{GPUArray, GPUMatrix, GPUVector}
import waterfall.MatrixProperties.createCholeskyWorkspace

class GPUSymmetricMatrixSpec extends FlatSpec with Assertions with Matchers {
  import GPUTestInit._

  override def withFixture(test: NoArgTest) = {
    assume(initialized)
    test()
  }


  "GPUSymmetricMatrix" should "perform matrix-matrix multiplication" in {
    val X = GPUMatrix.createFromColumnMajorArray(hostX)
    val XtX = GPUMatrix.createFromColumnMajorArray(hostXtX).declareSymmetric
    val XXtX = GPUMatrix.create(hostXnumRows, hostXnumCols)
    val XtXXt = GPUMatrix.create(hostXnumCols, hostXnumRows)

    XXtX =: X * XtX
    XtXXt =: XtX * X.T

    testGPUEquality(XXtX, hostXXtX.flatten)
    testGPUEquality(XtXXt, hostXXtX.transpose.flatten)
    testGPUEquality(XtX, hostXtX.flatten)
    testGPUEquality(X, hostX.flatten)
  }

  it should "perform matrix-vector multiplication" in {
    val XtX = GPUMatrix.createFromColumnMajorArray(hostXtX).declareSymmetric
    val v = GPUVector.createFromArray(hostV)
    val XtXv = GPUVector.create(v.length)

    XtXv =: XtX * v

    testGPUEquality(XtXv, hostXtXv)
    testGPUEquality(XtX, hostXtX.flatten)
    testGPUEquality(v, hostV)
  }

  it should "perform a Cholesky decomposition" in {
    val XtX = GPUMatrix.createFromColumnMajorArray(hostXtX).declareSymmetric
    val R = GPUMatrix.create(XtX.size, XtX.size).declareTriangular
    val ws = createCholeskyWorkspace(XtX)

    R =: XtX.computeCholesky(ws)

    R shouldEqual XtX.chol

    testGPUEquality(R, hostR.flatten)
  }

  it should "solve a linear system using provided Cholesky decomposition" in {
    cancel()
  }
}
