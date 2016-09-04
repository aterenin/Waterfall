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

class GPUSymmetricMatrixSpec extends FlatSpec with Assertions with Matchers {
  val hostX =  Array(
    Array(-1.3005037f, 0.2967214f, -2.2460886f, 0.5507635f, -0.2778561f),
    Array( 0.7178072f, 0.3204358f, -0.4091498f, 0.1456138f, -0.1742799f),
    Array(-1.3635131f, 0.7393017f, -0.2307020f, 0.5801271f, -0.2263644f)
  ).transpose // transpose to change to column major format
  val hostXnumRows = hostX.head.length
  val hostXnumCols = hostX.length

  val hostXXtX = Array(
    Array(-13.164660f,  4.34380343f, -16.9357003f,  5.6770347f, -2.69327581f),
    Array(  1.042531f, -0.05679918f,  -0.6914778f, -0.0666368f, -0.08547064f),
    Array( -8.019992f,  2.79532536f,  -6.9409501f,  3.1687155f, -1.35705768f)
  ).transpose // transpose to change to column major format

  val hostXtX = Array(
    Array(4.0657250f, -1.1639237f, 2.9419211f, -1.4027582f, 0.5449043f),
    Array(-1.1639237f, 0.7372897f, -0.9681272f, 0.6389721f, -0.3056430f),
    Array(2.9419211f, -0.9681272f, 5.2655410f, -1.4304780f, 0.7476187f),
    Array(-1.4027582f, 0.6389721f, -1.4304780f, 0.6610913f, -0.3097307f),
    Array(0.5449043f, -0.3056430f, 0.7476187f, -0.3097307f, 0.1588183f)
  ) // no need to transpose because symmetric

  val hostV = Array(0.6998572f, -1.0195756f, 1.0799649f, -0.6968716f, 0.4279191f)
  val hostXtXv = hostXtX.map(x => x.zip(hostV).map(p => p._1*p._2).sum)

  def testGPUEquality(A: GPUArray, B: Array[Float]) = {
    A.copyToHost.zip(B).foreach{
      case (l, c) => l shouldEqual (c +- 0.0001f)
    }
  }

  override def withFixture(test: NoArgTest) = {
    assume(GPUInitCheck.initialized)
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
    cancel()
  }

  it should "solve a linear system using provided Cholesky decomposition" in {
    cancel()
  }
}
