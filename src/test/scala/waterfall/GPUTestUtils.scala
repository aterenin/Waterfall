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

import org.scalatest.Matchers._

object GPUTestUtils {
  lazy val initialized = { // HACK: used to cancel all tests if no GPU on host system so that sbt assembly works
    try {
      Waterfall.init(); true
    } catch {
      case e: UnsatisfiedLinkError => false
    }
  }

  implicit class MatrixOperations(A: Array[Array[Float]]) {
    def multiplyBy(B: Array[Array[Float]]) = {
      A.transpose
        .map{
          aRow =>
            B.map{
              bCol =>
                aRow.zip(bCol).map{case (x,y) => x*y}.sum
            }
        }
        .transpose
    }

    def multiplyBy(v: Array[Float]) = A.transpose.map(aRow => aRow.zip(v).map{case (x,y) => x*y}.sum)
  }

  val hostX =  Array(
    Array(-1.3005037f,  0.7178072f, -1.3635131f),
    Array( 0.2967214f,  0.3204358f,  0.7393017f),
    Array(-2.2460886f, -0.4091498f, -0.2307020f),
    Array( 0.5507635f,  0.1456138f,  0.5801271f),
    Array(-0.2778561f, -0.1742799f, -0.2263644f)
  ).transpose // transpose to change to column major format
  val hostXnumRows = hostX.head.length
  val hostXnumCols = hostX.length

  val hostR = Array(
    Array(2.684178f,  0.07793054f,  1.0778745f),
    Array(0.000000f,  0.91150070f, -0.6665087f),
    Array(0.000000f,  0.00000000f,  1.1138668f)
  ).transpose // transpose to change to column major format

  val hostRinv = Array(
    Array(0.3725535f, -0.03185219f, -0.3795747f),
    Array(0.0000000f,  1.09709186f,  0.6564710f),
    Array(0.0000000f,  0.00000000f,  0.8977734f)
  ).transpose // transpose to change to column major format

  val hostV = Array(0.6998572f, -1.0195756f, 1.0799649f)

  val hostXtX = hostX.transpose.multiplyBy(hostX)

  val hostXtXinv = hostRinv.multiplyBy(hostRinv.transpose)

  val hostXXtX = hostX.multiplyBy(hostXtX)
  val hostXtXXt = hostXXtX.transpose

  val hostXtXv = hostXtX.multiplyBy(hostV)
  val hostvtXtX = hostXtXv

  val hostXv = hostX.multiplyBy(hostV)
  val hostvtXt = hostXv

  val hostXR = hostX.multiplyBy(hostR)
  val hostXRt = hostX.multiplyBy(hostR.transpose)
  val hostRXt = hostXRt.transpose
  val hostRtXt = hostXR.transpose

  val hostRv = hostR.multiplyBy(hostV)
  val hostvtRt = hostRv

  val hostXRinv = hostX.multiplyBy(hostRinv)
  val hostXRinvt = hostX.multiplyBy(hostRinv.transpose)
  val hostRinvXt = hostXRinvt.transpose
  val hostRinvtXt = hostXRinv.transpose

  val hostRinvV = hostRinv.multiplyBy(hostV)
  val hostVtRinvt = hostRinvV

  val hostXtXinvV = hostXtXinv.multiplyBy(hostV)
  val hostvtXtXinv = hostXtXinvV

  val hostXXtXinv = hostX.multiplyBy(hostXtXinv)
  val hostXtXinvXt = hostXXtXinv.transpose


  def testGPUEquality(A: GPUArray, B: Array[Float]) = {
    A.copyToHost.zip(B).foreach{
      case (test, correct) => if(correct==correct) test shouldEqual (correct +- 0.0001f) // correct==correct is false for NaN
    }
  }
  def testGPUEquality(A: GPUMatrix, B: Array[Array[Float]]): Unit = {
    A.numRows shouldEqual B.head.length
    A.numCols shouldEqual B.length
    val transposeMatchedB = if(A.isTranspose) B.transpose else B
    testGPUEquality(A: GPUArray, transposeMatchedB.flatten) // annotation needed to prevent infinite tail recursive loop
  }
  def testGPUEquality(A: GPUVector, B: Array[Float], transpose: Boolean = false): Unit = {
    A.length shouldEqual B.length
    A.isTranspose shouldEqual transpose
    testGPUEquality(A: GPUArray, B) // annotation needed to prevent infinite tail recursive loop
  }
}
