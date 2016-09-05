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

object GPUTestInit {
  lazy val initialized = {
    try {
      Waterfall.init()
      true
    } catch {
      case e: UnsatisfiedLinkError => false
    }
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

  val hostXXtX = Array(
    Array(-13.164660f,  1.04253063f, -8.019992f),
    Array(  4.343803f, -0.05679918f,  2.795325f),
    Array(-16.935700f, -0.69147777f, -6.940950f),
    Array(  5.677035f, -0.06663680f,  3.168716f),
    Array( -2.693276f, -0.08547064f, -1.357058f)
  ).transpose // transpose to change to column major format

  val hostXtX = Array(
    Array(7.2048119f,  0.2091794f,  2.8932072f),
    Array(0.2091794f,  0.8369067f, -0.5235238f),
    Array(2.8932072f, -0.5235238f,  2.8467467f)
  ) // no need to transpose becat(use symmetric

  val hostR = Array(
    Array(2.684178f,  0.07793054f,  1.0778745f),
    Array(0.000000f, -0.91150070f,  0.6665087f),
    Array(0.000000f,  0.00000000f, -1.1138668f)
  ).transpose // transpose to change to column major format

  val hostV = Array(0.6998572f, -1.0195756f, 1.0799649f)
  val hostXtXv = hostXtX.map(x => x.zip(hostV).map(p => p._1*p._2).sum)

  val hostXv = hostX.transpose.map(x => x.zip(hostV).map(p => p._1*p._2).sum)
  val hostvtXt = hostXv

  def testGPUEquality(A: GPUArray, B: Array[Float]) = {
    A.copyToHost.zip(B).foreach{
      case (l, c) => l shouldEqual (c +- 0.0001f)
    }
  }

  def testGPUTriangularEquality(A: GPUTriangularMatrix, B: Array[Float]) = {
    A.copyToHost.zip(B)
      .sliding(A.size)
      .toArray.zipWithIndex
      .flatMap{
        case (column, colIdx) =>
          column.zipWithIndex.map{
            case ((a, b), rowIdx) =>
              if(rowIdx >= colIdx) (a,b) else (0.0f, b)
          }
      }
      .foreach{
        case (l, c) => l shouldEqual (c +- 0.0001f)
      }
  }
}
