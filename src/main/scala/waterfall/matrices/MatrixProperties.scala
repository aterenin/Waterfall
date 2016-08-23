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

package waterfall.matrices

import waterfall.GPUMatrix

object MatrixProperties {
  sealed trait FillMode
  case object Upper extends FillMode
  case object Lower extends FillMode

  sealed trait Side
  case object Left extends Side
  case object Right extends Side

  sealed trait Decomposition
  case object NoDecomposition extends Decomposition
  case class CholeskyDecomposition(R: GPUMatrix with Triangular) extends Decomposition
}