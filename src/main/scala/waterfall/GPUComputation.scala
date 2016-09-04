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


/**
  * A container value that holds a computation together with its input, which is never mutated
  *
  * @author Alexander Terenin
  */
sealed trait GPUComputation

case class GPUAlphaXPlusY(v1: GPUVector, v2: GPUVector) extends GPUComputation

case class GPUDot(x: GPUVector, y: GPUVector) extends GPUComputation

case class GPUGeneralAddMatrix(A: GPUMatrix, B: GPUMatrix) extends GPUComputation

case class GPUGeneralMatrixMatrix(A: GPUMatrix, B: GPUMatrix) extends GPUComputation

case class GPULeftGeneralMatrixVector(x: GPUVector, A: GPUMatrix) extends GPUComputation
case class GPUGeneralMatrixVector(A: GPUMatrix, x: GPUVector) extends GPUComputation

case object GPUPositiveDefiniteTriangularFactorize extends GPUComputation
case object GPUPositiveDefiniteTriangularSolve extends GPUComputation

case class GPUSymmetricMatrixMatrix(A: GPUSymmetricMatrix, B: GPUMatrix) extends GPUComputation
case class GPULeftSymmetricMatrixMatrix(B: GPUMatrix, A: GPUSymmetricMatrix) extends GPUComputation

case class GPUSymmetricMatrixVector(A: GPUSymmetricMatrix, x: GPUVector) extends GPUComputation
case class GPULeftSymmetricMatrixVector(x: GPUVector, A: GPUSymmetricMatrix) extends GPUComputation

case class GPUTriangularMatrixMatrix(A: GPUTriangularMatrix, B: GPUMatrix) extends GPUComputation
case class GPULeftTriangularMatrixMatrix(B: GPUMatrix, A: GPUTriangularMatrix) extends GPUComputation

case class GPUTriangularMatrixVector(A: GPUTriangularMatrix, x: GPUVector) extends GPUComputation
case class GPULeftTriangularMatrixVector(x: GPUVector, A: GPUTriangularMatrix) extends GPUComputation

case object GPUTriangularSolve extends GPUComputation