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
case class GPUaxpy(x: GPUVector) extends GPUComputation
case class GPUmaxpy(A: GPUMatrix) extends GPUComputation
case object GPUdot extends GPUComputation
case class GPUgeam(A: GPUMatrix, B: GPUMatrix) extends GPUComputation
case class GPUgemm(A: GPUMatrix, B: GPUMatrix) extends GPUComputation
case class GPUlgemv(x: GPUVector, A: GPUMatrix) extends GPUComputation
case class GPUgemv(A: GPUMatrix, x: GPUVector) extends GPUComputation
case object GPUspotrf extends GPUComputation
case object GPUspotrs extends GPUComputation
case object GPUtrsv extends GPUComputation