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

import jcuda.driver.CUresult.{stringFor => cudaStatusStringFor, CUDA_SUCCESS}
import jcuda.jcublas.cublasStatus.{stringFor => cublasStatusStringFor, CUBLAS_STATUS_SUCCESS}
import jcuda.jcurand.curandStatus.{stringFor => curandStatusStringFor, CURAND_STATUS_SUCCESS}
import jcuda.jcusolver.cusolverStatus.{stringFor => cusolverStatusStringFor, CUSOLVER_STATUS_SUCCESS}
import jcuda.jcublas.cublasOperation.{stringFor => _, CUBLAS_OP_N, CUBLAS_OP_T}
import jcuda.jcublas.cublasFillMode.{stringFor => _, CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER}
import jcuda.jcublas.cublasSideMode.{stringFor => _, CUBLAS_SIDE_LEFT, CUBLAS_SIDE_RIGHT}
import jcuda.jcublas.cublasDiagType.{stringFor => _, CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT}
import MatrixProperties.{FillMode, Lower, Upper, Side, Left, Right}

/**
  * Various types of implicits used in Waterfall
  *
  * @author Alexander Terenin
  */
object Implicits {


  implicit class HostMatrixImplicits(a: Array[Array[Float]]) {
    def toColumnMajorArray = a.transpose.flatten
  }

  implicit class HostArrayImplicits(a: Array[Float]) {

  }

  implicit class SideImplicits(s: Side) {
    def toSideId = s match {
      case Left => CUBLAS_SIDE_LEFT
      case Right => CUBLAS_SIDE_RIGHT
    }
  }

  implicit class FillModeImplicits(fm: FillMode) {
    def toFillModeId = fm match{
      case Lower => CUBLAS_FILL_MODE_LOWER
      case Upper => CUBLAS_FILL_MODE_UPPER
    }
  }

  implicit class DiagUnitImplicits(b: Boolean) {
    def toDiagUnitId = b match {
      case false => CUBLAS_DIAG_NON_UNIT
      case true => CUBLAS_DIAG_UNIT
    }
  }

  implicit class TransposeImplicits(b: Boolean) {
    def toTransposeOpId = b match {
      case false => CUBLAS_OP_N
      case true => CUBLAS_OP_T
    }
  }

  implicit class DebugImplicits(i: Int) {
    def checkJCublasStatus() = assert(i == CUBLAS_STATUS_SUCCESS, cublasStatusStringFor(i))
    def checkJCusolverStatus() = assert(i == CUSOLVER_STATUS_SUCCESS, cusolverStatusStringFor(i))
    def checkJCurandStatus() = assert(i == CURAND_STATUS_SUCCESS, curandStatusStringFor(i))
    def checkJCudaStatus() = assert(i == CUDA_SUCCESS, cudaStatusStringFor(i))
  }
}
