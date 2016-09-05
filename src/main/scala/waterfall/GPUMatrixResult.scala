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

import jcuda.jcublas.JCublas2._
import jcuda.jcusolver.JCusolverDn.{cusolverDnSpotrf}
import Implicits.{DebugImplicits, FillModeImplicits, SideImplicits, TransposeImplicits}
import MatrixProperties.{CholeskyWorkspace, Left, Right}

/**
  * A not-yet-evaluated result of a computation that yields a matrix
  *
  * @author Alexander Terenin
  *
  * @param computation the computation containing the needed input that will yield selected result
  */
class GPUMatrixResult(computation: GPUComputation) {
  def =:(C: GPUMatrix): GPUMatrix = execute(C.mutateConstant(None).mutateTranspose(false))
  def +=:(C: GPUMatrix): GPUMatrix  = execute(C.setInPlaceAdditionConstant().mutateTranspose(false))

  private implicit class ResultImplicits(M: GPUMatrix) {
    def setInPlaceAdditionConstant() = if(M.constant.nonEmpty) M else M.mutateConstant(Some(Waterfall.Constants.one))
  }

  private def execute(C: GPUMatrix) = computation match {
    case GPUGeneralAddMatrix(a,b) => executeSgeam(a,b,C)
    case GPUGeneralMatrixMatrix(a,b) => executeSgemm(a,b,C)
    case GPUSymmetricMatrixMatrix(a,b) if !b.isTranspose => executeSsymm(a,b,C)
    case GPUSymmetricMatrixMatrix(a,b) if b.isTranspose => executeSsymm(b.T,a,C, transeposeC = true)
    case GPULeftSymmetricMatrixMatrix(b,a) if !b.isTranspose => executeSsymm(b,a,C)
    case GPULeftSymmetricMatrixMatrix(b,a) if b.isTranspose => executeSsymm(a,b.T,C, transeposeC = true)
    case GPUPositiveDefiniteTriangularFactorize(a,ws) => executeSpotrf(a,C,ws)
    case GPUTriangularMatrixMatrix(a: GPUTriangularMatrix, b: GPUMatrix) => ???
    case GPULeftTriangularMatrixMatrix(b: GPUMatrix, a: GPUTriangularMatrix) => ???
    case GPUTriangularSolveMatrix(ainv: GPUInverseTriangularMatrix, b: GPUMatrix) => ???
    case GPULeftTriangularSolveMatrix(b: GPUMatrix, ainv: GPUInverseTriangularMatrix) => ???
    case GPUPositiveDefiniteTriangularSolve(ainv: GPUInverseSymmetricMatrix, b: GPUMatrix) => ???
    case GPULeftPositiveDefiniteTriangularSolve(b: GPUMatrix, ainv: GPUInverseSymmetricMatrix) => ???
    case _ => throw new Exception("wrong matrix operation in execute(C)")
  }

  private def executeSgeam(A: GPUMatrix, B: GPUMatrix, C: GPUMatrix) = {
    // check for compatibility
    assert(A.numRows == B.numRows, s"mismatched matrix dimensions: got ${A.numRows} != ${B.numRows}")
    assert(B.numRows == C.numRows, s"mismatched matrix dimensions: got ${B.numRows} != ${C.numRows}")
    assert(A.numCols == B.numCols, s"mismatched matrix dimensions: got ${A.numCols} != ${B.numCols}")
    assert(B.numCols == C.numCols, s"mismatched matrix dimensions: got ${B.numCols} != ${C.numCols}")
    assert(!((A.ptr eq C.ptr) && (B.ptr eq C.ptr)), s"unsupported: cannot in-place add matrix to itself")
    assert(if(A.ptr eq C.ptr) !A.isTranspose else true, s"unsupported: cannot perform in-place transpose")
    assert(if(B.ptr eq C.ptr) !B.isTranspose else true, s"unsupported: cannot perform in-place transpose")
    assert(C.constant.isEmpty, s"unsupported: in-place addition supported only through =: for matrices")

    // determine constants
    val alpha = A.constant.getOrElse(Waterfall.Constants.one)
    val beta = B.constant.getOrElse(Waterfall.Constants.one)

    // perform single-precision general matrix-matrix addition
    cublasSgeam(Waterfall.cublasHandle,
      A.isTranspose.toTransposeOpId, B.isTranspose.toTransposeOpId,
      C.numRows, C.numCols,
      alpha.ptr,
      A.ptr, A.leadingDimension,
      beta.ptr,
      B.ptr, B.leadingDimension,
      C.ptr, C.leadingDimension
    ).checkJCublasStatus()

    // return result
    C.mutateConstant(None)
  }

  private def executeSgemm(A: GPUMatrix, B: GPUMatrix, C: GPUMatrix) = {
    // check for compatibility
    assert(A.numRows == C.numRows, s"mismatched matrix dimensions: got ${A.numRows} != ${C.numRows}")
    assert(B.numCols == C.numCols, s"mismatched matrix dimensions: got ${B.numCols} != ${C.numCols}")
    assert(A.numCols == B.numRows, s"mismatched matrix dimensions: got ${A.numCols} != ${B.numRows}")
    assert(A.constant.isEmpty || B.constant.isEmpty, s"unsupported: only one input constant can be defined")

    // determine constants
    val alpha = A.constant.getOrElse(B.constant.getOrElse(Waterfall.Constants.one))
    val beta = C.constant.getOrElse(Waterfall.Constants.zero)

    // perform single-precision general matrix-matrix multiplication
    cublasSgemm(Waterfall.cublasHandle,
      A.isTranspose.toTransposeOpId, B.isTranspose.toTransposeOpId,
      C.numRows, C.numCols, A.numCols,
      alpha.ptr,
      A.ptr, A.leadingDimension,
      B.ptr, B.leadingDimension,
      beta.ptr,
      C.ptr, C.leadingDimension
    ).checkJCublasStatus()

    // return result
    C.mutateConstant(None)
  }

  private def executeSsymm(M: GPUMatrix, N: GPUMatrix, C: GPUMatrix, transeposeC: Boolean = false) = {
    //determine parameters and check for compatibility
    val (a, b, side) = (M,N) match {
      case (b: GPUMatrix, a: GPUSymmetricMatrix) =>
        assert(a.size == b.numRows, s"mismatched matrix dimensions: got ${a.size} != ${b.numRows}")
        assert(a.size == C.numRows, s"mismatched matrix dimensions: got ${a.size} != ${C.numRows}")
        assert(b.numCols == C.numCols, s"mismatched matrix dimensions: got ${b.numCols} != ${C.numCols}")
        (a, b, Right)
      case (a: GPUSymmetricMatrix, b: GPUMatrix) =>
        assert(b.numCols == a.size, s"mismatched matrix dimensions: got ${b.numCols} != ${a.size}")
        assert(a.size == C.numCols, s"mismatched matrix dimensions: got ${a.size} != ${C.numCols}")
        assert(b.numRows == C.numRows, s"mismatched matrix dimensions: got ${b.numRows} != ${C.numRows}")
        (a, b, Left)
    }
    assert(a.constant.isEmpty || b.constant.isEmpty, s"unsupported: only one input constant can be defined")

    // determine constants
    val alpha = a.constant.getOrElse(b.constant.getOrElse(Waterfall.Constants.one))
    val beta = C.constant.getOrElse(Waterfall.Constants.zero)

    // perform single-precision general matrix-matrix multiplication
    cublasSsymm(Waterfall.cublasHandle,
      side.toSideId, a.fillMode.toFillModeId,
      C.numRows, C.numCols,
      alpha.ptr,
      a.ptr, a.leadingDimension,
      b.ptr, b.leadingDimension,
      beta.ptr,
      C.ptr, C.leadingDimension
    ).checkJCublasStatus()

    // return result
    C.mutateConstant(None).mutateTranspose(transeposeC)
  }

  private def executeSpotrf(A: GPUSymmetricMatrix, uncheckedC: GPUMatrix, ws: CholeskyWorkspace) = {
    // check if output has been declared
    assert(uncheckedC.isInstanceOf[GPUTriangularMatrix], s"unsupported: output of Cholesky factorization needs to be declared triangular")
    val C = uncheckedC.asInstanceOf[GPUTriangularMatrix]

    // check for compatibility
    assert(A.size == C.size, s"mismatched matrix dimensions: got ${A.size} != ${C.size}")
    assert(A.constant.isEmpty && C.constant.isEmpty, s"unsupported: cannot compute Cholesky with attached constants")
    assert(A.fillMode eq C.fillMode, s"unsupported: Cholesky input and output must have same fill mode")

    // if not in-place, copy to output, and attach Cholesky decomposition to original matrix
    if(!(A.ptr eq C.ptr)) {
      A.copyTo(C)
      A.attachCholesky(C, ws)
    }

    // perform positive definite triangular factorization
    cusolverDnSpotrf(Waterfall.cusolverDnHandle,
      C.fillMode.toFillModeId,
      C.size,
      C.ptr, C.leadingDimension,
      ws.workspace,
      ws.workspaceSize,
      ws.devInfo
    ).checkJCusolverStatus()

    // TODO: check devInfo to ensure non-singularity - need to be careful here when using CUDA streams


    // return result
    C
  }
}
