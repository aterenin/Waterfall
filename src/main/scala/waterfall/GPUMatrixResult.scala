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
import jcuda.jcusolver.JCusolverDn.{cusolverDnSetStream, cusolverDnSpotrf}
import Implicits.{DebugImplicits, DiagUnitImplicits, FillModeImplicits, SideImplicits, TransposeImplicits}
import MatrixProperties.{CholeskyWorkspace, Left, Right}
import waterfall.Stream.GPUStream

/**
  * A not-yet-evaluated result of a computation that yields a matrix
  *
  * @author Alexander Terenin
  *
  * @param computation the computation containing the needed input that will yield selected result
  */
class GPUMatrixResult(computation: GPUComputation) {
  /**
    * The assignment operator: executes the GPUComputation in the GPUMatrixResult and stored it in the GPUMatrix passed in as argument
    *
    * @author Alexander Terenin
    * @param C the GPUMatrix in which output will be stored
    */
  def =:(C: GPUMatrix): GPUMatrix = computation match {
    case GPUGeneralAddMatrix(a,b) => executeSgeam(a,b,C)
    case GPUGeneralMatrixMatrix(a,b) => executeSgemm(a,b,C)
    case GPUSymmetricMatrixMatrix(a,b) if !b.isTranspose => executeSsymm(a,b,C)
    case GPUSymmetricMatrixMatrix(a,b) if b.isTranspose => executeSsymm(b.T,a,C, transposeC = true)
    case GPULeftSymmetricMatrixMatrix(b,a) if !b.isTranspose => executeSsymm(b,a,C)
    case GPULeftSymmetricMatrixMatrix(b,a) if b.isTranspose => executeSsymm(a,b.T,C, transposeC = true)
    case GPUPositiveDefiniteTriangularFactorize(a,ws) => executeSpotrf(a,C,ws)
    case GPUTriangularMatrixMatrix(a: GPUTriangularMatrix, b: GPUMatrix) if !b.isTranspose => executeStrmm(a,b,C)
    case GPUTriangularMatrixMatrix(a: GPUTriangularMatrix, b: GPUMatrix) if b.isTranspose => executeStrmm(b.T,a.T,C, transposeC = true)
    case GPULeftTriangularMatrixMatrix(b: GPUMatrix, a: GPUTriangularMatrix) if !b.isTranspose => executeStrmm(b,a,C)
    case GPULeftTriangularMatrixMatrix(b: GPUMatrix, a: GPUTriangularMatrix) if b.isTranspose => executeStrmm(a.T,b.T,C, transposeC = true)
    case GPUTriangularSolveMatrix(ainv: GPUInverseTriangularMatrix, b: GPUMatrix) if !b.isTranspose => executeStrsm(ainv,b,C)
    case GPUTriangularSolveMatrix(ainv: GPUInverseTriangularMatrix, b: GPUMatrix) if b.isTranspose => executeStrsm(b.T,ainv.T,C, transposeC = true)
    case GPULeftTriangularSolveMatrix(b: GPUMatrix, ainv: GPUInverseTriangularMatrix) if !b.isTranspose => executeStrsm(b,ainv,C)
    case GPULeftTriangularSolveMatrix(b: GPUMatrix, ainv: GPUInverseTriangularMatrix) if b.isTranspose => executeStrsm(ainv.T,b.T,C, transposeC = true)
    case GPUPositiveDefiniteTriangularSolve(ainv: GPUInverseSymmetricMatrix, b: GPUMatrix) if !b.isTranspose => executeSpotrs(ainv,b,C)
    case GPUPositiveDefiniteTriangularSolve(ainv: GPUInverseSymmetricMatrix, b: GPUMatrix) if b.isTranspose => executeSpotrs(b.T,ainv,C, transposeC = true)
    case GPULeftPositiveDefiniteTriangularSolve(b: GPUMatrix, ainv: GPUInverseSymmetricMatrix) if !b.isTranspose => executeSpotrs(b,ainv,C)
    case GPULeftPositiveDefiniteTriangularSolve(b: GPUMatrix, ainv: GPUInverseSymmetricMatrix) if b.isTranspose => executeSpotrs(ainv,b.T,C, transposeC = true)
    case _ => throw new Exception("wrong matrix operation in =:")
  }

  /**
    * The in-place addition operator: executes the GPUComputation in the GPUMatrixResult and adds it to the GPUMatrix passed in as argument
    *
    * @author Alexander Terenin
    * @param C the GPUMatrix to which output will be added
    * @return
    */
  def +=:(C: GPUMatrix): GPUMatrix  = computation match {
    case GPUGeneralMatrixMatrix(a,b) if !C.isTranspose => executeSgemm(a,b,C, inplace = true)
    case GPUSymmetricMatrixMatrix(a,b) if !b.isTranspose && !C.isTranspose => executeSsymm(a,b,C, inplace = true)
    case GPUSymmetricMatrixMatrix(a,b) if b.isTranspose && !C.isTranspose => executeSsymm(b.T,a,C, inplace = true, transposeC = true)
    case GPULeftSymmetricMatrixMatrix(b,a) if !b.isTranspose && !C.isTranspose => executeSsymm(b,a,C, inplace = true)
    case GPULeftSymmetricMatrixMatrix(b,a) if b.isTranspose && !C.isTranspose => executeSsymm(a,b.T,C, inplace = true, transposeC = true)
    case _ => throw new Exception("unsupported: cannot execute given matrix operation in-place in +=:")
  }

  private def executeSgeam(A: GPUMatrix, B: GPUMatrix, C: GPUMatrix)(implicit stream: GPUStream = Stream.default) = {
    // prepare output
    C.mutateConstant(None).mutateTranspose(newTranspose = false)

    // check for compatibility
    assert(A.numRows == B.numRows, s"mismatched matrix dimensions: got ${A.numRows} != ${B.numRows}")
    assert(B.numRows == C.numRows, s"mismatched matrix dimensions: got ${B.numRows} != ${C.numRows}")
    assert(A.numCols == B.numCols, s"mismatched matrix dimensions: got ${A.numCols} != ${B.numCols}")
    assert(B.numCols == C.numCols, s"mismatched matrix dimensions: got ${B.numCols} != ${C.numCols}")
    assert(!((A.ptr eq C.ptr) && (B.ptr eq C.ptr)), s"unsupported: cannot in-place add matrix to itself")
    assert(if(A.ptr eq C.ptr) !A.isTranspose else true, s"unsupported: cannot perform in-place transpose")
    assert(if(B.ptr eq C.ptr) !B.isTranspose else true, s"unsupported: cannot perform in-place transpose")

    // determine constants
    val alpha = A.constant.getOrElse(Waterfall.Constants.one)
    val beta = B.constant.getOrElse(Waterfall.Constants.one)

    // set stream
    cublasSetStream(Waterfall.cublasHandle, stream.cudaStream_t).checkJCublasStatus()

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

  private def executeSgemm(A: GPUMatrix, B: GPUMatrix, C: GPUMatrix, inplace: Boolean = false)(implicit stream: GPUStream = Stream.default) = {
    // prepare output
    if(!inplace) C.mutateConstant(None) else if(C.constant.isEmpty) C.mutateConstant(Some(Waterfall.Constants.one))
    C.mutateTranspose(newTranspose = false)

    // check for compatibility
    assert(A.numRows == C.numRows, s"mismatched matrix dimensions: got ${A.numRows} != ${C.numRows}")
    assert(B.numCols == C.numCols, s"mismatched matrix dimensions: got ${B.numCols} != ${C.numCols}")
    assert(A.numCols == B.numRows, s"mismatched matrix dimensions: got ${A.numCols} != ${B.numRows}")
    assert(A.constant.isEmpty || B.constant.isEmpty, s"unsupported: only one input constant can be defined")

    // determine constants
    val alpha = A.constant.getOrElse(B.constant.getOrElse(Waterfall.Constants.one))
    val beta = C.constant.getOrElse(Waterfall.Constants.zero)

    // set stream
    cublasSetStream(Waterfall.cublasHandle, stream.cudaStream_t).checkJCublasStatus()

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

  private def executeSsymm(M: GPUMatrix, N: GPUMatrix, C: GPUMatrix, inplace: Boolean = false, transposeC: Boolean = false)(implicit stream: GPUStream = Stream.default) = {
    // prepare output
    if(!inplace) C.mutateConstant(None) else if(C.constant.isEmpty) C.mutateConstant(Some(Waterfall.Constants.one))
    C.mutateTranspose(transposeC).mutateTranspose(newTranspose = false, flagOnly = true)

    // determine parameters and check for compatibility
    val (a, b, side) = (M,N) match {
      case (b: GPUMatrix, a: GPUSymmetricMatrix) =>
        assert(b.numCols == a.size, s"mismatched matrix dimensions: got ${b.numCols} != ${a.size}")
        assert(a.size == C.numCols, s"mismatched matrix dimensions: got ${a.size} != ${C.numCols}")
        assert(b.numRows == C.numRows, s"mismatched matrix dimensions: got ${b.numRows} != ${C.numRows}")
        (a, b, Right)
      case (a: GPUSymmetricMatrix, b: GPUMatrix) =>
        assert(a.size == b.numRows, s"mismatched matrix dimensions: got ${a.size} != ${b.numRows}")
        assert(a.size == C.numRows, s"mismatched matrix dimensions: got ${a.size} != ${C.numRows}")
        assert(b.numCols == C.numCols, s"mismatched matrix dimensions: got ${b.numCols} != ${C.numCols}")
        (a, b, Left)
    }
    assert(a.constant.isEmpty || b.constant.isEmpty, s"unsupported: only one input constant can be defined")

    // determine constants
    val alpha = a.constant.getOrElse(b.constant.getOrElse(Waterfall.Constants.one))
    val beta = C.constant.getOrElse(Waterfall.Constants.zero)

    // set stream
    cublasSetStream(Waterfall.cublasHandle, stream.cudaStream_t).checkJCublasStatus()

    // perform single-precision symmetric matrix-matrix multiplication
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
    C.mutateConstant(None).mutateTranspose(transposeC)
  }

  private def executeStrmm(M: GPUMatrix, N: GPUMatrix, C: GPUMatrix, transposeC: Boolean = false)(implicit stream: GPUStream = Stream.default) = {
    // prepare output
    C.mutateTranspose(transposeC).mutateTranspose(newTranspose = false, flagOnly = true)

    // determine parameters and check for compatibility
    val (a, b, side) = (M,N) match {
      case (b: GPUMatrix, a: GPUTriangularMatrix) =>
        assert(b.numCols == a.size, s"mismatched matrix dimensions: got ${b.numCols} != ${a.size}")
        assert(a.size == C.numCols, s"mismatched matrix dimensions: got ${a.size} != ${C.numCols}")
        assert(b.numRows == C.numRows, s"mismatched matrix dimensions: got ${b.numRows} != ${C.numRows}")
        (a, b, Right)
      case (a: GPUTriangularMatrix, b: GPUMatrix) =>
        assert(a.size == b.numRows, s"mismatched matrix dimensions: got ${a.size} != ${b.numRows}")
        assert(a.size == C.numRows, s"mismatched matrix dimensions: got ${a.size} != ${C.numRows}")
        assert(b.numCols == C.numCols, s"mismatched matrix dimensions: got ${b.numCols} != ${C.numCols}")
        (a, b, Left)
    }
    assert(a.constant.isEmpty || b.constant.isEmpty, s"unsupported: only one input constant can be defined")

    // determine constants
    val alpha = a.constant.getOrElse(b.constant.getOrElse(Waterfall.Constants.one))

    // set stream
    cublasSetStream(Waterfall.cublasHandle, stream.cudaStream_t).checkJCublasStatus()

    // perform single-precision triangular matrix-matrix multiplication
    cublasStrmm(Waterfall.cublasHandle,
      side.toSideId, a.fillMode.toFillModeId,
      a.isTranspose.toTransposeOpId,
      false.toDiagUnitId, // Waterfall doesn't track triangular matrices with unit diagonals, so assume that is false
      C.numRows, C.numCols,
      alpha.ptr,
      a.ptr, a.leadingDimension,
      b.ptr, b.leadingDimension,
      C.ptr, C.leadingDimension
    ).checkJCublasStatus()

    // return result
    C.mutateConstant(None).mutateTranspose(transposeC)
  }

  private def executeStrsm(M: GPUArray, N: GPUArray, C: GPUMatrix, transposeC: Boolean = false)(implicit stream: GPUStream = Stream.default) = {
    // prepare output
    C.mutateTranspose(transposeC).mutateTranspose(newTranspose = false, flagOnly = true)

    // determine parameters and check for compatibility
    val (a, b, side) = (M,N) match {
      case (b: GPUMatrix, a: GPUInverseTriangularMatrix) =>
        assert(b.numCols == a.size, s"mismatched matrix dimensions: got ${b.numCols} != ${a.size}")
        assert(a.size == C.numCols, s"mismatched matrix dimensions: got ${a.size} != ${C.numCols}")
        assert(b.numRows == C.numRows, s"mismatched matrix dimensions: got ${b.numRows} != ${C.numRows}")
        (a, b, Right)
      case (a: GPUInverseTriangularMatrix, b: GPUMatrix) =>
        assert(a.size == b.numRows, s"mismatched matrix dimensions: got ${a.size} != ${b.numRows}")
        assert(a.size == C.numRows, s"mismatched matrix dimensions: got ${a.size} != ${C.numRows}")
        assert(b.numCols == C.numCols, s"mismatched matrix dimensions: got ${b.numCols} != ${C.numCols}")
        (a, b, Left)
    }

    // determine constants
    val alpha = b.constant.getOrElse(Waterfall.Constants.one)

    // if not in-place, copy to output
    if(!(b.ptr eq C.ptr)) b.copyTo(C)

    // set stream
    cublasSetStream(Waterfall.cublasHandle, stream.cudaStream_t).checkJCublasStatus()

    // perform single-precision triangular matrix-matrix multiplication
    cublasStrsm(Waterfall.cublasHandle,
      side.toSideId, a.fillMode.toFillModeId,
      a.isTranspose.toTransposeOpId,
      false.toDiagUnitId, // Waterfall doesn't track triangular matrices with unit diagonals, so assume that is false
      C.numRows, C.numCols,
      alpha.ptr,
      a.ptr, a.leadingDimension,
      C.ptr, C.leadingDimension
    ).checkJCublasStatus()

    // return result
    C.mutateConstant(None).mutateTranspose(transposeC)
  }

  private def executeSpotrs(M: GPUArray, N: GPUArray, C: GPUMatrix, transposeC: Boolean = false)(implicit stream: GPUStream = Stream.default) = {
    // prepare output
    C.mutateTranspose(transposeC).mutateTranspose(newTranspose = false, flagOnly = true)

    // determine parameters and check for compatibility
    val (ainv, b, side) = (M,N) match {
      case (b: GPUMatrix, ainv: GPUInverseSymmetricMatrix) =>
        assert(b.numCols == ainv.size, s"mismatched matrix dimensions: got ${b.numCols} != ${ainv.size}")
        assert(ainv.size == C.numCols, s"mismatched matrix dimensions: got ${ainv.size} != ${C.numCols}")
        assert(b.numRows == C.numRows, s"mismatched matrix dimensions: got ${b.numRows} != ${C.numRows}")
        (ainv, b, Right)
      case (ainv: GPUInverseSymmetricMatrix, b: GPUMatrix) =>
        assert(ainv.size == b.numRows, s"mismatched matrix dimensions: got ${ainv.size} != ${b.numRows}")
        assert(ainv.size == C.numRows, s"mismatched matrix dimensions: got ${ainv.size} != ${C.numRows}")
        assert(b.numCols == C.numCols, s"mismatched matrix dimensions: got ${b.numCols} != ${C.numCols}")
        (ainv, b, Left)
    }
    val firstTransposeOp = side match {case Left => true; case Right => false}

    // get Cholesky
    val R = ainv.underlyingCholesky
    val effectiveFirstTransposeOp = if(!R.isTranspose) firstTransposeOp else !firstTransposeOp

    // determine constants
    val alpha = b.constant.getOrElse(Waterfall.Constants.one)

    // if not in-place, copy to output
    if(!(b.ptr eq C.ptr)) b.copyTo(C)

    // set stream
    cublasSetStream(Waterfall.cublasHandle, stream.cudaStream_t).checkJCublasStatus()

    // perform single-precision triangular solve matrix, first one on transpose of Cholesky
    cublasStrsm(Waterfall.cublasHandle,
      side.toSideId, R.fillMode.toFillModeId,
      effectiveFirstTransposeOp.toTransposeOpId,
      false.toDiagUnitId, // Waterfall doesn't track triangular matrices with unit diagonals, so assume that is false
      C.numRows, C.numCols,
      alpha.ptr,
      R.ptr, R.leadingDimension,
      C.ptr, C.leadingDimension
    ).checkJCublasStatus()

    // perform single-precision triangular solve matrix, second one not on transpose of Cholesky
    cublasStrsm(Waterfall.cublasHandle,
      side.toSideId, R.fillMode.toFillModeId,
      (!effectiveFirstTransposeOp).toTransposeOpId,
      false.toDiagUnitId, // Waterfall doesn't track triangular matrices with unit diagonals, so assume that is false
      C.numRows, C.numCols,
      Waterfall.Constants.one.ptr, // constant was already used in previous triangular solve
      R.ptr, R.leadingDimension,
      C.ptr, C.leadingDimension
    ).checkJCublasStatus()

    // return result
    C.mutateConstant(None).mutateTranspose(transposeC)
  }

  private def executeSpotrf(A: GPUSymmetricMatrix, uncheckedC: GPUMatrix, ws: CholeskyWorkspace)(implicit stream: GPUStream = Stream.default) = {
    // check if output has been properly declared triangular
    assert(uncheckedC.isInstanceOf[GPUTriangularMatrix], s"unsupported: output of Cholesky factorization needs to be declared triangular")
    val C = uncheckedC.asInstanceOf[GPUTriangularMatrix]

    // prepare output
    if(!(A.ptr eq C.ptr)) C.mutateConstant(None) // don't invalidate constant check for A if performed in-place
    C.mutateTranspose(newTranspose = false)

    // check for compatibility
    assert(A.size == C.size, s"mismatched matrix dimensions: got ${A.size} != ${C.size}")
    assert(A.constant.isEmpty, s"unsupported: cannot compute Cholesky with attached constants")
    assert(A.fillMode eq C.fillMode, s"unsupported: Cholesky input and output must have same fill mode")

    // if not in-place, copy to output, and attach Cholesky decomposition to original matrix
    if(!(A.ptr eq C.ptr)) {
      A.copyTo(C)
      A.attachCholesky(C, ws)
    }

    // set stream
    cusolverDnSetStream(Waterfall.cusolverDnHandle, stream.cudaStream_t).checkJCusolverStatus()

    // perform single-precision positive definite triangular factorization
    cusolverDnSpotrf(Waterfall.cusolverDnHandle,
      C.fillMode.toFillModeId,
      C.size,
      C.ptr, C.leadingDimension,
      ws.workspace,
      ws.workspaceNumBytes,
      ws.devInfo
    ).checkJCusolverStatus()

    // TODO: zero out lower triangle after computation, else addition may behave strangely

    // TODO: check devInfo to ensure non-singularity - need to be careful here when using CUDA streams because devInfo won't be ready immediately

    // return result
    C
  }
}
