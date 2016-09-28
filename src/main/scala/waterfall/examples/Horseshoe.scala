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
package waterfall.examples

import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter}
import jcuda.runtime.JCuda._
import jcuda.runtime.cudaMemcpyKind._
import jcuda.{Pointer, Sizeof}
import waterfall.Implicits.DebugImplicits
import waterfall.Random.PhiloxState
import waterfall.Stream.GPUStream
import waterfall._
import waterfall.MatrixProperties.checkDevInfo
import scala.collection.mutable
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

/**
  * The Horseshoe Probit Gibbs sampler (floating point precision version). See: http://arxiv.org/abs/1608.04329
  *
  * @author Alexander Terenin
  */
object Horseshoe extends App {
  val (nMC, n, p, thinning) = args match {
    case Array(a1) => (a1.toInt, 1000, 100, 1)
    case Array(a1, a2, a3) => (a1.toInt, a2.toInt, a3.toInt, 1)
    case Array(a1, a2, a3, a4) => (a1.toInt, a2.toInt, a3.toInt, a4.toInt)
    case Array() => (100, 1000, 100, 1)
  }
  val numStoredVariables = math.min(87, p)
  val seed = 1

  println(s"initializing")
  Waterfall.init()
  Waterfall.printMemInfo()

  def genData(beta: GPUVector, z: GPUVector) = {
    // generates a synthetic data set to be used with the model
    val nonZeroBeta = Array(1.3f,4.0f,-1.0f,1.6f,5.0f,-2.0f)

    val X = GPUMatrix.create(n,p)
    val y = new Pointer
    cudaMalloc(y, n.toLong * Sizeof.INT.toLong).checkJCudaStatus()
    val mu = GPUVector.create(n)
    cudaMemcpy(beta.ptr, Pointer.to(nonZeroBeta), nonZeroBeta.length*Sizeof.FLOAT, cudaMemcpyHostToDevice).checkJCudaStatus()

    val drawY = CustomKernelFile("cuY.ptx").loadCustomKernel("cuda_draw_y")
    val (gridSizeX,blockSizeX) = getDefaultLaunchConfiguration(n)

    X =: Random.normal
    mu =: X * beta
    z =: Random.uniform
    drawY(gridSizeX, blockSizeX)(Array(n),z,mu,y)

    // cleanup
    Stream.default.synchronize()
    cudaMemcpy(beta.ptr, Pointer.to(Array.fill(p)(0.0f)), p*Sizeof.FLOAT, cudaMemcpyHostToDevice).checkJCudaStatus()
    cudaMemcpy(z.ptr, Pointer.to(Array.fill(n)(0.0f)), n*Sizeof.FLOAT, cudaMemcpyHostToDevice).checkJCudaStatus()
    cudaFree(mu.ptr).checkJCudaStatus()

    (y,X)
  }




  // create variables
  val z = GPUVector.createFromArray(Array.fill(n)(0.0f))
  val beta = GPUVector.createFromArray(Array.fill(p)(0.0f))
  val data = genData(beta, z)
  val y = data._1
  val X = data._2

  val lambdaSqInv = GPUVector.createFromArray(Array.fill(p)(1.0f))
  val nuInv = GPUVector.createFromArray(Array.fill(p)(1.0f))
  val xiInv = GPUConstant.create(1.0f)
  val tauSqInv = GPUConstant.create(1.0f)
  val tauSqInvShape = GPUConstant.create((p.toFloat + 1.0f) / 2.0f)

  val XtX = GPUMatrix.create(p,p).declareSymmetric
  val XtXdiag = new GPUVector(XtX.ptr, p, stride = p+1) // HACK: store diagonal of XtX as a vector
  val Sigma = GPUMatrix.create(p,p).declareSymmetric
  val SigmaDiag = new GPUVector(Sigma.ptr, p, stride = p+1) // HACK: store diagonal of XtX as a vector
  val mu = GPUVector.create(p)
  val betaSqScratch = GPUVector.create(p)

  val ws = MatrixProperties.createCholeskyWorkspace(Sigma)
  val rngStateZ = Random.allocateDeviceRNGState(PhiloxState)
  val rngStateTau = Random.allocateDeviceRNGState(PhiloxState)
  val rngInit = CustomKernelFile("cuRANDINIT.ptx").loadCustomKernel("cuda_rand_init")
  rngInit(1,1)(Array(seed+1000), rngStateZ)
  rngInit(1,1)(Array(seed+2000), rngStateTau)

  val drawZ = CustomKernelFile("cuTNORM.ptx").loadCustomKernel("cuda_onesided_unitvar_tnorm")
  val drawLambda = CustomKernelFile("cuLAMBDA.ptx").loadCustomKernel("cuda_lambdaSqInv")
  val drawNu = CustomKernelFile("cuNUXI.ptx").loadCustomKernel("cuda_nuInvXiInv")
  val drawXi = drawNu
  val drawTau = CustomKernelFile("cuTAU.ptx").loadCustomKernel("cuda_tauSqInv")
  val squareBeta = CustomKernelFile("cuBETA.ptx").loadCustomKernel("cuda_betaSq")

  val betaStream = Stream.create
  val zStream = Stream.create
  val lambdaTauStream = Stream.create
  val nuStream = Stream.create
  val xiStream = Stream.create

  val betaOut = new mutable.Queue[Array[Float]]()
  val zOut = new mutable.Queue[Array[Float]]()
  val lambdaOut = new mutable.Queue[Array[Float]]()
  val nuOut = new mutable.Queue[Array[Float]]()
  val tauOut = new mutable.Queue[Array[Float]]()
  val xiOut = new mutable.Queue[Array[Float]]()

  XtX =: X.T * X
  Sigma.attachCholesky(Sigma.declareTriangular, ws)

  println("data loaded into GPU, starting MCMC")
  Waterfall.printMemInfo()
  val time = System.nanoTime()




  // run MCMC
  for(i <- 0 until nMC) {
    if(i % thinning == 0) {
      downloadOutput(numStoredVariables, z, zOut, zStream)
      downloadOutput(numStoredVariables, lambdaSqInv, lambdaOut, lambdaTauStream)
      downloadOutput(1, tauSqInv, tauOut, lambdaTauStream)
    }
    updateBeta()
    updateNu()
    updateXi()
    syncAllStreams()

    if(i % thinning == 0) {
      downloadOutput(numStoredVariables, beta, betaOut, betaStream)
      downloadOutput(numStoredVariables, nuInv, nuOut, nuStream)
      downloadOutput(1, xiInv, xiOut, xiStream)
    }
    updateZ()
    updateLambda()
    updateTau()
    syncAllStreams()

    if (i % math.max(nMC / 1000, 1) == 0) println(s"total samples: $i")
  }

  println(s"finished, total run time in minutes: ${(System.nanoTime() - time).toDouble / 60000000000.0}")

  writeOutput()




  // utility functions to transform output to the scale we want to analyze in
  val fInvSqrt = { v: Float => math.sqrt(1.0 / v.toDouble).toFloat }
  val fInv = { v: Float => 1.0f / v }
  val fIdentity = { v: Float => v }

  def writeOutput() {
    for {
      ((out, name), f) <- Array(betaOut, zOut, lambdaOut, nuOut, tauOut, xiOut)
        .zip(Array("beta", "z", "lambda", "nu", "tau", "xi"))
        .zip(Array(fIdentity, fIdentity, fInvSqrt, fInv, fInvSqrt, fInv))
    } {
      val fileName = s"output/out-GPU-$name.csv"
      val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fileName)))
      println(s"writing output of length ${out.size} to $fileName")
      out.foreach { outRow => writer.write(s"${outRow.map(f).mkString(",")}\n") }
      writer.close()
    }
  }

  def syncAllStreams() = {
    betaStream.synchronize()
    zStream.synchronize()
    lambdaTauStream.synchronize()
    nuStream.synchronize()
    xiStream.synchronize()
  }

  def downloadOutput(numVariables: Int, gpuArray: GPUArray, output: mutable.Queue[Array[Float]], stream: GPUStream) = {
    implicit val s = stream
    output.enqueue(gpuArray.copyToHost(numVariables, async = true))
  }

  def getDefaultLaunchConfiguration(size: Int) = {
    val blockSizeX = 256
    val gridSizeX = math.ceil(size.toDouble / blockSizeX.toDouble).toInt
    (gridSizeX, blockSizeX)
  }





  def updateLambda() = {
    implicit val stream = lambdaTauStream
    val (gridSizeX,blockSizeX) = getDefaultLaunchConfiguration(p)

    lambdaSqInv =: Random.uniform
    drawLambda(gridSizeX,blockSizeX)(Array(p), lambdaSqInv, beta, nuInv, tauSqInv)
  }

  def updateNu() = {
    implicit val stream = nuStream
    val (gridSizeX,blockSizeX) = getDefaultLaunchConfiguration(p)

    nuInv =: Random.uniform
    drawNu(gridSizeX,blockSizeX)(Array(p), nuInv, lambdaSqInv)
  }

  def updateXi() = {
    implicit val stream = xiStream

    xiInv =: Random.uniform
    drawXi(1,1)(Array(1),xiInv,tauSqInv)
  }

  def updateTau() = {
    implicit val stream = lambdaTauStream
    val (gridSizeX,blockSizeX) = getDefaultLaunchConfiguration(p)

    squareBeta(gridSizeX,blockSizeX)(Array(p),beta,betaSqScratch)
    tauSqInv =: (lambdaSqInv dot betaSqScratch)
    drawTau(1,32,32*Sizeof.FLOAT)(rngStateTau, tauSqInv, tauSqInvShape, xiInv)
  }

  def updateBeta() = {
    implicit val stream = betaStream

    XtX copyTo Sigma
    SigmaDiag =: SigmaDiag + lambdaSqInv
    Sigma.chol =: Sigma.computeCholesky(ws); Future { checkDevInfo(ws.devInfo) } onFailure { case t => throw t }
    beta =: Random.normal
    beta =: Sigma.chol.inv * beta
    mu =: X.T * z
    mu =: Sigma.inv * mu
    beta =: beta + mu
  }

  def updateZ() = {
    implicit val stream = zStream
    val (gridSizeX,blockSizeX) = getDefaultLaunchConfiguration(n)

    z =: X * beta
    drawZ(gridSizeX,blockSizeX)(Array(n),rngStateZ,z,y)
  }
}
