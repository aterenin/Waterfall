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

import java.nio.Buffer
import jcuda.Pointer
import jcuda.driver.{CUfunction, CUmodule}
import jcuda.driver.JCudaDriver.{cuLaunchKernel, cuModuleGetFunction, cuModuleLoad}
import Implicits.DebugImplicits
import waterfall.Random.GPURNGState
import waterfall.Stream.GPUStream

class CustomKernel(module: CUmodule, function: CUfunction,
                   args: Option[Pointer] = None,
                   launchConfiguration: Option[(Int,Int,Int,Int,Int,Int)] = None,
                   sharedMemBytes: Option[Int] = None,
                   extra: Option[Pointer] = None) {
  def apply(gridX: Int, blockX: Int)(args: Any*): Unit = apply(gridX,1,1,blockX,1,1, 0)(args:_*)
  def apply(gridX: Int, blockX: Int, shareMemBytes: Int)(args: Any*): Unit = apply(gridX,1,1,blockX,1,1, shareMemBytes)(args:_*)
  def apply(gridX: Int, gridY: Int, blockX: Int, blockY: Int)(args: Any*): Unit = apply(gridX,gridY,1,blockX,blockY,1, 0)(args:_*)
  def apply(gridX: Int, gridY: Int, blockX: Int, blockY: Int, sharedMemBytes: Int)(args: Any*): Unit = apply(gridX,gridY,1,blockX,blockY,1,sharedMemBytes)(args:_*)
  def apply(gridX: Int, gridY: Int, gridZ: Int, blockX: Int, blockY: Int, blockZ: Int)(args: Any*): Unit = apply(gridX,gridY,gridZ,blockX,blockY,blockZ, 0)(args:_*)
  def apply(gridX: Int, gridY: Int, gridZ: Int, blockX: Int, blockY: Int, blockZ: Int, shareMemBytes: Int)(args: Any*) = {
    // this allows us to use syntax such as customKernel(256,4)(a1,a2,a3) for launching kernels, mimicking CUDA C
    withLaunchConfiguration(gridX, gridY, gridZ, blockX, blockY, blockZ)
      .withSharedMemBytes(0)
      .withArgs(args:_*)
      .execute()
  }
  def apply(args: Any*) = withArgs(args:_*).execute()


  def withArgs(args: Any*) = {
    // CUDA kernel arguments take the form Pointer.to( Pointer.to(g1), Pointer.to(h2) ) where g1 is a pointer to the GPU, and h2 is a pointer (or Java object) on the host
    val argPointerSeq = args.map{
      case p: Pointer => Pointer.to(p)
      case g: GPUArray => Pointer.to(g.ptr)
      case r: GPURNGState => Pointer.to(r.ptr)
      case h1: Array[Byte] => Pointer.to(h1)
      case h2: Array[Char] => Pointer.to(h2)
      case h3: Array[Short] => Pointer.to(h3)
      case h4: Array[Int] => Pointer.to(h4)
      case h5: Array[Float] => Pointer.to(h5)
      case h6: Array[Long] => Pointer.to(h6)
      case h7: Array[Double] => Pointer.to(h7)
      case b: Buffer => Pointer.toBuffer(b)
      case e => throw new Exception(s"unsupported: cannot create pointer for kernel argument $e")
    }
    updated(args = Some(Pointer.to(argPointerSeq:_*)))
  }

  def withLaunchConfiguration(gridX: Int, blockX: Int): CustomKernel = withLaunchConfiguration(gridX,1,1,blockX,1,1)
  def withLaunchConfiguration(gridX: Int, gridY: Int, blockX: Int, blockY: Int): CustomKernel = withLaunchConfiguration(gridX,gridY,1,blockX,blockY,1)
  def withLaunchConfiguration(gridX: Int, gridY: Int, gridZ: Int, blockX: Int, blockY: Int, blockZ: Int) = updated(launchConfiguration = Some(gridX, gridY, gridZ, blockX, blockY, blockZ))

  def withSharedMemBytes(sharedMemBytes: Int) = updated(sharedMemBytes = Some(sharedMemBytes))

  def withExtra(extra: Pointer) = updated(extra = Some(extra))

  def execute()(implicit stream: GPUStream = Stream.default): Unit = {
    val (gX, gY, gZ, bX, bY, bZ) = launchConfiguration.getOrElse(throw new Exception(s"unsupported: tried to launch custom kernel without setting launch configuration"))
    cuLaunchKernel(function, gX, gY, gZ, bX, bY, bZ,sharedMemBytes.getOrElse(0),stream.CUstream,args.orNull,extra.orNull).checkJCudaStatus()

  }

  private def updated(module: CUmodule = module, function: CUfunction = function,
                      args: Option[Pointer] = args,
                      launchConfiguration: Option[(Int,Int,Int,Int,Int,Int)] = launchConfiguration,
                      sharedMemBytes: Option[Int] = sharedMemBytes,
                      extra: Option[Pointer] = extra) = new CustomKernel(module, function, args, launchConfiguration, sharedMemBytes, extra)
}

case class CustomKernelFile(fileName: String) {
  private val module = new CUmodule()
  cuModuleLoad(module, fileName).checkJCudaStatus()

  def loadCustomKernel(kernelName: String) = {
    val function = new CUfunction()
    cuModuleGetFunction(function, module, kernelName).checkJCudaStatus()
    new CustomKernel(module, function)
  }
}