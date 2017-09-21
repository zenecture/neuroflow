package neuroflow.nets.gpu.cuda

import jcuda.CudaException
import jcuda.driver.JCudaDriver._
import jcuda.driver.{CUfunction, CUstream}
import org.bridj.Pointer

object CuKernel {
  def invoke(fn: CUfunction, gridDims: Dim3, blockDims: Dim3, sharedMemoryBytes: Int = 0)(args: Any*)(implicit context: CuContext):Unit = {
    context.withPush {
      val params = setupKernelParameters(args:_*)
      val cudaParams = jcuda.Pointer.to(params.map(_.toCuPointer):_*)
      cuLaunchKernel(fn,
        gridDims.x, gridDims.y, gridDims.z,
        blockDims.x, blockDims.y, blockDims.z,
        sharedMemoryBytes, new CUstream(),
        cudaParams, null)
      jcuda.runtime.JCuda.cudaFreeHost(cudaParams)

    }
  }

  /**
    * from VecUtils
    * Create a pointer to the given arguments that can be used as
    * the parameters for a kernel launch.
    *
    * @param args The arguments
    * @return The pointer for the kernel arguments
    * @throws NullPointerException If one of the given arguments is
    *                              <code>null</code>
    * @throws CudaException If one of the given arguments has a type
    *                       that can not be passed to a kernel (that is, a type that is
    *                       neither primitive nor a { @link Pointer})
    */
  private def setupKernelParameters(args: Any*) = {
    import java.lang._
    val kernelParameters: Array[Pointer[_]] = new Array[Pointer[_]](args.length)
    for( (arg, i) <- args.zipWithIndex) {
      arg match {
        case null =>
          throw new NullPointerException("Argument " + i + " is null")
        case argPointer: CuPointer =>
          val pointer = Pointer.pointerToPointer(cupointerToPointer(argPointer))
          kernelParameters(i) = pointer
        case value: Byte =>
          val pointer = Pointer.pointerToByte(value)
          kernelParameters(i) = pointer
        case value: Short =>
          val pointer = Pointer.pointerToShort(value)
          kernelParameters(i) = pointer
        case value: Integer =>
          val pointer = Pointer.pointerToInt(value)
          kernelParameters(i) = pointer
        case value: Long =>
          val pointer = Pointer.pointerToLong(value)
          kernelParameters(i) = pointer
        case value: Float =>
          val pointer = Pointer.pointerToFloat(value)
          kernelParameters(i) = pointer
        case value: Double =>
          val pointer = Pointer.pointerToDouble(value)
          kernelParameters(i) = pointer
        case _ =>
          throw new RuntimeException("Type " + arg.getClass + " may not be passed to a function")
      }
    }
    kernelParameters
  }

}
