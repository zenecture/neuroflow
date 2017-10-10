package neuroflow.nets.gpu.cuda

import jcuda.driver.{CUfunction, CUmodule}
import jcuda.driver.JCudaDriver._
import breeze.macros.arityize
import java.io.{ByteArrayOutputStream, InputStream}
import jcuda.{CudaException, Pointer}

/**
  * Wrapper around the [[jcuda.driver.CUmodule]] apis
  *
  * CuModule *owns* the module handle, and will delete it on finalize.
  *
  * @author dlwh
  **/
class CuModule(val module: CUmodule) {

  @arityize(20)
  def getKernel[@arityize.replicate T](name: String): (CuKernel[T@arityize.replicate]@arityize.relative(getKernel)) = {
    val fn = new CUfunction
    try {
      cuModuleGetFunction(fn, module, name)
    } catch {
      case ex: CudaException if ex.getMessage == "CUDA_ERROR_NOT_FOUND" =>
        throw new RuntimeException(s"couldn't load $name", ex)
      case ex: CudaException =>
        throw new RuntimeException(s"while loading $name", ex)
    }
    new (CuKernel[T@arityize.replicate]@arityize.relative(getKernel))(this, fn)
  }


  private var released = false

  override protected def finalize() {
    super.finalize()
    release()
  }

  def release() {
    if (!released) {
      released = true
      cuModuleUnload(module)
    }
  }

}

object CuModule {
  def apply(ptx: InputStream)(implicit ctxt: CuContext = CuContext.ensureContext): CuModule = {
    val data = loadData(ptx)
    val module = new CUmodule()
    cuModuleLoadDataEx(module, Pointer.to(data), 0, Array.empty[Int], Pointer.to(new Array[Int](0)))
    new CuModule(module)
  }

  /**
    *
    * From JCuda under MIT
    *
    * Reads the data from the given inputStream and returns it as
    * a 0-terminated byte array. The caller is responsible to
    * close the given stream.
    *
    * @param inputStream The inputStream to read
    * @return The data from the inputStream
    * @throws CudaException If there is an IO error
    */
  private def loadData(inputStream: InputStream): Array[Byte] = {
    val baos: ByteArrayOutputStream = new ByteArrayOutputStream
    try {
      val buffer = new Array[Byte](8192)
      var done = false
      while (!done) {
        val read: Int = inputStream.read(buffer)
        if (read == -1) {
          done = true
        } else {
          baos.write(buffer, 0, read)
        }
      }
      baos.write('\0')
      baos.flush()
      baos.toByteArray
    } finally {
      baos.close()
    }
  }
}
