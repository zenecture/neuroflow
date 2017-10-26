package neuroflow.cuda

import jcuda.driver.CUcontext
import jcuda.driver.JCudaDriver._

/**
  * @author dlwh
  **/
class CuContext(val context: CUcontext) extends AnyVal {
  def withPush[T](t: =>T) = {
    cuCtxPushCurrent(context)
    try {
      t
    } finally {
      cuCtxPopCurrent(context)
    }

  }
}

object CuContext {
  def ensureContext(implicit device: CuDevice) = {

    // Try to obtain the current context
    var context: CUcontext = new CUcontext
    cuCtxGetCurrent(context)

    // If the context is 'null', then a new context
    // has to be created.
    val nullContext: CUcontext = new CUcontext

    if (context == nullContext) {
      context = new CUcontext
      cuCtxCreate(context, 0, device.device)
    }

    new CuContext(context)

  }
}
