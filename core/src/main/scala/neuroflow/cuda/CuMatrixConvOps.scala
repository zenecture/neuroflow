package neuroflow.cuda

import jcuda.Pointer
import neuroflow.common.Logs

import scala.reflect.ClassTag

/**
  * @author bogdanski
  * @since 10.10.17
  */
trait CuMatrixConvOps extends Logs { this: CuMatrix.type =>

  class ConvOpsKernelBroker[V: ClassTag](typeName: String) {

    private val module: CuModule = {
      debug(s"Loading module: matrix_convops_$typeName.ptx")
      CuModule(getClass.getResourceAsStream(s"/cuda/matrix_convops_$typeName.ptx"))
    }

    private val kernel_convolute = module.getKernel13[Pointer, Pointer, Pointer, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int](s"convolute_$typeName")
    private val kernel_convolute_bp = module.getKernel11[Pointer, Int, Pointer, Int, Pointer, Int, Int, Int, Int, Int, Int](s"convolute_bp_$typeName")
    private val kernel_reshape_batch = module.getKernel11[Pointer, Int, Pointer, Int, Pointer, Int, Int, Int, Int, Int, Int](s"reshape_batch_$typeName")
    private val kernel_reshape_batch_bp = module.getKernel11[Pointer, Int, Pointer, Int, Pointer, Int, Int, Int, Int, Int, Int](s"reshape_batch_bp_$typeName")

    def convolute(in: CuMatrix[V])(implicit context: CuContext = CuContext.ensureContext): CuMatrix[V] = {

      val threads = (4, 4, 4)
      val blocks = (4, 4, 4)
//      kernel_convolute(blocks, threads)(in)
      ???

    }

    def convolute_bp(in: CuMatrix[V])(implicit context: CuContext = CuContext.ensureContext): CuMatrix[V] = {

      val threads = (4, 4, 4)
      val blocks = (4, 4, 4)
//      kernel_convolute_bp(blocks, threads)(in)
      ???

    }

    def reshape_batch(in: CuMatrix[V])(implicit context: CuContext = CuContext.ensureContext): CuMatrix[V] = {

      val threads = (4, 4, 4)
      val blocks = (4, 4, 4)
//      kernel_convolute(blocks, threads)(in)
      ???

    }

    def reshape_batch_bp(in: CuMatrix[V])(implicit context: CuContext = CuContext.ensureContext): CuMatrix[V] = {

      val threads = (4, 4, 4)
      val blocks = (4, 4, 4)
//      kernel_convolute(blocks, threads)(in)
      ???

    }

  }

}
