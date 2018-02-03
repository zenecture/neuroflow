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

    private val kernel_convolute = module.getKernel14[Pointer, Pointer, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int](s"convolute_$typeName")
    private val kernel_convolute_bp = module.getKernel14[Pointer, Pointer, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int](s"convolute_bp_$typeName")
    private val kernel_reshape_batch = module.getKernel6[Pointer, Pointer, Int, Int, Int, Int](s"reshape_batch_$typeName")
    private val kernel_reshape_batch_bp = module.getKernel6[Pointer, Pointer, Int, Int, Int, Int](s"reshape_batch_bp_$typeName")


    def convolute(in: CuMatrix[V], IX: Int, IY: Int, X: Int, Y: Int, Z: Int, BS: Int,
                  FX: Int, FY: Int, SX: Int, SY: Int, PX: Int, PY: Int)(implicit context: CuContext = CuContext.ensureContext): CuMatrix[V] = {

      val XB = X * BS

      val out = CuMatrix.zeros[V](FX * FY * Z, XB * Y)

      val threads = (4, 4, 4)

      val blocks = (
        math.ceil(XB.toDouble / threads._1.toDouble).toInt,
        math.ceil(Y.toDouble / threads._2.toDouble).toInt,
        math.ceil(Z.toDouble / threads._3.toDouble).toInt
      )

      kernel_convolute(blocks, threads)(in.offsetPointer, out.offsetPointer, IX, IY, X, Y, Z, BS, FX, FY, SX, SY, PX, PY)

      out

    }

    def convolute_bp(in: CuMatrix[V], IX: Int, IY: Int, X: Int, Y: Int, Z: Int, BS: Int,
                     FX: Int, FY: Int, SX: Int, SY: Int, PX: Int, PY: Int)(implicit context: CuContext = CuContext.ensureContext): CuMatrix[V] = {

      val XB = X * BS

      val out = CuMatrix.zeros[V](FX * FY * Z, IX * IY * BS)

      val threads = (4, 4, 4)

      val blocks = (
        math.ceil(XB.toDouble / threads._1.toDouble).toInt,
        math.ceil(Y.toDouble / threads._2.toDouble).toInt,
        math.ceil(Z.toDouble / threads._3.toDouble).toInt
      )

      kernel_convolute_bp(blocks, threads)(in.offsetPointer, out.offsetPointer, IX, IY, X, Y, Z, BS, FX, FY, SX, SY, PX, PY)

      out

    }

    def reshape_batch(in: CuMatrix[V], X: Int, Y: Int, Z: Int, BS: Int)(implicit context: CuContext = CuContext.ensureContext): CuMatrix[V] = {

      val dimOut = (X * Y * Z, BS)
      val out = CuMatrix.zeros[V](dimOut._2, dimOut._1)

      val threads = (4, 4, 1)

      val blocks = (
        math.ceil(dimOut._1.toDouble / threads._1.toDouble).toInt,
        math.ceil(dimOut._2.toDouble / threads._2.toDouble).toInt,
        1
      )

      kernel_reshape_batch(blocks, threads)(in.offsetPointer, out.offsetPointer, X, Y, Z, BS)

      out

    }

    def reshape_batch_bp(in: CuMatrix[V], X: Int, Y: Int, Z: Int, BS: Int)(implicit context: CuContext = CuContext.ensureContext): CuMatrix[V] = {

      val out = CuMatrix.zeros[V](Z, X * Y * BS)

      val threads = (4, 4, 1)

      val blocks = (
        math.ceil((X * Y * Z).toDouble / threads._1.toDouble).toInt,
        math.ceil(BS.toDouble / threads._2.toDouble).toInt,
        1
      )

      kernel_reshape_batch_bp(blocks, threads)(in.offsetPointer, out.offsetPointer, X, Y, Z, BS)

      out

    }

  }

}

