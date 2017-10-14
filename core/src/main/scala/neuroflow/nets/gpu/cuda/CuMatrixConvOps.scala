package neuroflow.nets.gpu.cuda

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

    private val im2col_kernel = module.getKernel12[Pointer, Pointer, Pointer, Int, Int, Int, Int, Int, Int, Int, Int, Int](s"im2col_$typeName")
    private val im2col_bp_kernel = module.getKernel11[Pointer, Int, Pointer, Int, Pointer, Int, Int, Int, Int, Int, Int](s"im2col_backprop_$typeName")

    def im2col(m: CuMatrix[V], idc: CuMatrix[Int], dim: (Int, Int, Int), field: (Int, Int), padding: (Int, Int), stride: (Int, Int))
              (implicit context: CuContext = CuContext.ensureContext): (CuMatrix[V], CuMatrix[Int]) = {

      val dimOut = (
        (dim._1 + 2 * padding._1 - field._1) / stride._1 + 1,
        (dim._2 + 2 * padding._2 - field._2) / stride._2 + 1
      )

      val resSize = (field._1 * field._2 * dim._3, dimOut._1 * dimOut._2)
      val res = CuMatrix.zeros[V](resSize._1, resSize._2)

      val threads = (8, 8, 8)

      val blocks = (
        math.ceil(dimOut._1.toDouble / threads._1.toDouble).toInt,
        math.ceil(dimOut._2.toDouble / threads._2.toDouble).toInt,
        math.ceil(dim._3.toDouble / threads._3.toDouble).toInt
      )

      im2col_kernel(blocks, threads, 0)(m.offsetPointer, res.offsetPointer, idc.offsetPointer, dim._1, dim._2, dim._3, field._1, field._2, padding._1, padding._2, stride._1, stride._2)

      (res, idc)

    }

    def im2col_backprop(m: CuMatrix[V], idc: CuMatrix[Int], dim: (Int, Int, Int), field: (Int, Int))
              (implicit context: CuContext = CuContext.ensureContext): CuMatrix[V] = {

      val resSize = (field._1 * field._2 * dim._3, dim._1 * dim._2)
      val res = CuMatrix.zeros[V](resSize._1, resSize._2)

      val threads = (8, 8, 8)

      val blocks = (
        math.ceil(dim._1.toDouble / threads._1.toDouble).toInt,
        math.ceil(dim._2.toDouble / threads._2.toDouble).toInt,
        math.ceil(dim._3.toDouble / threads._3.toDouble).toInt
      )

      im2col_bp_kernel(blocks, threads)(m.offsetPointer, m.majorStride, res.offsetPointer, res.majorStride, idc.offsetPointer, idc.majorStride, dim._1, dim._2, dim._3, field._1, field._2)

      res

    }

  }

}
