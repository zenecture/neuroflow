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

    private val im2col_kernel = module.getKernel12[Pointer, Pointer, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int](s"im2col_$typeName")

    def im2col(m: CuMatrix[V], layer: Int, dim: (Int, Int, Int), field: (Int, Int), padding: (Int, Int), stride: (Int, Int))
              (implicit context: CuContext = CuContext.ensureContext): CuMatrix[V] = {

      val dimOut = (
        (dim._1 + 2 * padding._1 - field._1) / stride._1 + 1,
        (dim._2 + 2 * padding._2 - field._2) / stride._2 + 1
      )

      val fieldSq = field._1 * field._2
      val out = (fieldSq * dim._3, dimOut._1 * dimOut._2)

      val res = CuMatrix.zeros[V](out._1, out._2)

      im2col_kernel(Dim3.default, Dim3.default, 0)(m.offsetPointer, res.offsetPointer, dim._1, dim._2, dim._3, layer, field._1, field._2, padding._1, padding._2, stride._1, stride._2)

      res

    }

    def im2col_backprop(m: CuMatrix[V], layer: Int, xyz: (Int, Int, Int))
              (implicit context: CuContext = CuContext.ensureContext): CuMatrix[V] = {
      ???
    }

  }

}
