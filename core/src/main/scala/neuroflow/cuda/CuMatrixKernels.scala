package neuroflow.cuda

import breeze.generic.UFunc
import jcuda.Pointer

import scala.reflect.ClassTag
import java.util.concurrent.ConcurrentHashMap

import breeze.linalg.{BroadcastedColumns, BroadcastedRows}
import neuroflow.common.Logs
import neuroflow.core._
import neuroflow.dsl.Convolution

/**
  * @author dlwh
  * @author bogdanski
  **/
trait CuMatrixKernels extends Logs { this: CuMatrix.type =>

  /* Default UFuncs */
  class KernelBroker[T: ClassTag](typeName: String) {

    private val module: CuModule = {
      debug(s"Loading module: matrix_kernels_$typeName.ptx")
      CuModule(getClass.getResourceAsStream(s"/cuda/matrix_kernels_$typeName.ptx"))
    }

    private val implCache = new ConcurrentHashMap[String, CuKernel6[Int, Int, Pointer, Int, Pointer, Int]]
    private val impl2Cache = new ConcurrentHashMap[String, CuKernel8[Int, Int, Pointer, Int, Pointer, Int, Pointer, Int]]
    private val impl2TransCache = new ConcurrentHashMap[String, CuKernel8[Int, Int, Pointer, Int, Pointer, Int, Pointer, Int]]
    private val impl2VSCache = new ConcurrentHashMap[String, CuKernel7[Int, Int, Pointer, Int, Pointer, Int, T]]
    private val impl2SVCache = new ConcurrentHashMap[String, CuKernel7[Int, Int, Pointer, Int, T, Pointer, Int]]
    private val reduceCache = new ConcurrentHashMap[String, CuKernel5[Int, Int, Pointer, Pointer, Int]]
    private val colReduceCache = new ConcurrentHashMap[String, CuKernel5[Int, Int, Pointer, Pointer, Int]]
    private val rowReduceCache = new ConcurrentHashMap[String, CuKernel5[Int, Int, Pointer, Pointer, Int]]

    def implFor[K <: UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext): UFunc.UImpl[K, CuMatrix[T], CuMatrix[T]] = {
      var kern = implCache.get(funName)
      if (kern == null) {
        kern = module.getKernel6[Int, Int, Pointer, Int, Pointer, Int](s"map_${funName}_$typeName")
        implCache.put(funName, kern)
      }


      new UFunc.UImpl[K, CuMatrix[T], CuMatrix[T]] {
        def apply(v: CuMatrix[T]): CuMatrix[T] = {

          val res = if (v.isTranspose) CuMatrix.create[T](v.cols, v.rows).t else CuMatrix.create[T](v.rows, v.cols)
          val minorSize = if (v.isTranspose) v.cols else v.rows
          kern((512, 20), (32, 1, 1))(minorSize, v.majorSize, res.offsetPointer, res.majorStride, v.offsetPointer, v.majorStride)
          res
        }
      }
    }

    def reducerFor[K <: UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext): UFunc.UImpl[K, CuMatrix[T], T] = {
      var kern = reduceCache.get(funName)
      if (kern == null) {
        kern = module.getKernel5[Int, Int, Pointer, Pointer, Int](s"reduce_${funName}_$typeName")
        reduceCache.put(funName, kern)
      }

      val byteSize = org.bridj.BridJ.sizeOf(implicitly[ClassTag[T]].runtimeClass)


      new UFunc.UImpl[K, CuMatrix[T], T] {
        def apply(v: CuMatrix[T]): T = {

          val tmpRows = 20
          val tmpCols = 512
          val tmp = CuMatrix.create[T](tmpRows, tmpCols)
          val minorSize = if (v.isTranspose) v.cols else v.rows
          kern((tmpCols, tmpRows), (32, 1), 32 * 1 * byteSize.toInt)(minorSize, v.majorSize, tmp.offsetPointer, v.offsetPointer, v.majorStride)
          kern(1, (32, 1))(tmpCols * tmpRows, 1, tmp.offsetPointer, tmp.offsetPointer, 1)
          tmp(0 to 0, 0 to 0).toDense.apply(0, 0)
        }
      }
    }

    def colReducerFor[K <: UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext): UFunc.UImpl[K, BroadcastedColumns[CuMatrix[T], CuMatrix[T]], CuMatrix[T]] = {
      var kern = colReduceCache.get(funName)
      if (kern == null) {
        kern = module.getKernel5[Int, Int, Pointer, Pointer, Int](s"reduce_col_${funName}_$typeName")
        colReduceCache.put(funName, kern)
      }

      val byteSize = org.bridj.BridJ.sizeOf(implicitly[ClassTag[T]].runtimeClass)


      new UFunc.UImpl[K, BroadcastedColumns[CuMatrix[T], CuMatrix[T]], CuMatrix[T]] {
        def apply(vx: BroadcastedColumns[CuMatrix[T], CuMatrix[T]]) = {
          val v = vx.underlying

          val tmp = CuMatrix.create[T](1, v.cols)
          val minorSize = if (v.isTranspose) v.cols else v.rows
          kern((512, 20), (32, 1), 32 * 1 * byteSize.toInt)(minorSize, v.majorSize, tmp.offsetPointer, v.offsetPointer, v.majorStride)
          tmp
        }
      }
    }

    def rowReducerFor[K <: UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext): UFunc.UImpl[K, BroadcastedRows[CuMatrix[T], CuMatrix[T]], CuMatrix[T]] = {
      var kern = rowReduceCache.get(funName)
      if (kern == null) {
        kern = module.getKernel5[Int, Int, Pointer, Pointer, Int](s"reduce_row_${funName}_$typeName")
        rowReduceCache.put(funName, kern)
      }

      val byteSize = org.bridj.BridJ.sizeOf(implicitly[ClassTag[T]].runtimeClass)


      new UFunc.UImpl[K, BroadcastedRows[CuMatrix[T], CuMatrix[T]], CuMatrix[T]] {
        def apply(vx: BroadcastedRows[CuMatrix[T], CuMatrix[T]]) = {
          val v = vx.underlying

          val tmp = CuMatrix.create[T](v.rows, 1)
          val minorSize = if (v.isTranspose) v.cols else v.rows
          kern((512, 1), (32, 1), 32 * 1 * byteSize.toInt)(minorSize, v.majorSize, tmp.offsetPointer, v.offsetPointer, v.majorStride)
          tmp
        }
      }
    }

    def inPlaceImplFor[K <: UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext): UFunc.InPlaceImpl[K, CuMatrix[T]] = {
      var kern = implCache.get(funName)
      if (kern == null) {
        kern = module.getKernel6[Int, Int, Pointer, Int, Pointer, Int](s"map_${funName}_$typeName")
        implCache.put(funName, kern)
      }


      new UFunc.InPlaceImpl[K, CuMatrix[T]] {
        def apply(v: CuMatrix[T]) = {
          val minorSize = if (v.isTranspose) v.cols else v.rows
          kern((512, 20), (32, 1))(minorSize, v.majorSize, v.offsetPointer, v.majorStride, v.offsetPointer, v.majorStride)
        }
      }
    }

    def impl2For[K <: UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext): UFunc.UImpl2[K, CuMatrix[T], CuMatrix[T], CuMatrix[T]] = {
      var kern = impl2Cache.get(funName)
      if (kern == null) {
        kern = module.getKernel8[Int, Int, Pointer, Int, Pointer, Int, Pointer, Int](s"map2_${funName}_$typeName")
        impl2Cache.put(funName, kern)
      }

      var transKern = impl2TransCache.get(funName)
      if (transKern == null) {
        transKern = module.getKernel8[Int, Int, Pointer, Int, Pointer, Int, Pointer, Int](s"map2_transpose_${funName}_$typeName")
        impl2TransCache.put(funName, transKern)
      }


      new UFunc.UImpl2[K, CuMatrix[T], CuMatrix[T], CuMatrix[T]] {
        def apply(v: CuMatrix[T], v2: CuMatrix[T]): CuMatrix[T] = {
          require(v.rows == v2.rows && v.cols == v2.cols, "Dimension mismatch!")
          val res = if (v.isTranspose) CuMatrix.create[T](v.cols, v.rows).t else CuMatrix.create[T](v.rows, v.cols)
          val minorSize = if (v.isTranspose) v.cols else v.rows
          if (v.isTranspose != v2.isTranspose) {
            transKern((512, 30), (32, 8))(minorSize, v.majorSize, res.offsetPointer, res.majorStride, v.offsetPointer, v.majorStride, v2.offsetPointer, v2.majorStride)
          } else {
            kern((512, 20), (32, 1))(minorSize, v.majorSize, res.offsetPointer, res.majorStride, v.offsetPointer, v.majorStride, v2.offsetPointer, v2.majorStride)
          }

          res
        }
      }
    }

    def inPlaceImpl2For[K <: UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext): UFunc.InPlaceImpl2[K, CuMatrix[T], CuMatrix[T]] = {
      var kern = impl2Cache.get(funName)
      if (kern == null) {
        kern = module.getKernel8[Int, Int, Pointer, Int, Pointer, Int, Pointer, Int](s"map2_${funName}_$typeName")
        impl2Cache.put(funName, kern)
      }

      var transKern = impl2TransCache.get(funName)
      if (transKern == null) {
        transKern = module.getKernel8[Int, Int, Pointer, Int, Pointer, Int, Pointer, Int](s"map2_transpose_${funName}_$typeName")
        impl2TransCache.put(funName, transKern)
      }


      new UFunc.InPlaceImpl2[K, CuMatrix[T], CuMatrix[T]] {
        def apply(v: CuMatrix[T], v2: CuMatrix[T]) {
          require(v.rows == v2.rows && v.cols == v2.cols, "Dimension mismatch!")
          val minorSize = if (v.isTranspose) v.cols else v.rows
          if (v.isTranspose != v2.isTranspose) {
            transKern((512, 30), (32, 8))(minorSize, v.majorSize, v.offsetPointer, v.majorStride, v.offsetPointer, v.majorStride, v2.offsetPointer, v2.majorStride)
          } else {
            kern((512, 20), (32, 1, 1))(minorSize, v.majorSize, v.offsetPointer, v.majorStride, v.offsetPointer, v.majorStride, v2.offsetPointer, v2.majorStride)
          }
        }
      }
    }

    def impl2For_v_s[K <: UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext): UFunc.UImpl2[K, CuMatrix[T], T, CuMatrix[T]] = {
      var kern = impl2VSCache.get(funName)
      if (kern == null) {
        kern = module.getKernel7[Int, Int, Pointer, Int, Pointer, Int, T](s"map2_v_s_${funName}_$typeName")
        impl2VSCache.put(funName, kern)
      }


      new UFunc.UImpl2[K, CuMatrix[T], T, CuMatrix[T]] {
        def apply(v: CuMatrix[T], v2: T): CuMatrix[T] = {

          val res = if (v.isTranspose) CuMatrix.create[T](v.cols, v.rows).t else CuMatrix.create[T](v.rows, v.cols)
          val minorSize = if (v.isTranspose) v.cols else v.rows
          kern((512, 20), (32, 1, 1))(minorSize, v.majorSize, res.offsetPointer, res.majorStride, v.offsetPointer, v.majorStride, v2)
          res
        }
      }
    }

    def inPlaceImpl2For_v_s[K <: UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext): UFunc.InPlaceImpl2[K, CuMatrix[T], T] = {
      var kern = impl2VSCache.get(funName)
      if (kern == null) {
        kern = module.getKernel7[Int, Int, Pointer, Int, Pointer, Int, T](s"map2_v_s_${funName}_$typeName")
        impl2VSCache.put(funName, kern)
      }


      new UFunc.InPlaceImpl2[K, CuMatrix[T], T] {
        def apply(v: CuMatrix[T], v2: T) = {

          val res = v
          val minorSize = if (v.isTranspose) v.cols else v.rows
          kern((512, 20), (32, 1, 1))(minorSize, v.majorSize, res.offsetPointer, res.majorStride, v.offsetPointer, v.majorStride, v2)
        }
      }
    }

    def impl2For_s_v[K <: UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext): UFunc.UImpl2[K, T, CuMatrix[T], CuMatrix[T]] = {
      var kern = impl2SVCache.get(funName)
      if (kern == null) {
        kern = module.getKernel7[Int, Int, Pointer, Int, T, Pointer, Int](s"map2_s_v_${funName}_$typeName")
        impl2SVCache.put(funName, kern)
      }


      new UFunc.UImpl2[K, T, CuMatrix[T], CuMatrix[T]] {
        def apply(v2: T, v: CuMatrix[T]): CuMatrix[T] = {

          val res = if (v.isTranspose) CuMatrix.create[T](v.cols, v.rows).t else CuMatrix.create[T](v.rows, v.cols)
          val minorSize = if (v.isTranspose) v.cols else v.rows
          kern((512, 20), (32, 1, 1))(minorSize, v.majorSize, res.offsetPointer, res.majorStride, v2, v.offsetPointer, v.majorStride)
          res
        }
      }
    }
  }

  /* ConvOps */
  class ConvOpsKernelBroker[V: ClassTag](typeName: String) {

    private val module: CuModule = {
      debug(s"Loading module: matrix_convops_$typeName.ptx")
      CuModule(getClass.getResourceAsStream(s"/cuda/matrix_convops_$typeName.ptx"))
    }

    private val kernel_convolute = module.getKernel14[Pointer, Pointer, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int](s"convolute_$typeName")
    private val kernel_convolute_bp = module.getKernel14[Pointer, Pointer, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int](s"convolute_bp_$typeName")
    private val kernel_reshape_batch = module.getKernel6[Pointer, Pointer, Int, Int, Int, Int](s"reshape_batch_$typeName")
    private val kernel_reshape_batch_bp = module.getKernel6[Pointer, Pointer, Int, Int, Int, Int](s"reshape_batch_bp_$typeName")


    def convoluteImpl(implicit context: CuContext = CuContext.ensureContext): convolute.Impl3[CuMatrix[V], Convolution[V], Int, CuMatrix[V]] = {

      new convolute.Impl3[CuMatrix[V], Convolution[V], Int, CuMatrix[V]] {
        def apply(in: CuMatrix[V], l: Convolution[V], bs: Int): CuMatrix[V] = {

          val (ix, iy, x, y, z, fx, fy, sx, sy, px, py) =
            (l.dimIn._1, l.dimIn._2, l.dimOut._1, l.dimOut._2,
              l.dimIn._3, l.field._1, l.field._2, l.stride._1,
              l.stride._2, l.padding._1, l.padding._2)

          val xb = x * bs

          val out = CuMatrix.zeros[V](fx * fy * z, xb * y)

          val threads = (4, 4, 4)

          val blocks = (
            math.ceil(xb.toDouble / threads._1.toDouble).toInt,
            math.ceil(y.toDouble / threads._2.toDouble).toInt,
            math.ceil(z.toDouble / threads._3.toDouble).toInt
          )

          kernel_convolute(blocks, threads)(in.offsetPointer, out.offsetPointer, ix, iy, x, y, z, bs, fx, fy, sx, sy, px, py)

          out

        }
      }

    }



    def convoluteBpImpl(implicit context: CuContext = CuContext.ensureContext): convolute_backprop.Impl3[CuMatrix[V], Convolution[V], Int, CuMatrix[V]] = {

      new convolute_backprop.Impl3[CuMatrix[V], Convolution[V], Int, CuMatrix[V]] {
        def apply(in: CuMatrix[V], l: Convolution[V], bs: Int): CuMatrix[V] = {

          val (ix, iy, x, y, z, fx, fy, sx, sy, px, py) =
            (l.dimIn._1, l.dimIn._2, l.dimOut._1, l.dimOut._2,
              l.dimOut._3, l.field._1, l.field._2, l.stride._1,
              l.stride._2, l.padding._1, l.padding._2)

          val xb = x * bs

          val out = CuMatrix.zeros[V](fx * fy * z, ix * iy * bs)

          val threads = (4, 4, 4)

          val blocks = (
            math.ceil(xb.toDouble / threads._1.toDouble).toInt,
            math.ceil(y.toDouble / threads._2.toDouble).toInt,
            math.ceil(z.toDouble / threads._3.toDouble).toInt
          )

          kernel_convolute_bp(blocks, threads)(in.offsetPointer, out.offsetPointer, ix, iy, x, y, z, bs, fx, fy, sx, sy, px, py)

          out

        }
      }

    }



    def reshapeBatchImpl(implicit context: CuContext = CuContext.ensureContext): reshape_batch.Impl3[CuMatrix[V], (Int, Int, Int), Int, CuMatrix[V]] = {

      new reshape_batch.Impl3[CuMatrix[V], (Int, Int, Int), Int, CuMatrix[V]] {
        def apply(in: CuMatrix[V], dim: (Int, Int, Int), bs: Int): CuMatrix[V] = {
          val (x, y, z) = dim
          val dimOut = (x * y * z, bs)
          val out = CuMatrix.zeros[V](dimOut._2, dimOut._1)

          val threads = (4, 4, 1)

          val blocks = (
            math.ceil(dimOut._1.toDouble / threads._1.toDouble).toInt,
            math.ceil(dimOut._2.toDouble / threads._2.toDouble).toInt,
            1
          )

          kernel_reshape_batch(blocks, threads)(in.offsetPointer, out.offsetPointer, x, y, z, bs)

          out
        }
      }

    }


    def reshapeBatchBpImpl(implicit context: CuContext = CuContext.ensureContext): reshape_batch_backprop.Impl3[CuMatrix[V], (Int, Int, Int), Int, CuMatrix[V]] = {

      new reshape_batch_backprop.Impl3[CuMatrix[V], (Int, Int, Int), Int, CuMatrix[V]] {
        def apply(in: CuMatrix[V], dim: (Int, Int, Int), bs: Int): CuMatrix[V] = {
          val (x, y, z) = dim
          val out = CuMatrix.zeros[V](z, x * y * bs)
          val threads = (4, 4, 1)

          val blocks = (
            math.ceil((x * y * z).toDouble / threads._1.toDouble).toInt,
            math.ceil(bs.toDouble / threads._2.toDouble).toInt,
            1
          )

          kernel_reshape_batch_bp(blocks, threads)(in.offsetPointer, out.offsetPointer, x, y, z, bs)

          out
        }
      }

    }

  }

  /* Misc */
  class MiscKernelBroker[V: ClassTag](typeName: String) {

    private val module: CuModule = {
      debug(s"Loading module: matrix_misc_$typeName.ptx")
      CuModule(getClass.getResourceAsStream(s"/cuda/matrix_misc_$typeName.ptx"))
    }

    private val kernel_subrowmax = module.getKernel4[Pointer, Pointer, Int, Int](s"subrowmax_$typeName")

    def subrowmax(implicit context: CuContext = CuContext.ensureContext): subRowMax.Impl[CuMatrix[V], CuMatrix[V]] = {

      new subRowMax.Impl[CuMatrix[V], CuMatrix[V]] {
        def apply(in: CuMatrix[V]): CuMatrix[V] = {
          val out = CuMatrix.zeros[V](in.rows, in.cols)
          val threads = (4, 1, 1)
          val blocks = (math.ceil(in.rows.toDouble / threads._1.toDouble).toInt, 1, 1)
          kernel_subrowmax(blocks, threads)(in.offsetPointer, out.offsetPointer, in.rows, in.cols)
          out
        }
      }

    }

  }



}

