package neuroflow.cuda

import breeze.generic.UFunc
import jcuda.Pointer

import scala.reflect.ClassTag
import java.util.concurrent.ConcurrentHashMap

import breeze.linalg.{BroadcastedColumns, BroadcastedRows}
import neuroflow.common.Logs

/**
  * @author dlwh
  **/
trait CuMatrixKernels extends Logs { this: CuMatrix.type =>
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

}

