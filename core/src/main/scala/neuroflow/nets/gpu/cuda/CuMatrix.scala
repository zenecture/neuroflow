package neuroflow.nets.gpu.cuda

import breeze.linalg.operators._
import breeze.linalg._
import breeze.linalg.support.{CanCollapseAxis, CanTranspose, CanSlice2}
import org.bridj.Pointer

import jcuda.jcublas.{cublasOperation, cublasHandle, JCublas2}
import jcuda.runtime.{cudaMemcpyKind, cudaStream_t, JCuda}
import jcuda.driver.CUstream
import jcuda.jcurand.{curandRngType, curandGenerator}
import breeze.math.{Semiring, Ring}
import breeze.numerics._
import breeze.generic.UFunc
import breeze.generic.UFunc.InPlaceImpl2
import breeze.stats.distributions.{Rand, RandBasis}
import scala.reflect._


/**
  * [[https://github.com/dlwh/gust/]] under Apache License 2.0
  * @author dlwh
  * @author bogdanski
  */
class CuMatrix[V](val rows: Int,
                  val cols: Int,
                  val data: Pointer[V],
                  val offset: Int,
                  val majorStride: Int,
                  val isTranspose: Boolean = false) extends NumericOps[CuMatrix[V]] {
  /** Creates a matrix with the specified data array, rows, and columns. Data must be column major */
  def this(rows: Int, cols: Int, data: Pointer[V], offset: Int) = this(rows, cols, data, offset, rows)
  def this(rows: Int, cols: Int, data: Pointer[V]) = this(rows, cols, data, 0, rows)
  /** Creates a matrix with the specified data array, rows, and columns. */
  def this(rows: Int, cols: Int)(implicit ct: ClassTag[V]) = this(rows, cols, allocate[V](rows * cols))

  def size = rows * cols

  /** Calculates the index into the data array for row and column */
  final def linearIndex(row: Int, col: Int): Int = {
    if (isTranspose)
      offset + col + row * majorStride
    else
      offset + row + col * majorStride
  }

  def repr = this

  def majorSize = if (isTranspose) rows else cols

  def activeSize = size

  def footprint = majorSize * majorStride

  def isActive(i: Int) = true
  def allVisitableIndicesActive = true

  def elemSize = data.getIO.getTargetSize
  def offsetPointer = data.toCuPointer.withByteOffset(elemSize * offset)

  def writeFromDense(b: DenseMatrix[V]): Int = {

    require(b.rows == this.rows, "Matrices must have same number of rows")
    require(b.cols == this.cols, "Matrices must have same number of columns")

    if (isTranspose) {
      return this.t.writeFromDense(b.t)
    }

    val _b = if (b.isTranspose) b.copy else b
    val bPtr = cuPointerToArray(_b.data)

    val (width, height) = if (isTranspose) (cols, rows) else (rows, cols)

    assert(majorStride >= width, majorStride + " " + width)
    assert(_b.majorStride >= width)

    JCuda.cudaMemcpy2D(data.toCuPointer.withByteOffset(offset * elemSize),
      majorStride * elemSize,
      bPtr.withByteOffset(offset * elemSize),
      _b.majorStride * elemSize,
      width * elemSize,
      height,
      cudaMemcpyKind.cudaMemcpyHostToDevice
    )

    JCuda.cudaFreeHost(bPtr)

  }

  private def isGapless = (!this.isTranspose && this.majorStride == this.rows) || (this.isTranspose && this.majorStride == this.cols)

  def writeFrom(b: CuMatrix[V])(implicit stream: CUstream = new CUstream(), blas: cublasHandle) = {

    require(b.rows == this.rows, "Matrices must have same number of rows")
    require(b.cols == this.cols, "Matrices must have same number of columns")

    val aPtr = data.toCuPointer.withByteOffset(offset * elemSize)
    val bPtr = b.data.toCuPointer.withByteOffset(offset * elemSize)

    val (width, height) = if (isTranspose) (cols, rows) else (rows, cols)

    if (b.isGapless && this.isGapless && b.isTranspose == this.isTranspose)  {
      JCuda.cudaMemcpyAsync(aPtr, bPtr, size * elemSize, cudaMemcpyKind.cudaMemcpyDeviceToDevice, new cudaStream_t(stream))
    } else if (b.isTranspose == this.isTranspose) {
      JCuda.cudaMemcpy2DAsync(aPtr,
        majorStride * elemSize,
        bPtr,
        b.majorStride * elemSize,
        width * elemSize,
        height,
        cudaMemcpyKind.cudaMemcpyDeviceToDevice,
        new cudaStream_t(stream)
      )
    } else {
      val op = if (elemSize == 4) {
        JCublas2.cublasSgeam _
      } else if (elemSize == 8) {
        JCublas2.cublasDgeam _
      } else {
        throw new UnsupportedOperationException("can't do a copy-transpose with elems that are not of size 4 or 8")
      }
      blas.withStream(stream) {
        op(blas, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_T,
          width, height,
          CuMatrix.hostOne,
          bPtr,
          b.majorStride,
          CuMatrix.hostZero,
          bPtr, b.majorStride, aPtr, majorStride)
      }
    }

  }

  private def canReshapeView = if (isTranspose) majorStride == cols else majorStride == rows

  /** Reshapes this matrix to have the given number of rows and columns
    * If view = true (or View.Require), throws an exception if we cannot return a view. otherwise returns a view.
    * If view == false (or View.Copy) returns a copy
    * If view == View.Prefer (the default), returns a view if possible, otherwise returns a copy.
    *
    * Views are only possible (if (isTranspose) majorStride == cols else majorStride == rows) == true
    *
    * rows * cols must equal size, or cols < 0 && (size / rows * rows == size)
    * @param rows the number of rows
    * @param cols the number of columns, or -1 to auto determine based on size and rows
    */
  def reshape(rows: Int, cols: Int, view: View = View.Prefer): CuMatrix[V] = {

    require(rows * cols == size, "Cannot reshape a (%d,%d) matrix to a (%d,%d) matrix!".format(this.rows, this.cols, rows, cols))

    view match {
      case View.Require =>
        if (!canReshapeView)
          throw new UnsupportedOperationException("Cannot make a view of this matrix.")
        else
          new CuMatrix(rows, cols, data, offset, if (isTranspose) cols else rows, isTranspose)
      case View.Copy =>
        // calling copy directly gives a verify error. TODO: submit bug
        val result = copy
        result.reshape(rows, cols, View.Require)
      case View.Prefer =>
        reshape(rows, cols, canReshapeView)
    }

  }

  /** Forcibly releases the buffer. Note that other slices will be invalidated! */
  def release(): Unit = {
    data.release()
  }

  def toDense: DenseMatrix[V] = {
    val arrayData = Pointer.allocateArray(data.getIO, size)
    val (_r, _c) = if (isTranspose) (cols, rows) else (rows, cols)

    JCublas2.cublasGetMatrix(_r, _c, elemSize.toInt, data.toCuPointer.withByteOffset(elemSize * offset), majorStride, arrayData.toCuPointer, _r)

    new DenseMatrix(rows, cols, arrayData.getArray.asInstanceOf[Array[V]], 0, _r, isTranspose)
  }

  def copy: CuMatrix[V] = ???

  /**
    * Method for slicing that is tuned for Matrices.
    * @return
    */
  def apply[Slice1, Slice2, Result](slice1: Slice1, slice2: Slice2)(implicit canSlice: CanSlice2[CuMatrix[V], Slice1, Slice2, Result]) = {
    canSlice(repr, slice1, slice2)
  }

  // to make life easier when debugging
  override def toString: String = this.toDense.toString + "\nPointer: " + data.toString + "\n"

  /**
    * A Frobenius norm of a matrix
    */
  def norm(implicit handle: cublasHandle, ct: ClassTag[V]): Double = {
    ct.toString() match {    // it doesn't feel like the right way to use the class tag but it works for now...
      case "Float" =>
        val normArr = Array(0.0f)
        JCublas2.cublasSnrm2(handle, size, offsetPointer, 1, jcuda.Pointer.to(normArr))

        normArr(0)

      case "Double" =>
        val normArr = Array(0.0)
        JCublas2.cublasDnrm2(handle, size, offsetPointer, 1, jcuda.Pointer.to(normArr))

        normArr(0)

      case _ => throw new UnsupportedOperationException("Can only calculate norm of Float or Double matrix")
    }
  }

  def isSymmetric(implicit handle: cublasHandle, ct: ClassTag[V]): Boolean =  if (rows != cols) false else ct.toString() match {
    case "Float" =>
      val hostOne = jcuda.Pointer.to(Array(1.0f))
      val hostMinusOne = jcuda.Pointer.to(Array(-1.0f))
      val rv = CuMatrix.create[Float](rows, cols)

      JCublas2.cublasSgeam(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T,
        rows, cols, hostOne,
        offsetPointer, majorStride, hostMinusOne, offsetPointer, majorStride,
        rv.offsetPointer, rv.majorStride)

      rv.norm < 1e-15

    case "Double" =>
      val hostOne = jcuda.Pointer.to(Array(1.0))
      val hostMinusOne = jcuda.Pointer.to(Array(-1.0))
      val rv = CuMatrix.create[Double](rows, cols)

      JCublas2.cublasDgeam(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T,
        rows, cols, hostOne,
        offsetPointer, majorStride, hostMinusOne, offsetPointer, majorStride,
        rv.offsetPointer, rv.majorStride)

      rv.norm < 1e-15

    case _ => throw new UnsupportedOperationException("isSymmetric accepts only Floats and Double")
  }

}

object CuMatrix extends LowPriorityNativeMatrix with CuMatrixOps with CuMatrixSliceOps with CuMatrixFuns {


  /**
    * The standard way to create an empty matrix, size is rows * cols
    */
  def zeros[V](rows: Int, cols: Int)(implicit ct: ClassTag[V]): CuMatrix[V] = {
    val mat = new CuMatrix[V](rows, cols)
    JCuda.cudaMemset(mat.data.toCuPointer, 0, mat.size * mat.elemSize)

    mat
  }

  /**
    * The standard way to create an empty matrix, size is rows * cols
    */
  def ones[V](rows: Int, cols: Int)(implicit ct: ClassTag[V], semiring: Semiring[V], canSet: OpSet.InPlaceImpl2[CuMatrix[V], V]): CuMatrix[V] = {
    val mat = new CuMatrix[V](rows, cols)
    mat := semiring.one

    mat
  }

  def eye[V](size: Int)(implicit ct: ClassTag[V], blas: cublasHandle, semiring: Semiring[V]): CuMatrix[V] = {
    val mat = CuMatrix.zeros[V](size, size)
    val diag = DenseVector.ones[V](size)
    val p = Pointer.pointerToArray(diag.data).as(ct.runtimeClass)

    JCublas2.cublasSetVector(size, mat.elemSize.toInt, p.toCuPointer, 1, mat.offsetPointer, mat.majorStride+1)
    mat
  }

  /**
    * Doesn't zero the matrix.
    */
  def create[V](rows: Int, cols: Int)(implicit ct: ClassTag[V]): CuMatrix[V] = {
    val mat = new CuMatrix[V](rows, cols)
    JCuda.cudaMemset(mat.data.toCuPointer, 0, mat.size * mat.elemSize)

    mat
  }

  def rand[V](rows: Int, cols: Int)(implicit rand: RandBasis = Rand, ct: ClassTag[V]): CuMatrix[V] = {
    import jcuda.jcurand.JCurand._
    val mat = new CuMatrix[V](rows, cols)
    val generator = new curandGenerator()
    curandCreateGenerator(generator, curandRngType.CURAND_RNG_PSEUDO_DEFAULT)
    curandSetPseudoRandomGeneratorSeed(generator, rand.randInt.draw())

    curandGenerateUniform(generator, mat.data.toCuPointer, rows * cols)
    curandDestroyGenerator(generator)

    mat
  }

  def fromDense[V <: AnyVal](mat: DenseMatrix[V])(implicit ct: ClassTag[V]): CuMatrix[V] = {
    val g = new CuMatrix[V](mat.rows, mat.cols)
    g := mat
    g
  }

  implicit def canTranspose[V]: CanTranspose[CuMatrix[V], CuMatrix[V]] = {
    new CanTranspose[CuMatrix[V], CuMatrix[V]] {
      def apply(from: CuMatrix[V]) = {
        new CuMatrix(data = from.data, offset = from.offset, cols = from.rows, rows = from.cols, majorStride = from.majorStride, isTranspose = !from.isTranspose)
      }
    }
  }

  implicit def setMDM[V](implicit stream: CUstream = new CUstream()): OpSet.InPlaceImpl2[CuMatrix[V], DenseMatrix[V]] = new OpSet.InPlaceImpl2[CuMatrix[V], DenseMatrix[V]] {
    def apply(v: CuMatrix[V], v2: DenseMatrix[V]): Unit = {
      v.writeFromDense(v2)
    }
  }

  protected val hostOnePtr = Pointer.pointerToFloat(1)
  protected val hostNegativeOnePtr = Pointer.pointerToFloat(-1)

  protected val hostOne = hostOnePtr.toCuPointer
  protected val hostNegativeOne = hostNegativeOnePtr.toCuPointer

  protected val hostZeroPtr = Pointer.pointerToFloat(0)
  protected val hostZero = hostZeroPtr.toCuPointer

  // Things go wrong if cublasDgemm is passed a float value (don't know why)
  protected val hostOnePtrDouble = Pointer.pointerToDouble(1.0)
  protected val hostOneDouble = hostOnePtrDouble.toCuPointer

  protected val hostZeroPtrDouble = Pointer.pointerToDouble(0.0)
  protected val hostZeroDouble = hostOnePtrDouble.toCuPointer

}

trait LowPriorityNativeMatrix1 {
}

trait LowPriorityNativeMatrix extends LowPriorityNativeMatrix1 { this: CuMatrix.type =>

  class SetCuMCuMVOp[V](implicit handle: cublasHandle) extends OpSet.InPlaceImpl2[CuMatrix[V], CuMatrix[V]] {
    def apply(a: CuMatrix[V], b: CuMatrix[V]) {
      a.writeFrom(b.asInstanceOf[CuMatrix[V]])
    }
  }

  implicit def setCuMCuMOp[V](implicit handle: cublasHandle):OpSet.InPlaceImpl2[CuMatrix[V], CuMatrix[V]] = new SetCuMCuMVOp[V]()

  def transposeOp(a: CuMatrix[_]): Int = {
    if (a.isTranspose) cublasOperation.CUBLAS_OP_T else cublasOperation.CUBLAS_OP_N
  }


}

trait CuMatrixOps extends CuMatrixFuns { this: CuMatrix.type =>
  implicit def CuMatrixDMulCuMatrixD(implicit blas: cublasHandle): OpMulMatrix.Impl2[CuMatrix[Double], CuMatrix[Double], CuMatrix[Double]] = new OpMulMatrix.Impl2[CuMatrix[Double], CuMatrix[Double], CuMatrix[Double]] {
    def apply(_a : CuMatrix[Double], _b : CuMatrix[Double]): CuMatrix[Double] = {
      require(_a.cols == _b.rows, s"Dimension mismatch: ${(_a.rows, _a.cols)} ${(_b.rows, _b.cols)}")
      val rv = CuMatrix.zeros[Double](_a.rows, _b.cols)

      if (_a.rows == 0 || _b.rows == 0 || _a.cols == 0 || _b.cols == 0) return rv

      // if we have a weird stride...
      val a:CuMatrix[Double] = if (_a.majorStride < math.max(if (_a.isTranspose) _a.cols else _a.rows, 1)) _a.copy else _a
      val b:CuMatrix[Double] = if (_b.majorStride < math.max(if (_b.isTranspose) _b.cols else _b.rows, 1)) _b.copy else _b

      JCublas2.cublasDgemm(blas, transposeOp(a), transposeOp(b),
        rv.rows, rv.cols, a.cols,
        hostOneDouble, a.data.toCuPointer.withByteOffset(a.offset * a.elemSize), a.majorStride,
        b.data.toCuPointer.withByteOffset(b.offset * b.elemSize), b.majorStride,
        hostZeroDouble, rv.data.toCuPointer, rv.rows)
      rv
    }
  }

  implicit def CuMatrixFMulCuMatrixF(implicit blas: cublasHandle): OpMulMatrix.Impl2[CuMatrix[Float], CuMatrix[Float], CuMatrix[Float]] = new OpMulMatrix.Impl2[CuMatrix[Float], CuMatrix[Float], CuMatrix[Float]] {
    def apply(_a : CuMatrix[Float], _b : CuMatrix[Float]): CuMatrix[Float] = {
      require(_a.cols == _b.rows, s"Dimension mismatch: ${(_a.rows, _a.cols)} ${(_b.rows, _b.cols)}")
      val rv = CuMatrix.zeros[Float](_a.rows, _b.cols)

      if (_a.rows == 0 || _b.rows == 0 || _a.cols == 0 || _b.cols == 0) return rv

      // if we have a weird stride...
      val a:CuMatrix[Float] = if (_a.majorStride < math.max(if (_a.isTranspose) _a.cols else _a.rows, 1)) _a.copy else _a
      val b:CuMatrix[Float] = if (_b.majorStride < math.max(if (_b.isTranspose) _b.cols else _b.rows, 1)) _b.copy else _b

      JCublas2.cublasSgemm(blas, transposeOp(a), transposeOp(b),
        rv.rows, rv.cols, a.cols,
        hostOne, a.data.toCuPointer.withByteOffset(a.offset * a.elemSize), a.majorStride,
        b.data.toCuPointer.withByteOffset(b.offset * b.elemSize), b.majorStride,
        hostZero, rv.data.toCuPointer, rv.rows)
      rv
    }
  }

  implicit def CuMatrixFMulScalarF(implicit blas: cublasHandle): OpMulMatrix.Impl2[CuMatrix[Float], Float, CuMatrix[Float]] =
    new OpMulMatrix.Impl2[CuMatrix[Float], Float, CuMatrix[Float]] {
      def apply(_a: CuMatrix[Float], k: Float) = {
        val rv = CuMatrix.create[Float](_a.rows, _a.cols)

        rv := _a
        val kArr = Array(k)
        JCublas2.cublasSscal(blas, rv.size, jcuda.Pointer.to(kArr), rv.offsetPointer, 1)
        rv
      }
    }

  implicit def CuMatrixDMulScalarD(implicit blas: cublasHandle): OpMulMatrix.Impl2[CuMatrix[Double], Double, CuMatrix[Double]] =
    new OpMulMatrix.Impl2[CuMatrix[Double], Double, CuMatrix[Double]] {
      def apply(_a: CuMatrix[Double], k: Double) = {
        val rv = CuMatrix.create[Double](_a.rows, _a.cols)

        rv := _a
        val kArr = Array(k)
        JCublas2.cublasDscal(blas, rv.size, jcuda.Pointer.to(kArr), rv.offsetPointer, 1)
        rv
      }
    }

  /* trace of a matrix */
  implicit def canTrace[V](implicit blas: cublasHandle, ct: ClassTag[V]): trace.Impl[CuMatrix[V], V] =
    new trace.Impl[CuMatrix[V], V] {
      def apply(_a: CuMatrix[V]) = {
        val diagSize = if (_a.rows < _a.cols) _a.rows else _a.cols
        val tracePtr = Pointer.allocateArray(_a.data.getIO, 1)

        val cublasOp = if (_a.elemSize == 4) JCublas2.cublasSasum _
        else if (_a.elemSize == 8) JCublas2.cublasDasum _
        else throw new UnsupportedOperationException("Can only compute trace of a matrix with elem sizes 4 or 8")

        cublasOp(blas, diagSize, _a.offsetPointer, _a.majorStride+1, tracePtr.toCuPointer)
        tracePtr(0)
      }
    }

  implicit def normImplFloat(implicit handle: cublasHandle): norm.Impl[CuMatrix[Float], Double] =
    new norm.Impl[CuMatrix[Float], Double] {
      def apply(_a: CuMatrix[Float]) = {
        _a.norm
      }
    }

  implicit def normImplDouble(implicit handle: cublasHandle): norm.Impl[CuMatrix[Double], Double] =
    new norm.Impl[CuMatrix[Double], Double] {
      def apply(_a: CuMatrix[Double]) = {
        _a.norm
      }
    }

  implicit def CuMatrixFAddCuMatrixF(implicit blas: cublasHandle): OpAdd.Impl2[CuMatrix[Float], CuMatrix[Float], CuMatrix[Float]] = new OpAdd.Impl2[CuMatrix[Float], CuMatrix[Float], CuMatrix[Float]] {
    def apply(a : CuMatrix[Float], b : CuMatrix[Float]): CuMatrix[Float] = {
      if (a.majorStride < math.max(if (a.isTranspose) a.cols else a.rows, 1)
        || b.majorStride < math.max(if (b.isTranspose) b.cols else b.rows, 1))  {
        addImpl[Float].apply(a, b)
      } else {
        require(a.rows == b.rows, s"Row dimension mismatch for addition: ${(a.rows, a.cols)} ${(b.rows, b.cols)}")
        require(a.cols == b.cols, s"Column dimension mismatch: ${(a.rows, a.cols)} ${(b.rows, b.cols)}")
        val rv = CuMatrix.zeros[Float](a.rows, b.cols)

        if (a.rows == 0 || b.rows == 0 || a.cols == 0 || b.cols == 0) return rv

        // if we have a weird stride (mostly stride 0), switch to custom implementation
        JCublas2.cublasSgeam(blas, transposeOp(a), transposeOp(b),
          rv.rows, rv.cols,
          hostOne, a.data.toCuPointer.withByteOffset(a.offset * a.elemSize), a.majorStride,
          hostOne,
          b.data.toCuPointer.withByteOffset(b.offset * b.elemSize), b.majorStride,
          rv.data.toCuPointer, rv.rows)
        rv
      }
    }
  }

  implicit def CuMatrixFSubCuMatrixF(implicit blas: cublasHandle): OpSub.Impl2[CuMatrix[Float], CuMatrix[Float], CuMatrix[Float]] = new OpSub.Impl2[CuMatrix[Float], CuMatrix[Float], CuMatrix[Float]] {
    def apply(a : CuMatrix[Float], b : CuMatrix[Float]): CuMatrix[Float] = {
      if (a.majorStride < math.max(if (a.isTranspose) a.cols else a.rows, 1)
        || b.majorStride < math.max(if (b.isTranspose) b.cols else b.rows, 1))  {
        subImpl[Float].apply(a, b)
      } else {
        require(a.rows == b.rows, s"Row dimension mismatch for addition: ${(a.rows, a.cols)} ${(b.rows, b.cols)}")
        require(a.cols == b.cols, s"Column dimension mismatch: ${(a.rows, a.cols)} ${(b.rows, b.cols)}")
        val rv = CuMatrix.zeros[Float](a.rows, b.cols)
        JCublas2.cublasSgeam(blas, transposeOp(a), transposeOp(b),
          rv.rows, rv.cols,
          hostOne, a.data.toCuPointer.withByteOffset(a.offset * a.elemSize), a.majorStride,
          hostNegativeOne,
          b.data.toCuPointer.withByteOffset(b.offset * b.elemSize), b.majorStride,
          rv.data.toCuPointer, rv.rows)
        rv
      }
    }
  }

  implicit def CuMatrixFAddCuMatrixFInPlace(implicit blas: cublasHandle): OpAdd.InPlaceImpl2[CuMatrix[Float], CuMatrix[Float]] = new OpAdd.InPlaceImpl2[CuMatrix[Float], CuMatrix[Float]] {
    def apply(_a : CuMatrix[Float], _b : CuMatrix[Float]):Unit = {
      if (_a.isTranspose) apply(_a.t, _b.t)
      else {
        require(_a.rows == _b.rows, s"Row dimension mismatch for addition: ${(_a.rows, _a.cols)} ${(_b.rows, _b.cols)}")
        require(_a.cols == _b.cols, s"Column dimension mismatch: ${(_a.rows, _a.cols)} ${(_b.rows, _b.cols)}")
        require(!_a.isTranspose)
        if (_a.rows == 0 || _b.rows == 0 || _a.cols == 0 || _b.cols == 0) return

        JCublas2.cublasSgeam(blas, cublasOperation.CUBLAS_OP_N, transposeOp(_b),
          _a.rows, _a.cols,
          hostOne, _a.data.toCuPointer.withByteOffset(_a.offset * _a.elemSize), _a.majorStride,
          hostOne,
          _b.data.toCuPointer.withByteOffset(_b.offset * _b.elemSize), _b.majorStride,
          _a.data.toCuPointer, _a.rows)
      }
    }
  }

  implicit def CuMatrixFSubCuMatrixFInPlace(implicit blas: cublasHandle): OpSub.InPlaceImpl2[CuMatrix[Float], CuMatrix[Float]] = new OpSub.InPlaceImpl2[CuMatrix[Float], CuMatrix[Float]] {
    def apply(_a : CuMatrix[Float], _b : CuMatrix[Float]):Unit = {
      if (_a.isTranspose) apply(_a.t, _b.t)
      else {
        require(_a.rows == _b.rows, s"Row dimension mismatch for addition: ${(_a.rows, _a.cols)} ${(_b.rows, _b.cols)}")
        require(_a.cols == _b.cols, s"Column dimension mismatch: ${(_a.rows, _a.cols)} ${(_b.rows, _b.cols)}")
        require(!_a.isTranspose)
        if (_a.rows == 0 || _b.rows == 0 || _a.cols == 0 || _b.cols == 0) return
        JCublas2.cublasSgeam(blas, cublasOperation.CUBLAS_OP_N, transposeOp(_b),
          _a.rows, _a.cols,
          hostOne, _a.data.toCuPointer.withByteOffset(_a.offset * _a.elemSize), _a.majorStride,
          hostNegativeOne,
          _b.data.toCuPointer.withByteOffset(_b.offset * _b.elemSize), _b.majorStride,
          _a.data.toCuPointer, _a.rows)
      }
    }
  }
}



trait CuMatrixSliceOps { this: CuMatrix.type =>
  implicit def canSliceRow[V]: CanSlice2[CuMatrix[V], Int, ::.type, CuMatrix[V]] = {
    new CanSlice2[CuMatrix[V], Int, ::.type, CuMatrix[V]] {
      def apply(m: CuMatrix[V], rowWNegative: Int, ignored: ::.type) = {
        if (rowWNegative < -m.rows || rowWNegative >= m.rows) throw new ArrayIndexOutOfBoundsException("Row must be in bounds for slice!")
        val row = if (rowWNegative<0) rowWNegative+m.rows else rowWNegative
        if (!m.isTranspose)
          new CuMatrix(1, m.cols, m.data, m.offset + row, m.majorStride)
        else
          new CuMatrix(1, m.cols, m.data, m.offset + row * m.cols, 1)
      }
    }
  }

  implicit def canSliceCol[V]: CanSlice2[CuMatrix[V], ::.type, Int, CuMatrix[V]] = {
    new CanSlice2[CuMatrix[V], ::.type, Int, CuMatrix[V]] {
      def apply(m: CuMatrix[V], ignored: ::.type, colWNegative: Int) = {
        if (colWNegative < -m.cols || colWNegative >= m.cols) throw new ArrayIndexOutOfBoundsException("Column must be in bounds for slice!")
        val col = if (colWNegative<0) colWNegative+m.cols else colWNegative
        if (!m.isTranspose)
          new CuMatrix(m.rows, 1, m.data, col * m.rows + m.offset, m.majorStride)
        else
          new CuMatrix(rows=m.rows, 1, m.data, offset = m.offset + col, majorStride = m.majorStride, true)
      }
    }
  }

  implicit def canSliceRows[V]: CanSlice2[CuMatrix[V], Range, ::.type, CuMatrix[V]] = {
    new CanSlice2[CuMatrix[V], Range, ::.type, CuMatrix[V]] {
      def apply(m: CuMatrix[V], rowsWNegative: Range, ignored: ::.type) = {
        val rows = rowsWNegative.getRangeWithoutNegativeIndexes(m.rows)
        if (rows.isEmpty) new CuMatrix(0, m.cols, m.data, 0, 0)
        else if (!m.isTranspose) {
          require(rows.step == 1, "Sorry, we can't support row ranges with step sizes other than 1")
          val first = rows.head
          require(rows.last < m.rows)
          if (rows.last >= m.rows) {
            throw new IndexOutOfBoundsException(s"Row slice of $rows was bigger than matrix rows of ${m.rows}")
          }
          new CuMatrix(rows.length, m.cols, m.data, m.offset + first, m.majorStride)
        } else {
          canSliceCols(m.t, ::, rows).t
        }
      }
    }
  }

  implicit def canSliceCols[V]: CanSlice2[CuMatrix[V], ::.type, Range, CuMatrix[V]] = {
    new CanSlice2[CuMatrix[V], ::.type, Range, CuMatrix[V]] {
      def apply(m: CuMatrix[V], ignored: ::.type, colsWNegative: Range) = {
        val cols = colsWNegative.getRangeWithoutNegativeIndexes(m.cols)
        if (cols.isEmpty) new CuMatrix(m.rows, 0, m.data, 0, 1)
        else if (!m.isTranspose) {
          val first = cols.head
          if (cols.last >= m.cols) {
            throw new IndexOutOfBoundsException(s"Col slice of $cols was bigger than matrix cols of ${m.cols}")
          }
          new CuMatrix(m.rows, cols.length, m.data, m.offset + first * m.majorStride, m.majorStride * cols.step)
        } else {
          canSliceRows(m.t, cols, ::).t
        }
      }
    }
  }

  implicit def canSliceColsAndRows[V]: CanSlice2[CuMatrix[V], Range, Range, CuMatrix[V]] = {
    new CanSlice2[CuMatrix[V], Range, Range, CuMatrix[V]] {
      def apply(m: CuMatrix[V], rowsWNegative: Range, colsWNegative: Range) = {
        val rows = rowsWNegative.getRangeWithoutNegativeIndexes(m.rows)
        val cols = colsWNegative.getRangeWithoutNegativeIndexes(m.cols)
        if (rows.isEmpty || cols.isEmpty) new CuMatrix(rows.size, cols.size, m.data, 0, 1)
        else if (!m.isTranspose) {
          require(rows.step == 1, "Sorry, we can't support row ranges with step sizes other than 1 for non transposed matrices")
          val first = cols.head
          if (rows.last >= m.rows) {
            throw new IndexOutOfBoundsException(s"Row slice of $rows was bigger than matrix rows of ${m.rows}")
          }
          if (cols.last >= m.cols) {
            throw new IndexOutOfBoundsException(s"Col slice of $cols was bigger than matrix cols of ${m.cols}")
          }
          new CuMatrix(rows.length, cols.length, m.data, m.offset + first * m.rows + rows.head, m.majorStride * cols.step)
        } else {
          require(cols.step == 1, "Sorry, we can't support col ranges with step sizes other than 1 for transposed matrices")
          canSliceColsAndRows(m.t, cols, rows).t
        }
      }
    }
  }

  implicit def negFromScale[V](implicit scale: OpMulScalar.Impl2[CuMatrix[V], V, CuMatrix[V]], field: Ring[V]) = {
    new OpNeg.Impl[CuMatrix[V], CuMatrix[V]] {
      override def apply(a : CuMatrix[V]) = {
        scale(a, field.negate(field.one))
      }
    }
  }

  implicit def canSlicePartOfRow[V]: CanSlice2[CuMatrix[V], Int, Range, CuMatrix[V]] = {
    new CanSlice2[CuMatrix[V], Int, Range, CuMatrix[V]] {
      def apply(m: CuMatrix[V], rowWNegative: Int, colsWNegative: Range) = {
        if (rowWNegative < -m.rows || rowWNegative >= m.rows) throw new ArrayIndexOutOfBoundsException("Row must be in bounds for slice!")
        val row = if (rowWNegative<0) rowWNegative + m.rows else rowWNegative
        val cols = colsWNegative.getRangeWithoutNegativeIndexes(m.cols)
        if (row < 0  || row > m.rows) throw new IndexOutOfBoundsException("Slice with out of bounds row! " + row)
        if (cols.isEmpty) new CuMatrix(0, 0, m.data, 0, 1)
        else if (!m.isTranspose) {
          val first = cols.head
          if (cols.last >= m.cols) {
            throw new IndexOutOfBoundsException(s"Col slice of $cols was bigger than matrix cols of ${m.cols}")
          }
          new CuMatrix(1, cols.length, m.data, m.offset + first * m.rows + row, m.majorStride * cols.step)
        } else {
          require(cols.step == 1, "Sorry, we can't support col ranges with step sizes other than 1 for transposed matrices")
          canSlicePartOfCol(m.t, cols, row).t
        }
      }
    }
  }

  implicit def canSlicePartOfCol[V]: CanSlice2[CuMatrix[V], Range, Int, CuMatrix[V]] = {
    new CanSlice2[CuMatrix[V], Range, Int, CuMatrix[V]] {
      def apply(m: CuMatrix[V], rowsWNegative: Range, colWNegative: Int) = {
        val rows = rowsWNegative.getRangeWithoutNegativeIndexes(m.rows)
        if (colWNegative < -m.cols || colWNegative >= m.cols) throw new ArrayIndexOutOfBoundsException("Row must be in bounds for slice!")
        val col = if (colWNegative<0) colWNegative + m.cols else colWNegative
        if (rows.isEmpty) new CuMatrix(0, 0, m.data)
        else if (!m.isTranspose) {
          if (rows.last >= m.rows) {
            throw new IndexOutOfBoundsException(s"Row slice of $rows was bigger than matrix rows of ${m.rows}")
          }
          new CuMatrix(rows.length, 1, m.data, col * m.rows + m.offset + rows.head, m.majorStride)
        } else {
          val m2 = canSlicePartOfRow(m.t, col, rows).t
          m2(::, 0)
        }
      }
    }
  }

}

trait CuMatrixFuns extends CuMatrixKernels { this: CuMatrix.type =>

  implicit val kernelsFloat  = new KernelBroker[Float]("float")
  implicit val kernelsDouble = new KernelBroker[Double]("double")

  implicit def acosImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[acos.type]("acos")
  implicit def asinImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[asin.type]("asin")
  implicit def atanImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[atan.type]("atan")

  implicit def acoshImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[acosh.type]("acosh")
  implicit def asinhImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[asinh.type]("asinh")
  implicit def atanhImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[atanh.type]("atanh")

  implicit def cosImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[cos.type]("cos")
  implicit def sinImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[sin.type]("sin")
  implicit def tanImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[tan.type]("tan")

  implicit def coshImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[cosh.type]("cosh")
  implicit def sinhImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[sinh.type]("sinh")
  implicit def tanhImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[tanh.type]("tanh")

  implicit def cbrtImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[cbrt.type]("cbrt")
  implicit def ceilImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[ceil.type]("ceil")
  //  implicit def cospiImpl[T](implicit broker: CuMapKernels[T]) =  broker.implFor[cospi.type]("cospi")
  implicit def erfcImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[erfc.type]("erfc")
  implicit def erfcinvImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[erfcinv.type]("erfcinv")
  implicit def erfImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[erf.type]("erf")
  implicit def erfinvImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[erfinv.type]("erfinv")
  implicit def expImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[exp.type]("exp")
  implicit def expm1Impl[T](implicit broker: KernelBroker[T]) =  broker.implFor[expm1.type]("expm1")
  implicit def fabsImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[abs.type]("fabs")
  implicit def floorImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[floor.type]("floor")
  implicit def j0Impl[T](implicit broker: KernelBroker[T]) =  broker.implFor[Bessel.i0.type]("j0")
  implicit def j1Impl[T](implicit broker: KernelBroker[T]) =  broker.implFor[Bessel.i1.type]("j1")
  implicit def lgammaImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[lgamma.type]("lgamma")
  implicit def log10Impl[T](implicit broker: KernelBroker[T]) =  broker.implFor[log10.type]("log10")
  implicit def log1pImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[log1p.type]("log1p")
  //  implicit def log2Impl[T](implicit broker: CuMapKernels[T]) =  broker.implFor[log2.type]("log2")
  //  implicit def logbImpl[T](implicit broker: CuMapKernels[T]) =  broker.implFor[logb.type]("logb")
  implicit def logImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[log.type]("log")
  implicit def sqrtImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[sqrt.type]("sqrt")
  implicit def rintImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[rint.type]("rint")
  //  implicit def truncImpl[T](implicit broker: CuMapKernels[T]) =  broker.implFor[trunc.type]("trunc")

  implicit def acosIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[acos.type]("acos")
  implicit def asinIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[asin.type]("asin")
  implicit def atanIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[atan.type]("atan")

  implicit def acoshIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[acosh.type]("acosh")
  implicit def asinhIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[asinh.type]("asinh")
  implicit def atanhIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[atanh.type]("atanh")

  implicit def cosIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[cos.type]("cos")
  implicit def sinIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[sin.type]("sin")
  implicit def tanIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[tan.type]("tan")

  implicit def coshIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[cosh.type]("cosh")
  implicit def sinhIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[sinh.type]("sinh")
  implicit def tanhIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[tanh.type]("tanh")

  implicit def cbrtIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[cbrt.type]("cbrt")
  implicit def ceilIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[ceil.type]("ceil")
  //  implicit def cospiIntoImpl[T](implicit broker: CuMapKernels[T]) =  broker.inPlaceImplFor[cospi.type]("cospi")
  implicit def erfcIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[erfc.type]("erfc")
  implicit def erfcinvIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[erfcinv.type]("erfcinv")
  implicit def erfIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[erf.type]("erf")
  implicit def erfinvIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[erfinv.type]("erfinv")
  implicit def expIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[exp.type]("exp")
  implicit def expm1IntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[expm1.type]("expm1")
  implicit def fabsIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[abs.type]("fabs")
  implicit def floorIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[floor.type]("floor")
  implicit def j0IntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[Bessel.i0.type]("j0")
  implicit def j1IntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[Bessel.i1.type]("j1")
  implicit def lgammaIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[lgamma.type]("lgamma")
  implicit def log10IntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[log10.type]("log10")
  implicit def log1pIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[log1p.type]("log1p")
  //  implicit def log2IntoImpl[T](implicit broker: CuMapKernels[T]) =  broker.inPlaceImplFor[log2.type]("log2")
  //  implicit def logbIntoImpl[T](implicit broker: CuMapKernels[T]) =  broker.inPlaceImplFor[logb.type]("logb")
  implicit def logIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[log.type]("log")
  implicit def sqrtIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[sqrt.type]("sqrt")
  implicit def rintIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[rint.type]("rint")


  implicit def addImpl[T](implicit broker: KernelBroker[T]) =  broker.impl2For[OpAdd.type]("add")
  implicit def subImpl[T](implicit broker: KernelBroker[T]) =  broker.impl2For[OpSub.type]("sub")
  implicit def mulImpl[T](implicit broker: KernelBroker[T]) =  broker.impl2For[OpMulScalar.type]("mul")
  implicit def divImpl[T](implicit broker: KernelBroker[T]) =  broker.impl2For[OpDiv.type]("div")
  implicit def modImpl[T](implicit broker: KernelBroker[T]) =  broker.impl2For[OpMod.type]("mod")
  implicit def maxImpl[T](implicit broker: KernelBroker[T]) =  broker.impl2For[max.type]("max")
  implicit def minImpl[T](implicit broker: KernelBroker[T]) =  broker.impl2For[min.type]("min")
  implicit def powImpl[T](implicit broker: KernelBroker[T]) =  broker.impl2For[OpPow.type]("pow")

  implicit def addIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For[OpAdd.type]("add")
  implicit def subIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For[OpSub.type]("sub")
  implicit def mulIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For[OpMulScalar.type]("mul")
  implicit def divIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For[OpDiv.type]("div")
  implicit def modIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For[OpMod.type]("mod")
  implicit def maxIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For[max.type]("max")
  implicit def minIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For[min.type]("min")
  implicit def powIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For[OpPow.type]("pow")

  implicit def addIntoImpl_S[T](implicit broker: KernelBroker[T]) = broker.inPlaceImpl2For_v_s[OpAdd.type]("add")
  implicit def subIntoImpl_S[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For_v_s[OpSub.type]("sub")
  implicit def mulIntoImpl_S[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For_v_s[OpMulScalar.type]("mul")
  implicit def divIntoImpl_S[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For_v_s[OpDiv.type]("div")
  implicit def modIntoImpl_S[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For_v_s[OpMod.type]("mod")
  implicit def maxIntoImpl_S[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For_v_s[max.type]("max")
  implicit def minIntoImpl_S[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For_v_s[min.type]("min")
  implicit def powIntoImpl_S[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For_v_s[OpPow.type]("pow")
  implicit def setIntoImpl_S[T](implicit broker: KernelBroker[T]): InPlaceImpl2[OpSet.type, CuMatrix[T], T] =  broker.inPlaceImpl2For_v_s[OpSet.type]("set")

  implicit def addImplVS[T](implicit broker: KernelBroker[T]) =  broker.impl2For_v_s[OpAdd.type]("add")
  implicit def subImplVS[T](implicit broker: KernelBroker[T]) =  broker.impl2For_v_s[OpSub.type]("sub")
  implicit def mulImplVS[T](implicit broker: KernelBroker[T]) =  broker.impl2For_v_s[OpMulScalar.type]("mul")
  implicit def mulMatrixImplVS[T](implicit broker: KernelBroker[T]) =  broker.impl2For_v_s[OpMulMatrix.type]("mul")
  implicit def divImplVS[T](implicit broker: KernelBroker[T]) =  broker.impl2For_v_s[OpDiv.type]("div")
  implicit def modImplVS[T](implicit broker: KernelBroker[T]) =  broker.impl2For_v_s[OpMod.type]("mod")
  implicit def powImplVS[T](implicit broker: KernelBroker[T]) =  broker.impl2For_v_s[OpPow.type]("pow")

  implicit def addImplSV[T](implicit broker: KernelBroker[T]) =  broker.impl2For_s_v[OpAdd.type]("add")
  implicit def subImplSV[T](implicit broker: KernelBroker[T]) =  broker.impl2For_s_v[OpSub.type]("sub")
  implicit def mulImplSV[T](implicit broker: KernelBroker[T]) =  broker.impl2For_s_v[OpMulScalar.type]("mul")
  implicit def mulMatrixImplSV[T](implicit broker: KernelBroker[T]) =  broker.impl2For_s_v[OpMulMatrix.type]("mul")
  implicit def divImplSV[T](implicit broker: KernelBroker[T]) =  broker.impl2For_s_v[OpDiv.type]("div")
  implicit def modImplSV[T](implicit broker: KernelBroker[T]) =  broker.impl2For_s_v[OpMod.type]("mod")
  implicit def powImplSV[T](implicit broker: KernelBroker[T]) =  broker.impl2For_s_v[OpPow.type]("pow")

  implicit def sumImpl[T](implicit broker: KernelBroker[T]) =  broker.reducerFor[sum.type]("add")
  implicit def maxReduceImpl[T](implicit broker: KernelBroker[T]) =  broker.reducerFor[max.type]("max")
  implicit def minReduceImpl[T](implicit broker: KernelBroker[T]) =  broker.reducerFor[min.type]("min")

  implicit def sumColImpl[T](implicit broker: KernelBroker[T]) =  broker.colReducerFor[sum.type]("add")
  implicit def maxColImpl[T](implicit broker: KernelBroker[T]) =  broker.colReducerFor[max.type]("max")
  implicit def minColImpl[T](implicit broker: KernelBroker[T]) =  broker.colReducerFor[min.type]("min")

  implicit def sumRowImpl[T](implicit broker: KernelBroker[T]) =  broker.rowReducerFor[sum.type]("add")
  implicit def maxRowImpl[T](implicit broker: KernelBroker[T]) =  broker.rowReducerFor[max.type]("max")
  implicit def minRowImpl[T](implicit broker: KernelBroker[T]) =  broker.rowReducerFor[min.type]("min")


  implicit def handhold0[T]: CanCollapseAxis.HandHold[CuMatrix[T], Axis._0.type, CuMatrix[T]] = null
  implicit def handhold1[T]: CanCollapseAxis.HandHold[CuMatrix[T], Axis._1.type, CuMatrix[T]] = null


  implicit def broadcastLHSColOpFromBinOp[Func, T](implicit op: UFunc.UImpl2[Func, CuMatrix[T], CuMatrix[T], CuMatrix[T]]):UFunc.UImpl2[Func, BroadcastedColumns[CuMatrix[T], CuMatrix[T]], CuMatrix[T], CuMatrix[T]] = {
    new UFunc.UImpl2[Func, BroadcastedColumns[CuMatrix[T], CuMatrix[T]], CuMatrix[T], CuMatrix[T]] {
      override def apply(vb: BroadcastedColumns[CuMatrix[T], CuMatrix[T]], v2: CuMatrix[T]) = {
        val v = vb.underlying
        require(v2.cols == 1)
        require(!v2.isTranspose)
        require(v.rows == v2.rows)

        // trick: if the major stride is 0, then we iterate over the same column over and over again
        op(v, new CuMatrix(v.rows, v.cols, v2.data, v2.offset, 0, v2.isTranspose))
      }
    }
  }

  implicit def broadcastRHSColOpFromBinOp[Func, T](implicit op: UFunc.UImpl2[Func, CuMatrix[T], CuMatrix[T], CuMatrix[T]]):UFunc.UImpl2[Func, CuMatrix[T], BroadcastedColumns[CuMatrix[T], CuMatrix[T]], CuMatrix[T]] = {
    new UFunc.UImpl2[Func, CuMatrix[T], BroadcastedColumns[CuMatrix[T], CuMatrix[T]], CuMatrix[T]] {
      override def apply(v2: CuMatrix[T], vb: BroadcastedColumns[CuMatrix[T], CuMatrix[T]]) = {
        val v = vb.underlying
        require(v2.cols == 1)
        require(!v2.isTranspose)
        require(v.rows == v2.rows)

        // trick: if the major stride is 0, then we iterate over the same column over and over again
        op(new CuMatrix(v.rows, v.cols, v2.data, v2.offset, 0, v2.isTranspose), v)
      }
    }
  }


  implicit object softmaxImplFloat extends softmax.Impl[CuMatrix[Float], Float] {
    override def apply(v: CuMatrix[Float]): Float = {
      val m: Float = max(v)
      val temp = v - m
      exp.inPlace(temp)
      val res = log(sum(temp)) + m
      temp.data.release()
      res
    }
  }

  // softmax(m(*, ::)) ==> softmaxes each row, given a single column
  implicit object softmaxRowsImplFloat extends softmax.Impl[BroadcastedRows[CuMatrix[Float], CuMatrix[Float]], CuMatrix[Float]] {
    override def apply(v: BroadcastedRows[CuMatrix[Float], CuMatrix[Float]]): CuMatrix[Float] = {
      val m = max(v)
      val temp = v.underlying(::, *) - m
      exp.inPlace(temp)
      val temp2 = sum(temp(*, ::))
      log.inPlace(temp2)
      temp2 += m
      temp.data.release()
      temp2
    }
  }

}
