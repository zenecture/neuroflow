package neuroflow.core

import breeze.generic.UFunc
import breeze.linalg.DenseMatrix
import breeze.storage.Zero
import neuroflow.dsl.Convolution

import scala.reflect.ClassTag

/**
  * Collection of common operators expressed as [[UFunc]].
  * The CPU implementations are found here, the GPU implicits
  * are found in the [[neuroflow.cuda.CuMatrix]] area.
  *
  * @author bogdanski
  * @since 07.03.18
  */


/**
  * Subtracts row maximum from row elements.
  * Example given:
  *   |1 2 1|    |-1 0 -1|
  *   |2 2 2| -> | 0 0  0|
  *   |0 1 0|    |-1 0 -1|
  */
object subRowMax extends UFunc {

  implicit object subRowMaxImplDouble extends subRowMax.Impl[DenseMatrix[Double], DenseMatrix[Double]] {
    def apply(in: DenseMatrix[Double]): DenseMatrix[Double] = {
      val out = in.copy
      var (r, c) = (0, 0)
      while (r < in.rows) {
        var max = in(r, 0)
        while (c < in.cols) {
          val t = in(r, c)
          if (t > max) max = t
          c += 1
        }
        c = 0
        while (c < in.cols) {
          val t = in(r, c)
          out.update(r, c, t - max)
          c += 1
        }
        r += 1
      }
      out
    }
  }

  implicit object subRowMaxImplFloat extends subRowMax.Impl[DenseMatrix[Float], DenseMatrix[Float]] {
    def apply(in: DenseMatrix[Float]): DenseMatrix[Float] = {
      val out = in.copy
      var (r, c) = (0, 0)
      while (r < in.rows) {
        var max = in(r, 0)
        while (c < in.cols) {
          val t = in(r, c)
          if (t > max) max = t
          c += 1
        }
        c = 0
        while (c < in.cols) {
          val t = in(r, c)
          out.update(r, c, t - max)
          c += 1
        }
        r += 1
      }
      out
    }
  }

}



/**
  * Convolutes [[neuroflow.common.Tensor3D]] linearized in `in`, producing a new one.
  */
object convolute extends UFunc {

  implicit object convoluteImplDouble extends convolute.Impl3[DenseMatrix[Double], Convolution[Double], Int, DenseMatrix[Double]] {
    def apply(in: DenseMatrix[Double], l: Convolution[Double], batchSize: Int): DenseMatrix[Double] = gapply(in, l, batchSize)
  }

  implicit object convoluteImplFloat extends convolute.Impl3[DenseMatrix[Float], Convolution[Double], Int, DenseMatrix[Float]] {
    def apply(in: DenseMatrix[Float], l: Convolution[Double], batchSize: Int): DenseMatrix[Float] = gapply(in, l, batchSize)
  }

  private def gapply[V: ClassTag : Zero](in: DenseMatrix[V], l: Convolution[_], batchSize: Int): DenseMatrix[V] = {

    val IX = l.dimIn._1
    val IY = l.dimIn._2

    val X = l.dimOut._1
    val Y = l.dimOut._2
    val Z = l.dimIn._3

    val XB = X * batchSize

    val FX = l.field._1
    val FY = l.field._2
    val SX = l.stride._1
    val SY = l.stride._2
    val PX = l.padding._1
    val PY = l.padding._2

    val out = DenseMatrix.zeros[V](FX * FY * Z, XB * Y)

    var (x, y, z) = (0, 0, 0)

    while (x < XB) {
      while (y < Y) {
        while (z < Z) {
          var (fX, fY) = (0, 0)
          while (fX < FX) {
            while (fY < FY) {
              val xs = x % X
              val xb = x / X
              val a = (xs * SX) + fX
              val b = (y * SY) + fY
              if (a >= PX && a < (PX + IX) &&
                  b >= PY && b < (PY + IY)) {
                val aNp = a - PX
                val bNp = b - PY
                val p = in(z, (xb * IX * IY) + aNp * IY + bNp)
                out.update((z * FX * FY) + fX * FY + fY, (xb * X * Y) + xs * Y + y, p)
              }
              fY += 1
            }
            fY = 0
            fX += 1
          }
          z += 1
        }
        z = 0
        y += 1
      }
      y = 0
      x += 1
    }

    out

  }

}



/**
  * Backprops convolution of a [[neuroflow.common.Tensor3D]] linearized in `in`.
  */
object convolute_backprop extends UFunc {

  implicit object convoluteBpImplDouble extends convolute_backprop.Impl3[DenseMatrix[Double], Convolution[Double], Int, DenseMatrix[Double]] {
    def apply(in: DenseMatrix[Double], l: Convolution[Double], batchSize: Int): DenseMatrix[Double] = gapply(in, l, batchSize)
  }

  implicit object convoluteBpImplFloat extends convolute_backprop.Impl3[DenseMatrix[Float], Convolution[Double], Int, DenseMatrix[Float]] {
    def apply(in: DenseMatrix[Float], l: Convolution[Double], batchSize: Int): DenseMatrix[Float] = gapply(in, l, batchSize)
  }

  private def gapply[V: ClassTag : Zero](in: DenseMatrix[V], l: Convolution[_], batchSize: Int): DenseMatrix[V] = {

    val IX = l.dimIn._1
    val IY = l.dimIn._2

    val X = l.dimOut._1
    val Y = l.dimOut._2
    val Z = l.dimOut._3

    val XB = X * batchSize

    val FX = l.field._1
    val FY = l.field._2
    val SX = l.stride._1
    val SY = l.stride._2
    val PX = l.padding._1
    val PY = l.padding._2

    val out = DenseMatrix.zeros[V](FX * FY * Z, IX * IY * batchSize)

    var (x, y, z) = (0, 0, 0)

    while (x < XB) {
      while (y < Y) {
        while (z < Z) {
          var (fX, fY) = (0, 0)
          while (fX < FX) {
            while (fY < FY) {
              val xs = x % X
              val xb = x / X
              val a = (xs * SX) + fX
              val b = (y * SY) + fY
              if (a >= PX && a < (PX + IX) &&
                  b >= PY && b < (PY + IY)) {
                val aNp = a - PX
                val bNp = b - PY
                val d = in(z, (xb * X * Y) + xs * Y + y)
                out.update((z * FX * FY) + fX * FY + fY, (xb * IX * IY) + aNp * IY + bNp, d)
              }
              fY += 1
            }
            fY = 0
            fX += 1
          }
          z += 1
        }
        z = 0
        y += 1
      }
      y = 0
      x += 1
    }

    out

  }

}



/**
  * Reshapes matrix `in` by transposing the batch.
  * Examples given:
  *   |1 2 3|    |1 1 1|
  *   |1 2 3| -> |2 2 2|
  *   |1 2 3|    |3 3 3|
  *
  *   |1 1 2 2|    |1 1 1 1 1 1|
  *   |1 1 2 2| -> |2 2 2 2 2 2|
  *   |1 1 2 2|
  */
object reshape_batch extends UFunc {

  implicit object reshapeBatchImplDouble extends reshape_batch.Impl3[DenseMatrix[Double], (Int, Int, Int), Int, DenseMatrix[Double]] {
    def apply(in: DenseMatrix[Double], dim: (Int, Int, Int), batchSize: Int): DenseMatrix[Double] = gapply(in, dim, batchSize)
  }

  implicit object convoluteBatchImplFloat extends reshape_batch.Impl3[DenseMatrix[Float], (Int, Int, Int), Int, DenseMatrix[Float]] {
    def apply(in: DenseMatrix[Float], dim: (Int, Int, Int), batchSize: Int): DenseMatrix[Float] = gapply(in, dim, batchSize)
  }

  private def gapply[V: ClassTag : Zero](in: DenseMatrix[V], dim: (Int, Int, Int), batchSize: Int): DenseMatrix[V] = {

    val X = dim._1
    val Y = dim._2
    val Z = dim._3

    val out = DenseMatrix.zeros[V](batchSize, X * Y * Z)

    var (x, y) = (0, 0)

    while (x < X * Y * Z) {
      while (y < batchSize) {
        val a = x % (X * Y)
        val b = x / (X * Y)
        val c = y * (X * Y)
        val p = in(b, c + a)
        out.update(y, x, p)
        y += 1
      }
      y = 0
      x += 1
    }

    out

  }

}


/**
  * Reshapes matrix `in` by transposing the batch.
  * Examples given:
  *   |1 2 3|    |1 1 1|
  *   |1 2 3| <- |2 2 2|
  *   |1 2 3|    |3 3 3|
  *
  *   |1 1 2 2|    |1 1 1 1 1 1|
  *   |1 1 2 2| <- |2 2 2 2 2 2|
  *   |1 1 2 2|
  */
object reshape_batch_backprop extends UFunc {

  implicit object reshapeBatchBpImplDouble extends reshape_batch_backprop.Impl3[DenseMatrix[Double], (Int, Int, Int), Int, DenseMatrix[Double]] {
    def apply(in: DenseMatrix[Double], dim: (Int, Int, Int), batchSize: Int): DenseMatrix[Double] = gapply(in, dim, batchSize)
  }

  implicit object convoluteBatchBpImplFloat extends reshape_batch_backprop.Impl3[DenseMatrix[Float], (Int, Int, Int), Int, DenseMatrix[Float]] {
    def apply(in: DenseMatrix[Float], dim: (Int, Int, Int), batchSize: Int): DenseMatrix[Float] = gapply(in, dim, batchSize)
  }

  private def gapply[V: ClassTag : Zero](in: DenseMatrix[V], dim: (Int, Int, Int), batchSize: Int): DenseMatrix[V] = {

    val X = dim._1
    val Y = dim._2
    val Z = dim._3

    val out = DenseMatrix.zeros[V](Z, X * Y * batchSize)

    var (x, y) = (0, 0)

    while (x < X * Y * Z) {
      while (y < batchSize) {
        val a = x % (X * Y)
        val b = x / (X * Y)
        val c = y * (X * Y)
        val p = in(y, x)
        out.update(b, c + a, p)
        y += 1
      }
      y = 0
      x += 1
    }

    out

  }

}

