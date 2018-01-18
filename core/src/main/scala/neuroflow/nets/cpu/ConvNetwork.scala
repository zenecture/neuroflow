package neuroflow.nets.cpu

import breeze.linalg.Options.{Dimensions2, Zero}
import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import neuroflow.core.Network._
import neuroflow.core._

import scala.annotation.tailrec
import scala.collection.Seq
import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool
import scala.util.Try

/**
  *
  * Convolutional Neural Network,
  * gradient descent to optimize the loss function.
  *
  * @author bogdanski
  * @since 31.08.17
  *
  */

object ConvNetwork {

  implicit val double: Constructor[Double, ConvNetworkDouble] = new Constructor[Double, ConvNetworkDouble] {
    def apply(ls: Seq[Layer], settings: Settings[Double])(implicit weightProvider: WeightProvider[Double]): ConvNetworkDouble = {
      ConvNetworkDouble(ls, settings, weightProvider(ls))
    }
  }

}

//<editor-fold defaultstate="collapsed" desc="Double Precision Impl">

private[nets] case class ConvNetworkDouble(layers: Seq[Layer], settings: Settings[Double], weights: Weights[Double],
                                           identifier: String = "neuroflow.nets.cpu.ConvNetwork", numericPrecision: String = "Double")
  extends CNN[Double] with WaypointLogic[Double] {

  type Vector   = Network.Vector[Double]
  type Vectors  = Network.Vectors[Double]
  type Matrix   = Network.Matrix[Double]
  type Matrices = Network.Matrices[Double]

  private val _allLayers = layers.map {
    case f: Focus[Double]         => f.inner
    case d: Dense[Double]         => d
    case c: Convolution[Double]   => c
    case o: Output[Double]        => o
  }.toArray

  private val _focusLayer         = layers.collectFirst { case c: Focus[_] => c }
  private val _lastWlayerIdx      = weights.size - 1
  private val _convLayers         = _allLayers.zipWithIndex.map(_.swap).filter {
    case (_, _: Convolution[_])   => true
    case _                        => false
  }.toMap.mapValues {
    case c: Convolution[Double]   => c
  }

  private val _outputDim = _allLayers.last.neurons
  private val _lastC     = _convLayers.maxBy(_._1)._1
  private val _lastL     = _allLayers.indices.last

  private type Indices   = Map[(Int, Int), DenseMatrix[Int]]
  private val _indices   = collection.mutable.Map.empty[Int, Indices]

  /**
    * Computes output for `x`.
    */
  def apply(x: Matrix): Vector = {
    _focusLayer.map { cl =>
      flow(x, layers.indexOf(cl), batchSize = 1)
    }.getOrElse {
      val r = flow(x, _lastWlayerIdx, batchSize = 1)
      settings.lossFunction match {
        case _: SquaredMeanError[_] => r
        case _: Softmax[_]          => SoftmaxImpl(r)
        case _                      => r
      }
    }.toDenseVector
  }

  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Matrices, ys: Vectors): Unit = {
    require(xs.size == ys.size, "Mismatch between sample sizes!")
    import settings._
    val batchSize = settings.batchSize.getOrElse(xs.size)
    if (settings.verbose) info(s"Training with ${xs.size} samples, batch size = $batchSize, batches = ${math.ceil(xs.size.toDouble / batchSize.toDouble).toInt} ...")
    val xsys = xs.zip(ys.map(_.asDenseMatrix)).grouped(batchSize).toSeq.map { batch =>
      batch.reduce((x, y) => DenseMatrix.horzcat(x._1, y._1) -> DenseMatrix.vertcat(x._2, y._2))
    }
    run(xsys, learningRate(1 -> 1.0), xs.size, batchSize, precision, batch = 0, batches = xsys.size, iteration = 1, iterations)
  }

  /**
    * The training loop.
    */
  @tailrec private def run(xsys: Seq[(Matrix, Matrix)], stepSize: Double, sampleSize: Double, batchSize: Int,
                           precision: Double, batch: Int, batches: Int, iteration: Int, maxIterations: Int): Unit = {
    val (x, y) = (xsys(batch)._1, xsys(batch)._2)
    val loss =
      if (settings.approximation.isDefined) adaptWeightsApprox(x, y, stepSize, batchSize)
      else adaptWeights(x, y, stepSize, batchSize)
    val lossMean = mean(loss)
    if (settings.verbose) info(f"Iteration $iteration. Ø Batch Loss: $lossMean%.6g. Loss Vector: $loss")
    maybeGraph(lossMean)
    waypoint(iteration)
    if (lossMean > precision && iteration < maxIterations) {
      run(xsys, settings.learningRate(iteration + 1 -> stepSize),
        sampleSize, batchSize, precision, (batch + 1) % batches, batches, iteration + 1, maxIterations)
    } else {
      info(f"Took $iteration iterations of $maxIterations with Loss = $lossMean%.6g")
    }
  }

  private def flow(in: Matrix, target: Int, batchSize: Int): Matrix = {

    val _fa = ArrayBuffer.empty[Matrix]

    @tailrec def conv(_in: Matrix, i: Int): Unit = {
      val l = _convLayers(i)
      val p = weights(i) * convolute(_in, l, batchSize)
      val a = p.map(l.activator)
      _fa += { if (i == _lastC) reshape_batch(a, l, batchSize) else a }
      if (i < _lastC) conv(a, i + 1)
    }

    @tailrec def fully(_in: Matrix, i: Int): Unit = {
      val l = _allLayers(i)
      val p = _in * weights(i)
      val a = p.map(l.activator)
      _fa += a
      if (i < _lastL) fully(a, i + 1)
    }

    conv(in, 0)
    fully(_fa(_lastC), _lastC + 1)

    _fa(target)

  }

  private def convolute(m: Matrix, l: Convolution[_], batchSize: Int): Matrix = {

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

    val out = DenseMatrix.zeros[Double](FX * FY * Z, XB * Y)

    var (x, y, z) = (0, 0, 0)

    while (x < XB) {
      while (y < Y) {
        while (z < Z) {
          var (fX, fY) = (0, 0)
          while (fX < FX) {
            while (fY < FY) {
              val xs = x % X
              val xb = x / X
              val a = fX + (xs * SX)
              val b = fY + (y * SY)
              if (a >= PX && a < (IX + PX) &&
                  b >= PY && b < (IY + PY)) {
                val c = z * FX * FY
                val aNp = a - PX
                val bNp = b - PY
                val p = m(z, (xb * IX * IY) + aNp * IY + bNp)
                val k = fX * FY + fY
                out.update(c + k, (xb * X * Y) + xs * Y + y, p)
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

  private def convolute_bp(m: Matrix, l: Convolution[_], batchSize: Int): Matrix = {

    ???

  }

  private def reshape_batch(m: Matrix, l: Convolution[_], batchSize: Int): Matrix = {

    val X = l.dimOut._1
    val Y = l.dimOut._2
    val Z = l.dimOut._3

    val out = DenseMatrix.zeros[Double](batchSize, X * Y * Z)

    var (x, y) = (0, 0)

    while (x < X * Y * Z) {
      while (y < batchSize) {
        val a = x % (X * Y)
        val b = x / (X * Y)
        val c = y * (X * Y)
        val p = m(b, c + a)
        out.update(y, x, p)
        y += 1
      }
      y = 0
      x += 1
    }

    out

  }

  private def reshape_batch_bp(m: Matrix, l: Convolution[_], batchSize: Int): Matrix = {

    val X = l.dimOut._1
    val Y = l.dimOut._2
    val Z = l.dimOut._3

    val out = DenseMatrix.zeros[Double](Z, X * Y * batchSize)

    var (x, y) = (0, 0)

    while (x < X * Y * Z) {
      while (y < batchSize) {
        val a = x % (X * Y)
        val b = x / (X * Y)
        val c = y * (X * Y)
        val p = m(y, x)
        out.update(b, c + a, p)
        y += 1
      }
      y = 0
      x += 1
    }

    out

  }

//  private def im2col(ms: Matrices, l: Convolution[_], withIndices: Boolean = false): (Matrix, Indices) = {
//    val out = DenseMatrix.zeros[Double](l.field._1 * l.field._2 * l.dimIn._3, l.dimOut._1 * l.dimOut._2)
//    val idc = if (withIndices) {
//      ms.head.keysIterator.map { k =>
//        k -> DenseMatrix.zeros[Int](l.field._1, l.field._2)
//      }.toMap
//    } else null
//    var (x, y, i) = (0, 0, 0)
//    while (x < l.dimOut._1) {
//      while (y < l.dimOut._2) {
//        var (fX, fY) = (0, 0)
//        while (fX < l.field._1) {
//          while (fY < l.field._2) {
//            var z = 0
//            val (a, b) = (fY + (y * l.stride._2), fX + (x * l.stride._1))
//            if (a >= l.padding._2 && a < (l.dimIn._2 + l.padding._2) && b >= l.padding._1 && b < (l.dimIn._1 + l.padding._1)) {
//              while (z < l.dimIn._3) {
//                val c = z * l.field._1 * l.field._2
//                val (_a, _b) = (a - l.padding._2, b - l.padding._1)
//                val value = ms(z)(_a, _b)
//                val k = fX * l.field._2 + fY
//                out.update(c + k, x * l.dimOut._2 + y, value)
//                if (withIndices) idc(_a, _b).update(fX, fY, i + 1)
//                z += 1
//              }
//            }
//            fY += 1
//          }
//          fY  = 0
//          fX += 1
//        }
//        i += 1
//        y += 1
//      }
//      y  = 0
//      x += 1
//    }
//    (out, idc)
//  }
//
//  private def im2col_backprop(d: Matrix, c: Convolution[_], idx: Indices): Matrix = {
//    val dp = padLeft(d, Dimensions2(d.rows, d.cols + 1), Zero)
//    val dc = DenseMatrix.zeros[Double](c.field._1 * c.field._2 * c.filters, c.dimIn._1 * c.dimIn._2)
//    var z  = 0
//    while (z < c.filters) {
//      val _de = dp(z, ::)
//      var (x, y, q) = (0, 0, 0)
//      while (x < c.dimIn._1) {
//        while (y < c.dimIn._2) {
//          var p = 0
//          idx(y, x).foreachPair { (pp, v) =>
//            val t = (z * c.field._1 * c.field._2 + p, q)
//            dc.update(t, _de(v))
//            p += 1
//          }
//          y += 1
//          q += 1
//        }
//        y = 0
//        x += 1
//      }
//      z += 1
//    }
//    dc
//  }
//
//  private def col2im(matrix: Matrix, dim: (Int, Int, Int)): Matrices = {
//    var i = 0
//    val out = new Array[Matrix](dim._3)
//    while (i < dim._3) {
//      val v = matrix.t(::, i).asDenseMatrix.reshape(dim._2, dim._1)
//      out(i) = v
//      i += 1
//    }
//    out
//  }

  /**
    * Computes gradient for weights with respect to given batch,
    * adapts their value using gradient descent and returns the error matrix.
    */
  private def adaptWeights(x: Matrix, y: Matrix, stepSize: Double, batchSize: Int): Matrix = {

    import settings.lossFunction

    val errSum = DenseMatrix.zeros[Double](batchSize, _outputDim)

    val fa  = collection.mutable.Map.empty[Int, Matrix]
    val fb  = collection.mutable.Map.empty[Int, Matrix]
    val fc  = collection.mutable.Map.empty[Int, Matrix]
    val dws = collection.mutable.Map.empty[Int, Matrix]
    val ds  = collection.mutable.Map.empty[Int, Matrix]

    @tailrec def conv(_in: Matrix, i: Int): Unit = {
      val l = _convLayers(i)
      val c = convolute(_in, l, batchSize)
      val p = weights(i) * c
      val a = p.map(l.activator)
      val b = p.map(l.activator.derivative)
      fa += i -> { if (i == _lastC) reshape_batch(a, l, batchSize) else a }
      fb += i -> b
      fc += i -> c
      if (i < _lastC) conv(a, i + 1)
    }

    @tailrec def fully(_in: Matrix, i: Int): Unit = {
      val l = _allLayers(i)
      val p = _in * weights(i)
      val a = p.map(l.activator)
      val b = p.map(l.activator.derivative)
      fa += i -> a
      fb += i -> b
      if (i < _lastL) fully(a, i + 1)
    }

    @tailrec def derive(i: Int): Unit = {
      if (i == _lastWlayerIdx) {
        val (err, grad) = lossFunction(y, fa(i))
        val d = grad *:* fb(i)
        val dw = fa(i - 1).t * d
        dws += i -> dw
        ds += i -> d
        errSum += err
        derive(i - 1)
      } else if (i < _lastWlayerIdx && i > _lastC) {
        val d = (ds(i + 1) * weights(i + 1).t) *:* fb(i)
        val dw = fa(i - 1).t * d
        dws += i -> dw
        ds += i -> d
        derive(i - 1)
      } else if (i == _lastC) {
        val l = _convLayers(i)
        val d1 = ds(i + 1) * weights(i + 1).t
        val d2 = reshape_batch_bp(d1, l, batchSize)
        val d = d2 *:* fb(i)
        val dw = d * fc(i).t
        dws += i -> dw
        ds += i -> d
        if (i > 0) derive(i - 1)
      } else {
//        val dc = im2col_backprop(ds(i + 1), _convLayers(i + 1), _indices(i + 1))
//        val d = _ww(i) * dc *:* fb(i)
        val d: Matrix = ???
        val dw = d * fc(i).t
        dws += i -> dw
        ds += i -> d
        if (i > 0) derive(i - 1)
      }
    }

    conv(x, 0)
    fully(fa(_lastC), _lastC + 1)
    derive(_lastWlayerIdx)

    var i = 0
    while (i <= _lastWlayerIdx) {
      Try(settings.updateRule(weights(i), dws(i), stepSize, i)) // Untry.
      i += 1
    }

    val errSumReduced = (errSum.t * DenseMatrix.ones[Double](errSum.rows, 1)).t
    errSumReduced

  }

  /** Approximates the gradient based on finite central differences. (For debugging) */
  private def adaptWeightsApprox(x: Matrix, y: Matrix, stepSize: Double, batchSize: Int): Matrix = {

    require(settings.updateRule.isInstanceOf[Debuggable[Double]])
    val _rule: Debuggable[Double] = settings.updateRule.asInstanceOf[Debuggable[Double]]

    def errorFunc(): Matrix = {
      val errSum = settings.lossFunction(y, flow(x, _lastWlayerIdx, batchSize))._1
      val errSumReduced = (errSum.t * DenseMatrix.ones[Double](errSum.rows, 1)).t
      errSumReduced
    }

    val out = errorFunc()

    def approximateErrorFuncDerivative(weightLayer: Int, weight: (Int, Int)): Matrix = {
      val Δ = settings.approximation.get.Δ
      val v = weights(weightLayer)(weight)
      weights(weightLayer).update(weight, v - Δ)
      val a = errorFunc()
      weights(weightLayer).update(weight, v + Δ)
      val b = errorFunc()
      weights(weightLayer).update(weight, v)
      (b - a) / (2 * Δ)
    }

    val updates = collection.mutable.HashMap.empty[(Int, (Int, Int)), Double]
    val grads   = collection.mutable.HashMap.empty[(Int, (Int, Int)), Double]
    val debug   = collection.mutable.HashMap.empty[Int, Matrix]

    weights.zipWithIndex.foreach {
      case (l, idx) =>
        debug += idx -> l.copy
        l.foreachPair { (k, v) =>
          val grad = sum(approximateErrorFuncDerivative(idx, k))
          updates += (idx, k) -> (v - (stepSize * grad))
          grads += (idx, k) -> grad
        }
    }

    updates.foreach {
      case ((wl, k), v) =>
        weights(wl).update(k, v)
    }

    grads.foreach {
      case ((wl, k), v) =>
        debug(wl).update(k, v)
    }

    _rule.lastGradients = debug

    out

  }

}

//</editor-fold>