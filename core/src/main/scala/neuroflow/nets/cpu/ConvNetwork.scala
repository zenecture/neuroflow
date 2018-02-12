package neuroflow.nets.cpu

import breeze.linalg._
import breeze.stats._
import neuroflow.core
import neuroflow.core.Network._
import neuroflow.core._
import neuroflow.dsl.{Convolution, Dense, Focus, Layer}

import scala.annotation.tailrec
import scala.collection.Seq
import scala.collection.mutable.ArrayBuffer

/**
  *
  * Convolutional Neural Network running on CPU,
  * uses gradient descent to optimize the loss function.
  *
  * @author bogdanski
  * @since 31.08.17
  *
  */

object ConvNetwork {

  implicit object double extends Constructor[Double, ConvNetworkDouble] {
    def apply(ls: Seq[Layer], loss: LossFunction[Double], settings: Settings[Double])(implicit weightProvider: WeightProvider[Double]): ConvNetworkDouble = {
      ConvNetworkDouble(ls, loss, settings, weightProvider(ls))
    }
  }

  implicit object weights_double extends neuroflow.core.WeightProvider.CNN[Double]

  implicit object single extends Constructor[Float, ConvNetworkSingle] {
    def apply(ls: Seq[Layer], loss: LossFunction[Float], settings: Settings[Float])(implicit weightProvider: WeightProvider[Float]): ConvNetworkSingle = {
      ConvNetworkSingle(ls, loss, settings, weightProvider(ls))
    }
  }

  implicit object weights_single extends neuroflow.core.WeightProvider.CNN[Float]

}

//<editor-fold defaultstate="collapsed" desc="Double Precision Impl">

private[nets] case class ConvNetworkDouble(layers: Seq[Layer], lossFunction: LossFunction[Double], settings: Settings[Double], weights: Weights[Double],
                                           identifier: String = "neuroflow.nets.cpu.ConvNetwork", numericPrecision: String = "Double")
  extends CNN[Double] with WaypointLogic[Double] {

  type Vector   = DenseVector[Double]
  type Matrix   = DenseMatrix[Double]
  type Tensor   = neuroflow.common.Tensor3D[Double]
  type Vectors  = Seq[DenseVector[Double]]
  type Matrices = Seq[DenseMatrix[Double]]
  type Tensors  = Seq[neuroflow.common.Tensor3D[Double]]

  private val _allLayers = layers.map {
    case f: Focus[Double]         => f.inner
    case d: Dense[Double]         => d
    case c: Convolution[Double]   => c
  }.toArray

  private val _focusLayer         = layers.collectFirst { case c: Focus[_] => c }
  private val _lastWlayerIdx      = weights.size - 1
  private val _convLayers         = _allLayers.zipWithIndex.map(_.swap).filter {
    case (_, _: Convolution[_])   => true
    case _                        => false
  }.toMap.mapValues {
    case c: Convolution[Double]   => c
  }

  private val _activators = _allLayers.map(_.activator)

  private val _outputDim = _allLayers.last.neurons
  private val _lastC     = _convLayers.maxBy(_._1)._1
  private val _lastL     = _allLayers.indices.last

  /**
    * Computes output for `x`.
    */
  def apply(x: Tensor): Vector = {
    _focusLayer.map { cl =>
      flow(x.matrix, layers.indexOf(cl), batchSize = 1)
    }.getOrElse {
      val r = flow(x.matrix, _lastWlayerIdx, batchSize = 1)
      lossFunction match {
        case _: SquaredMeanError[_] => r
        case _: Softmax[_]          => SoftmaxImpl(r)
        case _                      => r
      }
    }.toDenseVector
  }


  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Tensors, ys: Vectors): Unit = {
    import settings._
    val batchSize = settings.batchSize.getOrElse(xs.size)
    require(xs.size == ys.size, s"Mismatch between sample sizes. (${xs.size} != ${ys.size})")
    require(xs.size % batchSize == 0, s"Batches are not even. (${xs.size} % $batchSize = ${xs.size % batchSize} != 0)")
    if (settings.verbose) {
      info(s"Training with ${xs.size} samples, batch size = $batchSize, batches = ${math.ceil(xs.size.toDouble / batchSize.toDouble).toInt}.")
      info(s"Breeding batches ...")
    }
    val xsys = xs.map(_.matrix).zip(ys.map(_.asDenseMatrix)).grouped(batchSize).toSeq.map { batch =>
      batch.par.reduce((x, y) => DenseMatrix.horzcat(x._1, y._1) -> DenseMatrix.vertcat(x._2, y._2))
    }
    run(xsys, learningRate(1 -> 1.0), batchSize, precision, batch = 0, batches = xsys.size, iteration = 1, iterations)
  }

  /**
    * The training loop.
    */
  @tailrec private def run(xsys: Seq[(Matrix, Matrix)], stepSize: Double, batchSize: Int, precision: Double,
                           batch: Int, batches: Int, iteration: Int, maxIterations: Int): Unit = {
    val (x, y) = (xsys(batch)._1, xsys(batch)._2)
    val loss =
      if (settings.approximation.isDefined) adaptWeightsApprox(x, y, stepSize, batchSize)
      else adaptWeights(x, y, stepSize, batchSize)
    val lossMean = mean(loss)
    if (settings.verbose) info(f"Iteration $iteration.${batch + 1}, Avg. Loss = $lossMean%.6g, Vector: $loss")
    maybeGraph(lossMean)
    waypoint(iteration)
    if (lossMean > precision && iteration < maxIterations) {
      run(xsys, settings.learningRate(iteration + 1 -> stepSize), batchSize,
        precision, (batch + 1) % batches, batches, iteration + 1, maxIterations)
    } else {
      info(f"Took $iteration of $maxIterations iterations.")
    }
  }

  private def flow(in: Matrix, target: Int, batchSize: Int): Matrix = {

    val _fa = ArrayBuffer.empty[Matrix]

    @tailrec def conv(_in: Matrix, i: Int): Unit = {
      val l = _convLayers(i)
      val p = weights(i) * convolute(_in, l, batchSize)
      val a = p.map(_activators(i))
      _fa += { if (i == _lastC) reshape_batch(a, l.dimOut, batchSize) else a }
      if (i < _lastC) conv(a, i + 1)
    }

    @tailrec def fully(_in: Matrix, i: Int): Unit = {
      val l = _allLayers(i)
      val p = _in * weights(i)
      val a = p.map(_activators(i))
      _fa += a
      if (i < _lastL) fully(a, i + 1)
    }

    conv(in, 0)
    fully(_fa(_lastC), _lastC + 1)

    _fa(target)

  }

  private def convolute(in: Matrix, l: Convolution[_], batchSize: Int): Matrix = {

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

  private def convolute_bp(in: Matrix, l: Convolution[_], batchSize: Int): Matrix = {

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

    val out = DenseMatrix.zeros[Double](FX * FY * Z, IX * IY * batchSize)

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

  private def reshape_batch(in: Matrix, dim: (Int, Int, Int), batchSize: Int): Matrix = {

    val X = dim._1
    val Y = dim._2
    val Z = dim._3

    val out = DenseMatrix.zeros[Double](batchSize, X * Y * Z)

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

  private def reshape_batch_bp(in: Matrix, dim: (Int, Int, Int), batchSize: Int): Matrix = {

    val X = dim._1
    val Y = dim._2
    val Z = dim._3

    val out = DenseMatrix.zeros[Double](Z, X * Y * batchSize)

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

  /**
    * Computes gradient for weights with respect to given batch,
    * adapts their value using gradient descent and returns the loss matrix.
    */
  private def adaptWeights(x: Matrix, y: Matrix, stepSize: Double, batchSize: Int): Matrix = {

    import settings.updateRule

    val loss = DenseMatrix.zeros[Double](batchSize, _outputDim)

    val fa  = collection.mutable.Map.empty[Int, Matrix]
    val fb  = collection.mutable.Map.empty[Int, Matrix]
    val fc  = collection.mutable.Map.empty[Int, Matrix]
    val dws = collection.mutable.Map.empty[Int, Matrix]
    val ds  = collection.mutable.Map.empty[Int, Matrix]

    @tailrec def conv(_in: Matrix, i: Int): Unit = {
      val l = _convLayers(i)
      val c = convolute(_in, l, batchSize)
      val p = weights(i) * c
      val a = p.map(_activators(i))
      val b = p.map(_activators(i).derivative)
      fa += i -> { if (i == _lastC) reshape_batch(a, l.dimOut, batchSize) else a }
      fb += i -> b
      fc += i -> c
      if (i < _lastC) conv(a, i + 1)
    }

    @tailrec def fully(_in: Matrix, i: Int): Unit = {
      val l = _allLayers(i)
      val p = _in * weights(i)
      val a = p.map(_activators(i))
      val b = p.map(_activators(i).derivative)
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
        loss += err
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
        val d2 = reshape_batch_bp(d1, l.dimOut, batchSize)
        val d = d2 *:* fb(i)
        val dw = d * fc(i).t
        dws += i -> dw
        ds += i -> d
        if (i > 0) derive(i - 1)
      } else {
        val l = _convLayers(i + 1)
        val ww = reshape_batch(weights(i + 1), (l.field._1, l.field._2, l.filters), l.dimIn._3)
        val dc = convolute_bp(ds(i + 1), l, batchSize)
        val d = ww * dc *:* fb(i)
        val dw = d * fc(i).t
        dws += i -> dw
        ds += i -> d
        if (i > 0) derive(i - 1)
      }
    }

    conv(x, 0)
    fully(fa(_lastC), _lastC + 1)
    derive(_lastWlayerIdx)

    (0 to _lastWlayerIdx).foreach(i => updateRule(weights(i), dws(i), stepSize, i))

    val lossReduced = (loss.t * DenseMatrix.ones[Double](loss.rows, 1)).t
    lossReduced

  }

  /** For debugging, approximates the gradients using `settings.approximation`. */
  private def adaptWeightsApprox(xs: Matrix, ys: Matrix, stepSize: Double, batchSize: Int): Matrix = {

    require(settings.updateRule.isInstanceOf[Debuggable[Double]])
    val _rule: Debuggable[Double] = settings.updateRule.asInstanceOf[Debuggable[Double]]

    def lossFunc(): Matrix = {
      val loss = lossFunction(ys, flow(xs, _lastWlayerIdx, batchSize))._1
      val reduced = (loss.t * DenseMatrix.ones[Double](loss.rows, 1)).t
      reduced
    }

    val out = lossFunc()

    def approximateGradient(weightLayer: Int, weight: (Int, Int)): Double = {
      sum(settings.approximation.get.apply(weights, lossFunc, () => (), weightLayer, weight))
    }

    val updates = collection.mutable.HashMap.empty[(Int, (Int, Int)), Double]
    val grads   = collection.mutable.HashMap.empty[(Int, (Int, Int)), Double]
    val debug   = collection.mutable.HashMap.empty[Int, Matrix]

    weights.zipWithIndex.foreach {
      case (l, idx) =>
        debug += idx -> l.copy
        l.foreachPair { (k, v) =>
          val grad = approximateGradient(idx, k)
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

//<editor-fold defaultstate="collapsed" desc="Single Precision Impl">

private[nets] case class ConvNetworkSingle(layers: Seq[Layer], lossFunction: LossFunction[Float], settings: Settings[Float], weights: Weights[Float],
                                           identifier: String = "neuroflow.nets.cpu.ConvNetwork", numericPrecision: String = "Single")
  extends CNN[Float] with WaypointLogic[Float] {

  type Vector   = DenseVector[Float]
  type Matrix   = DenseMatrix[Float]
  type Tensor   = neuroflow.common.Tensor3D[Float]
  type Vectors  = Seq[DenseVector[Float]]
  type Matrices = Seq[DenseMatrix[Float]]
  type Tensors  = Seq[neuroflow.common.Tensor3D[Float]]

  private val _allLayers = layers.map {
    case f: Focus[Double]         => f.inner
    case d: Dense[Double]         => d
    case c: Convolution[Double]   => c
  }.toArray

  private val _focusLayer         = layers.collectFirst { case c: Focus[_] => c }
  private val _lastWlayerIdx      = weights.size - 1
  private val _convLayers         = _allLayers.zipWithIndex.map(_.swap).filter {
    case (_, _: Convolution[_])   => true
    case _                        => false
  }.toMap.mapValues {
    case c: Convolution[Double]   => c
  }

  private val _activators = _allLayers.map(_.activator.map[Float](_.toDouble, _.toFloat))

  private val _outputDim = _allLayers.last.neurons
  private val _lastC     = _convLayers.maxBy(_._1)._1
  private val _lastL     = _allLayers.indices.last

  /**
    * Computes output for `x`.
    */
  def apply(x: Tensor): Vector = {
    _focusLayer.map { cl =>
      flow(x.matrix, layers.indexOf(cl), batchSize = 1)
    }.getOrElse {
      val r = flow(x.matrix, _lastWlayerIdx, batchSize = 1)
      lossFunction match {
        case _: SquaredMeanError[_] => r
        case _: Softmax[_]          => SoftmaxImpl(r)
        case _                      => r
      }
    }.toDenseVector
  }


  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Tensors, ys: Vectors): Unit = {
    import settings._
    val batchSize = settings.batchSize.getOrElse(xs.size)
    require(xs.size == ys.size, s"Mismatch between sample sizes. (${xs.size} != ${ys.size})")
    require(xs.size % batchSize == 0, s"Batches are not even. (${xs.size} % $batchSize = ${xs.size % batchSize} != 0)")
    if (settings.verbose) {
      info(s"Training with ${xs.size} samples, batch size = $batchSize, batches = ${math.ceil(xs.size.toDouble / batchSize.toDouble).toInt}.")
      info(s"Breeding batches ...")
    }
    val xsys = xs.map(_.matrix).zip(ys.map(_.asDenseMatrix)).grouped(batchSize).toSeq.map { batch =>
      batch.par.reduce((x, y) => DenseMatrix.horzcat(x._1, y._1) -> DenseMatrix.vertcat(x._2, y._2))
    }
    run(xsys, learningRate(1 -> 1.0).toFloat, batchSize, precision, batch = 0, batches = xsys.size, iteration = 1, iterations)
  }

  /**
    * The training loop.
    */
  @tailrec private def run(xsys: Seq[(Matrix, Matrix)], stepSize: Float, batchSize: Int, precision: Double,
                           batch: Int, batches: Int, iteration: Int, maxIterations: Int): Unit = {
    val (x, y) = (xsys(batch)._1, xsys(batch)._2)
    val loss =
      if (settings.approximation.isDefined) adaptWeightsApprox(x, y, stepSize, batchSize)
      else adaptWeights(x, y, stepSize, batchSize)
    val lossMean = mean(loss)
    if (settings.verbose) info(f"Iteration $iteration.${batch + 1}, Avg. Loss = $lossMean%.6g, Vector: $loss")
    maybeGraph(lossMean)
    waypoint(iteration)
    if (lossMean > precision && iteration < maxIterations) {
      run(xsys, settings.learningRate(iteration + 1 -> stepSize).toFloat, batchSize,
        precision, (batch + 1) % batches, batches, iteration + 1, maxIterations)
    } else {
      info(f"Took $iteration of $maxIterations iterations.")
    }
  }

  private def flow(in: Matrix, target: Int, batchSize: Int): Matrix = {

    val _fa = ArrayBuffer.empty[Matrix]

    @tailrec def conv(_in: Matrix, i: Int): Unit = {
      val l = _convLayers(i)
      val p = weights(i) * convolute(_in, l, batchSize)
      val a = p.map(_activators(i))
      _fa += { if (i == _lastC) reshape_batch(a, l.dimOut, batchSize) else a }
      if (i < _lastC) conv(a, i + 1)
    }

    @tailrec def fully(_in: Matrix, i: Int): Unit = {
      val l = _allLayers(i)
      val p = _in * weights(i)
      val a = p.map(_activators(i))
      _fa += a
      if (i < _lastL) fully(a, i + 1)
    }

    conv(in, 0)
    fully(_fa(_lastC), _lastC + 1)

    _fa(target)

  }

  private def convolute(in: Matrix, l: Convolution[_], batchSize: Int): Matrix = {

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

    val out = DenseMatrix.zeros[Float](FX * FY * Z, XB * Y)

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

  private def convolute_bp(in: Matrix, l: Convolution[_], batchSize: Int): Matrix = {

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

    val out = DenseMatrix.zeros[Float](FX * FY * Z, IX * IY * batchSize)

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

  private def reshape_batch(in: Matrix, dim: (Int, Int, Int), batchSize: Int): Matrix = {

    val X = dim._1
    val Y = dim._2
    val Z = dim._3

    val out = DenseMatrix.zeros[Float](batchSize, X * Y * Z)

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

  private def reshape_batch_bp(in: Matrix, dim: (Int, Int, Int), batchSize: Int): Matrix = {

    val X = dim._1
    val Y = dim._2
    val Z = dim._3

    val out = DenseMatrix.zeros[Float](Z, X * Y * batchSize)

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

  /**
    * Computes gradient for weights with respect to given batch,
    * adapts their value using gradient descent and returns the loss matrix.
    */
  private def adaptWeights(x: Matrix, y: Matrix, stepSize: Float, batchSize: Int): Matrix = {

    import settings.updateRule

    val loss = DenseMatrix.zeros[Float](batchSize, _outputDim)

    val fa  = collection.mutable.Map.empty[Int, Matrix]
    val fb  = collection.mutable.Map.empty[Int, Matrix]
    val fc  = collection.mutable.Map.empty[Int, Matrix]
    val dws = collection.mutable.Map.empty[Int, Matrix]
    val ds  = collection.mutable.Map.empty[Int, Matrix]

    @tailrec def conv(_in: Matrix, i: Int): Unit = {
      val l = _convLayers(i)
      val c = convolute(_in, l, batchSize)
      val p = weights(i) * c
      val a = p.map(_activators(i))
      val b = p.map(_activators(i).derivative)
      fa += i -> { if (i == _lastC) reshape_batch(a, l.dimOut, batchSize) else a }
      fb += i -> b
      fc += i -> c
      if (i < _lastC) conv(a, i + 1)
    }

    @tailrec def fully(_in: Matrix, i: Int): Unit = {
      val l = _allLayers(i)
      val p = _in * weights(i)
      val a = p.map(_activators(i))
      val b = p.map(_activators(i).derivative)
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
        loss += err
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
        val d2 = reshape_batch_bp(d1, l.dimOut, batchSize)
        val d = d2 *:* fb(i)
        val dw = d * fc(i).t
        dws += i -> dw
        ds += i -> d
        if (i > 0) derive(i - 1)
      } else {
        val l = _convLayers(i + 1)
        val ww = reshape_batch(weights(i + 1), (l.field._1, l.field._2, l.filters), l.dimIn._3)
        val dc = convolute_bp(ds(i + 1), l, batchSize)
        val d = ww * dc *:* fb(i)
        val dw = d * fc(i).t
        dws += i -> dw
        ds += i -> d
        if (i > 0) derive(i - 1)
      }
    }

    conv(x, 0)
    fully(fa(_lastC), _lastC + 1)
    derive(_lastWlayerIdx)

    (0 to _lastWlayerIdx).foreach(i => updateRule(weights(i), dws(i), stepSize, i))

    val lossReduced = (loss.t * DenseMatrix.ones[Float](loss.rows, 1)).t
    lossReduced

  }

  /** For debugging, approximates the gradients using `settings.approximation`. */
  private def adaptWeightsApprox(xs: Matrix, ys: Matrix, stepSize: Float, batchSize: Int): Matrix = {

    require(settings.updateRule.isInstanceOf[Debuggable[Float]])
    val _rule: Debuggable[Float] = settings.updateRule.asInstanceOf[Debuggable[Float]]

    def lossFunc(): Matrix = {
      val loss = lossFunction(ys, flow(xs, _lastWlayerIdx, batchSize))._1
      val reduced = (loss.t * DenseMatrix.ones[Float](loss.rows, 1)).t
      reduced
    }

    val out = lossFunc()

    def approximateGradient(weightLayer: Int, weight: (Int, Int)): Float = {
      sum(settings.approximation.get.apply(weights, lossFunc, () => (), weightLayer, weight))
    }

    val updates = collection.mutable.HashMap.empty[(Int, (Int, Int)), Float]
    val grads   = collection.mutable.HashMap.empty[(Int, (Int, Int)), Float]
    val debug   = collection.mutable.HashMap.empty[Int, Matrix]

    weights.zipWithIndex.foreach {
      case (l, idx) =>
        debug += idx -> l.copy
        l.foreachPair { (k, v) =>
          val grad = approximateGradient(idx, k)
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