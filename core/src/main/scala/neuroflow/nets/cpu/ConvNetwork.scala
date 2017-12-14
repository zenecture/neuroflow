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

/**
  *
  * This is a convolutional feed-forward neural network.
  * It uses gradient descent to optimize the specified loss function.
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

  implicit val single: Constructor[Float, ConvNetworkSingle] = new Constructor[Float, ConvNetworkSingle] {
    def apply(ls: Seq[Layer], settings: Settings[Float])(implicit weightProvider: WeightProvider[Float]): ConvNetworkSingle = {
      ConvNetworkSingle(ls, settings, weightProvider(ls))
    }
  }

}

//<editor-fold defaultstate="collapsed" desc="Double Precision Impl">

private[nets] case class ConvNetworkDouble(layers: Seq[Layer], settings: Settings[Double], weights: Weights[Double],
                                           identifier: String = "neuroflow.nets.cpu.ConvNetwork", numericPrecision: String = "Double")
  extends CNN[Double] with KeepBestLogic[Double] with WaypointLogic[Double] {

  type Vector   = Network.Vector[Double]
  type Vectors  = Network.Vectors[Double]
  type Matrix   = Network.Matrix[Double]
  type Matrices = Network.Matrices[Double]

  private val _forkJoinTaskSupport = new ForkJoinTaskSupport(new ForkJoinPool(settings.parallelism.getOrElse(1)))

  private val _allLayers = layers.map {
    case f: Focus[Double]         => f.inner
    case d: Dense[Double]         => d
    case c: Convolution[Double]   => c
    case o: Output[Double]        => o
  }.toArray

  private val _focusLayer         = layers.collect { case c: Focus[_] => c }.headOption
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
  def apply(x: Matrices): Vector = {
    _focusLayer.map { cl =>
      flow(x, layers.indexOf(cl))
    }.getOrElse {
      val r = flow(x, _lastWlayerIdx)
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
  def train(xs: Seq[Matrices], ys: Vectors): Unit = {
    require(xs.size == ys.size, "Mismatch between sample sizes!")
    import settings._
    val batchSize = settings.batchSize.getOrElse(xs.size)
    if (settings.verbose) info(s"Training with ${xs.size} samples, batch size = $batchSize, batches = ${math.ceil(xs.size.toDouble / batchSize.toDouble).toInt} ...")
    val xsys = xs.zip(ys.map(_.asDenseMatrix)).grouped(batchSize).toSeq
    run(xsys, learningRate(1 -> 1.0), xs.size, precision, 1, iterations)
  }

  /**
    * The training loop.
    */
  @tailrec private def run(xsys: Seq[Seq[(Matrices, Matrix)]], stepSize: Double, sampleSize: Double, precision: Double,
                           iteration: Int, maxIterations: Int): Unit = {
    val _em = xsys.map { batch =>
      val (x, y) = (batch.map(_._1), batch.map(_._2))
      val error =
        if (settings.approximation.isDefined)
          adaptWeightsApprox(x, y, stepSize)
        else adaptWeights(x, y, stepSize)
        debug(s"Batch Error: $error")
      error
    }.reduce(_ + _)
    val errorPerS = _em / sampleSize
    val errorMean = mean(errorPerS)
    if (settings.verbose) info(f"Iteration $iteration - Loss $errorMean%.6g - Loss Vector $errorPerS")
    maybeGraph(errorMean)
    keepBest(errorMean)
    waypoint(iteration)
    if (errorMean > precision && iteration < maxIterations) {
      run(xsys, settings.learningRate(iteration + 1 -> stepSize), sampleSize, precision, iteration + 1, maxIterations)
    } else {
      info(f"Took $iteration iterations of $maxIterations with Loss = $errorMean%.6g")
      takeBest()
    }
  }

  private def flow(in: Matrices, target: Int): Matrix = {

    val _fa = ArrayBuffer.empty[Matrix]

    @tailrec def conv(_in: Matrices, i: Int): Unit = {
      val l = _convLayers(i)
      val p = weights(i) * im2col(_in, l)._1
      val a = p.map(l.activator)
      _fa += a
      if (i < _lastC) conv(col2im(a, l.dimOut), i + 1)
    }

    @tailrec def fully(_in: Matrix, i: Int): Unit = {
      val l = _allLayers(i)
      val p = _in * weights(i)
      val a = p.map(l.activator)
      _fa += a
      if (i < _lastL) fully(a, i + 1)
    }

    conv(in, 0)
    fully(_fa(_lastC).reshape(1, _convLayers(_lastC).neurons), _lastC + 1)

    _fa(target)

  }

  private def im2col(ms: Matrices, l: Convolution[_], withIndices: Boolean = false): (Matrix, Indices) = {
    val out = DenseMatrix.zeros[Double](l.field._1 * l.field._2 * l.dimIn._3, l.dimOut._1 * l.dimOut._2)
    val idc = if (withIndices) {
      ms.head.keysIterator.map { k =>
        k -> DenseMatrix.zeros[Int](l.field._1, l.field._2)
      }.toMap
    } else null
    var (x, y, i) = (0, 0, 0)
    while (x < l.dimOut._1) {
      while (y < l.dimOut._2) {
        var (fX, fY) = (0, 0)
        while (fX < l.field._1) {
          while (fY < l.field._2) {
            var z = 0
            val (a, b) = (fY + (y * l.stride._2), fX + (x * l.stride._1))
            if (a >= l.padding._2 && a < (l.dimIn._2 + l.padding._2) && b >= l.padding._1 && b < (l.dimIn._1 + l.padding._1)) {
              while (z < l.dimIn._3) {
                val c = z * l.field._1 * l.field._2
                val (_a, _b) = (a - l.padding._2, b - l.padding._1)
                val value = ms(z)(_a, _b)
                val k = fX * l.field._2 + fY
                out.update(c + k, x * l.dimOut._2 + y, value)
                if (withIndices) idc(_a, _b).update(fX, fY, i + 1)
                z += 1
              }
            }
            fY += 1
          }
          fY  = 0
          fX += 1
        }
        i += 1
        y += 1
      }
      y  = 0
      x += 1
    }
    (out, idc)
  }

  private def im2col_backprop(d: Matrix, c: Convolution[_], idx: Indices): Matrix = {
    val dp = padLeft(d, Dimensions2(d.rows, d.cols + 1), Zero)
    val dc = DenseMatrix.zeros[Double](c.field._1 * c.field._2 * c.filters, c.dimIn._1 * c.dimIn._2)
    var z  = 0
    while (z < c.filters) {
      val _de = dp(z, ::)
      var (x, y, q) = (0, 0, 0)
      while (x < c.dimIn._1) {
        while (y < c.dimIn._2) {
          var p = 0
          idx(y, x).foreachPair { (pp, v) =>
            val t = (z * c.field._1 * c.field._2 + p, q)
            dc.update(t, _de(v))
            p += 1
          }
          y += 1
          q += 1
        }
        y = 0
        x += 1
      }
      z += 1
    }
    dc
  }

  private def col2im(matrix: Matrix, dim: (Int, Int, Int)): Matrices = {
    var i = 0
    val out = new Array[Matrix](dim._3)
    while (i < dim._3) {
      val v = matrix.t(::, i).asDenseMatrix.reshape(dim._2, dim._1)
      out(i) = v
      i += 1
    }
    out
  }

  /**
    * Computes gradient for all weights in parallel,
    * adapts their value using gradient descent and returns the error matrix.
    */
  private def adaptWeights(xs: Seq[Matrices], ys: Seq[Matrix], stepSize: Double): Matrix = {

    import settings.lossFunction

    val xsys = xs.par.zip(ys)
    xsys.tasksupport = _forkJoinTaskSupport

    val _dws = (0 to _lastWlayerIdx).map { i =>
      i -> DenseMatrix.zeros[Double](weights(i).rows, weights(i).cols)
    }.toMap

    val _ww = _convLayers.map {
      case (i, _) if i < _lastC =>
        val l  = _convLayers(i)
        val l2 = _convLayers(i + 1)
        val fieldSq = l2.field._1 * l2.field._2
        val wr = weights(i + 1)
        val ww = DenseMatrix.zeros[Double](l.filters, l2.filters * fieldSq)
        var (filter, depth) = (0, 0)
        while (filter < l2.filters) {
          while (depth < l.filters) {
            val ws = wr(filter, (depth * fieldSq) until ((depth * fieldSq) + fieldSq))
            var i = 0
            while (i < fieldSq) {
              ww.update(depth, filter * fieldSq + i, ws(i))
              i += 1
            }
            depth += 1
          }
          depth = 0
          filter += 1
        }
        i -> ww
      case (i, _) => i -> null
    } - _lastC

    val _errSum  = DenseMatrix.zeros[Double](1, _outputDim)

    xsys.map { xy =>

      val (x, y) = xy
      val fa  = collection.mutable.Map.empty[Int, Matrix]
      val fb  = collection.mutable.Map.empty[Int, Matrix]
      val fc  = collection.mutable.Map.empty[Int, Matrix]
      val dws = collection.mutable.Map.empty[Int, Matrix]
      val ds  = collection.mutable.Map.empty[Int, Matrix]
      val e   = DenseMatrix.zeros[Double](1, _outputDim)

      @tailrec def conv(_in: Matrices, i: Int): Unit = {
        val l = _convLayers(i)
        val seen = _indices.isDefinedAt(i)
        val (c, x) = im2col(_in, l, withIndices = !seen)
        val p = weights(i) * c
        var a = p.map(l.activator)
        var b = p.map(l.activator.derivative)
        if (i == _lastC) {
          a = a.reshape(1, l.neurons)
          b = b.reshape(1, l.neurons)
        }
        fa += i -> a
        fb += i -> b
        fc += i -> c
        if (!seen) _indices += i -> x
        if (i < _lastC) conv(col2im(a, l.dimOut), i + 1)
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
          e += err
          derive(i - 1)
        } else if (i < _lastWlayerIdx && i > _lastC) {
          val d = (ds(i + 1) * weights(i + 1).t) *:* fb(i)
          val dw = fa(i - 1).t * d
          dws += i -> dw
          ds += i -> d
          derive(i - 1)
        } else if (i == _lastC) {
          val l = _convLayers(i)
          val d = ((ds(i + 1) * weights(i + 1).t) *:* fb(i))
            .reshape(l.filters, l.dimOut._1 * l.dimOut._2)
          val dw = d * fc(i).t
          dws += i -> dw
          ds += i -> d
          if (i > 0) derive(i - 1)
        } else {
          val dc = im2col_backprop(ds(i + 1), _convLayers(i + 1), _indices(i + 1))
          val d = _ww(i) * dc *:* fb(i)
          val dw = d * fc(i).t
          dws += i -> dw
          ds += i -> d
          if (i > 0) derive(i - 1)
        }
      }

      conv(x, 0)
      fully(fa(_lastC), _lastC + 1)
      derive(_lastWlayerIdx)
      (dws, e)

    }.seq.foreach { ab =>
      _errSum += ab._2
      var i = 0
      while (i <= _lastWlayerIdx) {
        val m = _dws(i)
        val n = ab._1(i)
        m += n
        i += 1
      }
    }

    var i = 0
    while (i <= _lastWlayerIdx) {
      settings.updateRule(weights(i), _dws(i), stepSize, i)
      i += 1
    }

    _errSum

  }

  /** Approximates the gradient based on finite central differences. (For debugging) */
  private def adaptWeightsApprox(xs: Seq[Matrices], ys: Matrices, stepSize: Double): Matrix = {

    require(settings.updateRule.isInstanceOf[Debuggable[Double]])
    val _rule: Debuggable[Double] = settings.updateRule.asInstanceOf[Debuggable[Double]]

    def errorFunc(): Matrix = {
      val xsys = xs.zip(ys).par
      xsys.tasksupport = _forkJoinTaskSupport
      xsys.map { case (x, y) => settings.lossFunction(y, flow(x, _lastWlayerIdx))._1 }.reduce(_ + _)
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

//<editor-fold defaultstate="collapsed" desc="Single Precision Impl">

private[nets] case class ConvNetworkSingle(layers: Seq[Layer], settings: Settings[Float], weights: Weights[Float],
                                           identifier: String = "neuroflow.nets.cpu.ConvNetwork", numericPrecision: String = "Single")
  extends CNN[Float] with KeepBestLogic[Float] with WaypointLogic[Float] {

  type Vector   = Network.Vector[Float]
  type Vectors  = Network.Vectors[Float]
  type Matrix   = Network.Matrix[Float]
  type Matrices = Network.Matrices[Float]

  private val _forkJoinTaskSupport = new ForkJoinTaskSupport(new ForkJoinPool(settings.parallelism.getOrElse(1)))

  private val _allLayers = layers.map {
    case f: Focus[Double]         => f.inner
    case d: Dense[Double]         => d
    case c: Convolution[Double]   => c
    case o: Output[Double]        => o
  }.toArray

  private val _focusLayer         = layers.collect { case c: Focus[_] => c }.headOption
  private val _lastWlayerIdx      = weights.size - 1
  private val _activators         = _allLayers.map { hd => hd.activator.map[Float](_.toDouble, _.toFloat) }
  private val _convLayers         = _allLayers.zipWithIndex.map(_.swap).filter {
    case (_, _: Convolution[_])   => true
    case _                        => false
  }.toMap.mapValues {
    case c: Convolution[Double]   => c.copy(activator = c.activator.map[Float](_.toDouble, _.toFloat))
  }

  private val _outputDim = _allLayers.last.neurons
  private val _lastC     = _convLayers.maxBy(_._1)._1
  private val _lastL     = _allLayers.indices.last

  private type Indices   = Map[(Int, Int), DenseMatrix[Int]]
  private val _indices   = collection.mutable.Map.empty[Int, Indices]

  /**
    * Computes output for `x`.
    */
  def apply(x: Matrices): Vector = {
    _focusLayer.map { cl =>
      flow(x, layers.indexOf(cl))
    }.getOrElse {
      val r = flow(x, _lastWlayerIdx)
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
  def train(xs: Seq[Matrices], ys: Vectors): Unit = {
    require(xs.size == ys.size, "Mismatch between sample sizes!")
    import settings._
    val batchSize = settings.batchSize.getOrElse(xs.size)
    if (settings.verbose) info(s"Training with ${xs.size} samples, batch size = $batchSize, batches = ${math.ceil(xs.size.toDouble / batchSize.toDouble).toInt} ...")
    val xsys = xs.zip(ys.map(_.asDenseMatrix)).grouped(batchSize).toSeq
    run(xsys, learningRate(1 -> 1.0).toFloat, xs.size, precision, 1, iterations)
  }

  /**
    * The training loop.
    */
  @tailrec private def run(xsys: Seq[Seq[(Matrices, Matrix)]], stepSize: Float, sampleSize: Float, precision: Double,
                           iteration: Int, maxIterations: Int): Unit = {
    val _em = xsys.map { batch =>
      val (x, y) = (batch.map(_._1), batch.map(_._2))
      val error =
        if (settings.approximation.isDefined)
          adaptWeightsApprox(x, y, stepSize)
        else adaptWeights(x, y, stepSize)
        debug(s"Batch Error: $error")
      error
    }.reduce(_ + _)
    val errorPerS = _em / sampleSize
    val errorMean = mean(errorPerS)
    if (settings.verbose) info(f"Iteration $iteration - Loss $errorMean%.6g - Loss Vector $errorPerS")
    maybeGraph(errorMean)
    keepBest(errorMean)
    waypoint(iteration)
    if (errorMean > precision && iteration < maxIterations) {
      run(xsys, settings.learningRate(iteration + 1 -> stepSize).toFloat, sampleSize, precision, iteration + 1, maxIterations)
    } else {
      info(f"Took $iteration iterations of $maxIterations with Loss = $errorMean%.6g")
      takeBest()
    }
  }

  private def flow(in: Matrices, target: Int): Matrix = {

    val _fa = ArrayBuffer.empty[Matrix]

    @tailrec def conv(_in: Matrices, i: Int): Unit = {
      val l = _convLayers(i)
      val p = weights(i) * im2col(_in, l)._1
      val a = p.map(l.activator)
      _fa += a
      if (i < _lastC) conv(col2im(a, l.dimOut), i + 1)
    }

    @tailrec def fully(_in: Matrix, i: Int): Unit = {
      val l = _activators(i)
      val p = _in * weights(i)
      val a = p.map(l)
      _fa += a
      if (i < _lastL) fully(a, i + 1)
    }

    conv(in, 0)
    fully(_fa(_lastC).reshape(1, _convLayers(_lastC).neurons), _lastC + 1)

    _fa(target)

  }

  private def im2col(ms: Matrices, l: Convolution[_], withIndices: Boolean = false): (Matrix, Indices) = {
    val out = DenseMatrix.zeros[Float](l.field._1 * l.field._2 * l.dimIn._3, l.dimOut._1 * l.dimOut._2)
    val idc = if (withIndices) {
      ms.head.keysIterator.map { k =>
        k -> DenseMatrix.zeros[Int](l.field._1, l.field._2)
      }.toMap
    } else null
    var (x, y, i) = (0, 0, 0)
    while (x < l.dimOut._1) {
      while (y < l.dimOut._2) {
        var (fX, fY) = (0, 0)
        while (fX < l.field._1) {
          while (fY < l.field._2) {
            var z = 0
            val (a, b) = (fY + (y * l.stride._2), fX + (x * l.stride._1))
            if (a >= l.padding._2 && a < (l.dimIn._2 + l.padding._2) && b >= l.padding._1 && b < (l.dimIn._1 + l.padding._1)) {
              while (z < l.dimIn._3) {
                val c = z * l.field._1 * l.field._2
                val (_a, _b) = (a - l.padding._2, b - l.padding._1)
                val value = ms(z)(_a, _b)
                val k = fX * l.field._2 + fY
                out.update(c + k, x * l.dimOut._2 + y, value)
                if (withIndices) idc(_a, _b).update(fX, fY, i + 1)
                z += 1
              }
            }
            fY += 1
          }
          fY  = 0
          fX += 1
        }
        i += 1
        y += 1
      }
      y  = 0
      x += 1
    }
    (out, idc)
  }

  private def im2col_backprop(d: Matrix, c: Convolution[_], idx: Indices): Matrix = {
    val dp = padLeft(d, Dimensions2(d.rows, d.cols + 1), Zero)
    val dc = DenseMatrix.zeros[Float](c.field._1 * c.field._2 * c.filters, c.dimIn._1 * c.dimIn._2)
    var z  = 0
    while (z < c.filters) {
      val _de = dp(z, ::)
      var (x, y, q) = (0, 0, 0)
      while (x < c.dimIn._1) {
        while (y < c.dimIn._2) {
          var p = 0
          idx(y, x).foreachPair { (pp, v) =>
            val t = (z * c.field._1 * c.field._2 + p, q)
            dc.update(t, _de(v))
            p += 1
          }
          y += 1
          q += 1
        }
        y = 0
        x += 1
      }
      z += 1
    }
    dc
  }

  private def col2im(matrix: Matrix, dim: (Int, Int, Int)): Matrices = {
    var i = 0
    val out = new Array[Matrix](dim._3)
    while (i < dim._3) {
      val v = matrix.t(::, i).asDenseMatrix.reshape(dim._2, dim._1)
      out(i) = v
      i += 1
    }
    out
  }

  /**
    * Computes gradient for all weights in parallel,
    * adapts their value using gradient descent and returns the error matrix.
    */
  private def adaptWeights(xs: Seq[Matrices], ys: Seq[Matrix], stepSize: Float): Matrix = {

    import settings.lossFunction

    val xsys = xs.par.zip(ys)
    xsys.tasksupport = _forkJoinTaskSupport

    val _dws = (0 to _lastWlayerIdx).map { i =>
      i -> DenseMatrix.zeros[Float](weights(i).rows, weights(i).cols)
    }.toMap

    val _ww = _convLayers.map {
      case (i, _) if i < _lastC =>
        val l  = _convLayers(i)
        val l2 = _convLayers(i + 1)
        val fieldSq = l2.field._1 * l2.field._2
        val wr = weights(i + 1)
        val ww = DenseMatrix.zeros[Float](l.filters, l2.filters * fieldSq)
        var (filter, depth) = (0, 0)
        while (filter < l2.filters) {
          while (depth < l.filters) {
            val ws = wr(filter, (depth * fieldSq) until ((depth * fieldSq) + fieldSq))
            var i = 0
            while (i < fieldSq) {
              ww.update(depth, filter * fieldSq + i, ws(i))
              i += 1
            }
            depth += 1
          }
          depth = 0
          filter += 1
        }
        i -> ww
      case (i, _) => i -> null
    } - _lastC

    val _errSum  = DenseMatrix.zeros[Float](1, _outputDim)

    xsys.map { xy =>

      val (x, y) = xy
      val fa  = collection.mutable.Map.empty[Int, Matrix]
      val fb  = collection.mutable.Map.empty[Int, Matrix]
      val fc  = collection.mutable.Map.empty[Int, Matrix]
      val dws = collection.mutable.Map.empty[Int, Matrix]
      val ds  = collection.mutable.Map.empty[Int, Matrix]
      val e   = DenseMatrix.zeros[Float](1, _outputDim)

      @tailrec def conv(_in: Matrices, i: Int): Unit = {
        val l = _convLayers(i)
        val seen = _indices.isDefinedAt(i)
        val (c, x) = im2col(_in, l, withIndices = !seen)
        val p = weights(i) * c
        var a = p.map(l.activator)
        var b = p.map(l.activator.derivative)
        if (i == _lastC) {
          a = a.reshape(1, l.neurons)
          b = b.reshape(1, l.neurons)
        }
        fa += i -> a
        fb += i -> b
        fc += i -> c
        if (!seen) _indices += i -> x
        if (i < _lastC) conv(col2im(a, l.dimOut), i + 1)
      }

      @tailrec def fully(_in: Matrix, i: Int): Unit = {
        val l = _activators(i)
        val p = _in * weights(i)
        val a = p.map(l)
        val b = p.map(l.derivative)
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
          e += err
          derive(i - 1)
        } else if (i < _lastWlayerIdx && i > _lastC) {
          val d = (ds(i + 1) * weights(i + 1).t) *:* fb(i)
          val dw = fa(i - 1).t * d
          dws += i -> dw
          ds += i -> d
          derive(i - 1)
        } else if (i == _lastC) {
          val l = _convLayers(i)
          val d = ((ds(i + 1) * weights(i + 1).t) *:* fb(i))
            .reshape(l.filters, l.dimOut._1 * l.dimOut._2)
          val dw = d * fc(i).t
          dws += i -> dw
          ds += i -> d
          if (i > 0) derive(i - 1)
        } else {
          val dc = im2col_backprop(ds(i + 1), _convLayers(i + 1), _indices(i + 1))
          val d = _ww(i) * dc *:* fb(i)
          val dw = d * fc(i).t
          dws += i -> dw
          ds += i -> d
          if (i > 0) derive(i - 1)
        }
      }

      conv(x, 0)
      fully(fa(_lastC), _lastC + 1)
      derive(_lastWlayerIdx)
      (dws, e)

    }.seq.foreach { ab =>
      _errSum += ab._2
      var i = 0
      while (i <= _lastWlayerIdx) {
        val m = _dws(i)
        val n = ab._1(i)
        m += n
        i += 1
      }
    }

    var i = 0
    while (i <= _lastWlayerIdx) {
      settings.updateRule(weights(i), _dws(i), stepSize, i)
      i += 1
    }

    _errSum

  }

  /** Approximates the gradient based on finite central differences. (For debugging) */
  private def adaptWeightsApprox(xs: Seq[Matrices], ys: Matrices, stepSize: Float): Matrix = {

    require(settings.updateRule.isInstanceOf[Debuggable[Float]])
    val _rule: Debuggable[Float] = settings.updateRule.asInstanceOf[Debuggable[Float]]

    def errorFunc(): Matrix = {
      val xsys = xs.zip(ys).par
      xsys.tasksupport = _forkJoinTaskSupport
      xsys.map { case (x, y) => settings.lossFunction(y, flow(x, _lastWlayerIdx))._1 }.reduce(_ + _)
    }

    val out = errorFunc()

    def approximateErrorFuncDerivative(weightLayer: Int, weight: (Int, Int)): Matrix = {
      val Δ = settings.approximation.get.Δ.toFloat
      val v = weights(weightLayer)(weight)
      weights(weightLayer).update(weight, v - Δ)
      val a = errorFunc()
      weights(weightLayer).update(weight, v + Δ)
      val b = errorFunc()
      weights(weightLayer).update(weight, v)
      (b - a) / (2 * Δ)
    }

    val updates = collection.mutable.HashMap.empty[(Int, (Int, Int)), Float]
    val grads   = collection.mutable.HashMap.empty[(Int, (Int, Int)), Float]
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
