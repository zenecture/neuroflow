package neuroflow.nets.cpu

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import neuroflow.core.Network._
import neuroflow.core._
import neuroflow.nets.Registry

import scala.annotation.tailrec
import scala.collection.Seq
import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

/**
  *
  * This is a convolutional feed-forward neural network.
  * It uses gradient descent to optimize the error function Σ1/2(y - net(x))².
  *
  * Use the parallelism parameter with care, as it greatly affects memory usage.
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
                                     identifier: String = Registry.register(), numericPrecision: String = "Double")
  extends CNN[Double] with KeepBestLogic[Double] with WaypointLogic[Double] {

  import Convolution.IntTupler

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

  private val _clusterLayer       = layers.collect { case c: Focus[_] => c }.headOption
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
    _clusterLayer.map { cl =>
      flow(x, layers.indexOf(cl)).toDenseVector
    }.getOrElse {
      flow(x, _lastWlayerIdx).toDenseVector
    }
  }

  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Seq[Matrices], ys: Vectors): Unit = {
    require(xs.size == ys.size, "Mismatch between sample sizes!")
    import settings._
    val batchSize = settings.batchSize.getOrElse(xs.size)
    if (settings.verbose) info(s"Training with ${xs.size} samples, batchize = $batchSize ...")
    val xsys = xs.zip(ys.map(_.asDenseMatrix)).grouped(batchSize).toSeq
    run(xsys, learningRate(0 -> 1.0), xs.size, batchSize, precision, 1, iterations)
  }

  /**
    * The training loop.
    */
  @tailrec private def run(xsys: Seq[Seq[(Matrices, Matrix)]], stepSize: Double, sampleSize: Int, batchSize: Int, precision: Double,
                           iteration: Int, maxIterations: Int): Unit = {
    val _em = xsys.map { batch =>
      val (x, y) = (batch.map(_._1), batch.map(_._2))
      val error =
        if (settings.approximation.isDefined)
          adaptWeightsApprox(x, y, stepSize)
        else adaptWeights(x, y, stepSize)
      error
    }.reduce(_ + _)
    val errorMean = mean(_em)
    val errorRel  = math.sqrt((errorMean / sampleSize.toDouble) * 2.0)
    if (settings.verbose) info(f"Iteration $iteration - Mean Error $errorMean%.6g (≈ $errorRel%.3g rel.) - Error Vector ${_em}")
    maybeGraph(errorMean)
    keepBest(errorMean, weights)
    waypoint(iteration)
    if (errorMean > precision && iteration < maxIterations) {
      run(xsys, settings.learningRate(iteration + 1 -> stepSize), sampleSize, batchSize, precision, iteration + 1, maxIterations)
    } else {
      if (settings.verbose) info(f"Took $iteration iterations of $maxIterations with Mean Error = $errorMean%.6g")
      takeBest()
    }
  }

  private def flow(in: Matrices, target: Int): Matrix = {

    val _fa = ArrayBuffer.empty[Matrix]

    @tailrec def conv(_in: Matrices, i: Int): Unit = {
      val l = _convLayers(i)
      val p = weights(i) * im2col(_in, l.field, l.stride`²`)._1
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

  private def im2col(ms: Matrices, field: (Int, Int), stride: (Int, Int), withIndices: Boolean = false): (Matrix, Indices) = {
    val dim = (ms.head.rows, ms.head.cols, ms.size)
    val dimOut = ((dim._1 - field._1) / stride._1 + 1, (dim._2 - field._2) / stride._2 + 1)
    val fieldSq = field._1 * field._2
    val out = DenseMatrix.zeros[Double](fieldSq * dim._3, dimOut._1 * dimOut._2)
    val idc = if (withIndices) {
      ms.head.keysIterator.map { k =>
        k -> DenseMatrix.zeros[Int](field._1, field._2)
      }.toMap
    } else null
    var (w, h, i) = (0, 0, 0)
    while (w < ((dim._1 - field._1) / stride._1) + 1) {
      while (h < ((dim._2 - field._2) / stride._2) + 1) {
        var (x, y, z, wi) = (0, 0, 0, 0)
        while (x < field._1) {
          while (y < field._2) {
            while (z < dim._3) {
              val (a, b, c) = (x + (w * stride._1), y + (h * stride._2), z * fieldSq)
              val value = ms(z)(a, b)
              val lin = c + wi
              out.update(lin, i, value)
              if (withIndices) idc(a, b).update(x, y, i + 1)
              z += 1
            }
            z   = 0
            wi += 1
            y += 1
          }
          y  = 0
          x += 1
        }
        i += 1
        h += 1
      }
      h  = 0
      w += 1
    }
    (out, idc)
  }

  private def col2im(matrix: Matrix, dim: (Int, Int, Int)): Matrices = {
    var i = 0
    val out = new Array[Matrix](dim._3)
    while (i < dim._3) {
      val v = matrix.t(::, i).asDenseMatrix.reshape(dim._2, dim._1).t
      out(i) = v
      i += 1
    }
    out
  }

  private def adaptWeights(xs: Seq[Matrices], ys: Seq[Matrix], stepSize: Double): Matrix = {

    val xsys = xs.par.zip(ys)
    xsys.tasksupport = _forkJoinTaskSupport

    val _ds = (0 to _lastWlayerIdx).map { i =>
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
      case (i, _) => i -> weights(i)
    }

    val _errSum  = DenseMatrix.zeros[Double](1, _outputDim)
    val _square  = DenseMatrix.zeros[Double](1, _outputDim)
    _square := 2.0

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
        val (c, x) = im2col(_in, l.field, l.stride`²`, withIndices = !seen)
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
          val yf = y - fa(i)
          val d = -yf *:* fb(i)
          val dw = fa(i - 1).t * d
          dws += i -> dw
          ds += i -> d
          e += yf
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
          val l1 = _convLayers(i + 1)
          val id = _indices(i + 1)
          val de = ds(i + 1)
          val fs = l1.field._1 * l1.field._2
          val dc = DenseMatrix.zeros[Double](fs * l1.filters, l1.dimIn._1 * l1.dimIn._2)
          var f  = 0
          while (f < de.rows) {
            var (x, y, q) = (0, 0, 0)
            while (x < l1.dimIn._1) {
              while (y < l1.dimIn._2) {
                var p = 0
                id(x, y).foreachPair { (_, v) =>
                  val t = (f * fs + p, q)
                  val d = if (v > 0) de(f, v - 1) else 0.0
                  dc.update(t, d)
                  p += 1
                }
                y += 1
                q += 1
              }
              y = 0
              x += 1
            }
            f += 1
          }
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
      e :^= _square
      e *= 0.5
      (dws, e)
    }.seq.foreach { ab =>
      _errSum += ab._2
      var i = 0
      while (i <= _lastWlayerIdx) {
        val m = _ds(i)
        val n = ab._1(i)
        m += n
        i += 1
      }
    }

    var i = 0
    while (i <= _lastWlayerIdx) {
      settings.updateRule(weights(i), _ds(i), stepSize, i)
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
      xsys.map { case (x, y) => 0.5 * pow(y - flow(x, _lastWlayerIdx), 2) }.reduce(_ + _)
    }

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

    errorFunc()

  }

}

//</editor-fold>

//<editor-fold defaultstate="collapsed" desc="Single Precision Impl">

private[nets] case class ConvNetworkSingle(layers: Seq[Layer], settings: Settings[Float], weights: Weights[Float],
                                           identifier: String = Registry.register(), numericPrecision: String = "Single")
  extends CNN[Float] with KeepBestLogic[Float] with WaypointLogic[Float] {

  import Convolution.IntTupler

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

  private val _clusterLayer       = layers.collect { case c: Focus[_] => c }.headOption
  private val _lastWlayerIdx      = weights.size - 1
  private val _fullLayers         = _allLayers.map { hd => hd.activator.map[Float](_.toDouble, _.toFloat) }
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
    _clusterLayer.map { cl =>
      flow(x, layers.indexOf(cl)).toDenseVector
    }.getOrElse {
      flow(x, _lastWlayerIdx).toDenseVector
    }
  }

  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Seq[Matrices], ys: Vectors): Unit = {
    require(xs.size == ys.size, "Mismatch between sample sizes!")
    import settings._
    val batchSize = settings.batchSize.getOrElse(xs.size)
    if (settings.verbose) info(s"Training with ${xs.size} samples, batchize = $batchSize ...")
    val xsys = xs.zip(ys.map(_.asDenseMatrix)).grouped(batchSize).toSeq
    run(xsys, learningRate(0 -> 1.0).toFloat, xs.size, batchSize, precision, 1, iterations)
  }

  /**
    * The training loop.
    */
  @tailrec private def run(xsys: Seq[Seq[(Matrices, Matrix)]], stepSize: Float, sampleSize: Int, batchSize: Int, precision: Double,
                           iteration: Int, maxIterations: Int): Unit = {
    val _em = xsys.map { batch =>
      val (x, y) = (batch.map(_._1), batch.map(_._2))
      val error =
        if (settings.approximation.isDefined)
          adaptWeightsApprox(x, y, stepSize)
        else adaptWeights(x, y, stepSize)
      error
    }.reduce(_ + _)
    val errorMean = mean(_em)
    val errorRel  = math.sqrt((errorMean / sampleSize.toDouble) * 2.0)
    if (settings.verbose) info(f"Iteration $iteration - Mean Error $errorMean%.6g (≈ $errorRel%.3g rel.) - Error Vector ${_em}")
    maybeGraph(errorMean)
    keepBest(errorMean, weights)
    waypoint(iteration)
    if (errorMean > precision && iteration < maxIterations) {
      run(xsys, settings.learningRate(iteration + 1 -> stepSize).toFloat, sampleSize, batchSize, precision, iteration + 1, maxIterations)
    } else {
      if (settings.verbose) info(f"Took $iteration iterations of $maxIterations with Mean Error = $errorMean%.6g")
      takeBest()
    }
  }

  private def flow(in: Matrices, target: Int): Matrix = {

    val _fa = ArrayBuffer.empty[Matrix]

    @tailrec def conv(_in: Matrices, i: Int): Unit = {
      val l = _convLayers(i)
      val p = weights(i) * im2col(_in, l.field, l.stride`²`)._1
      val a = p.map(l.activator)
      _fa += a
      if (i < _lastC) conv(col2im(a, l.dimOut), i + 1)
    }

    @tailrec def fully(_in: Matrix, i: Int): Unit = {
      val l = _fullLayers(i)
      val p = _in * weights(i)
      val a = p.map(l)
      _fa += a
      if (i < _lastL) fully(a, i + 1)
    }

    conv(in, 0)
    fully(_fa(_lastC).reshape(1, _convLayers(_lastC).neurons), _lastC + 1)

    _fa(target)

  }

  private def im2col(ms: Matrices, field: (Int, Int), stride: (Int, Int), withIndices: Boolean = false): (Matrix, Indices) = {
    val dim = (ms.head.rows, ms.head.cols, ms.size)
    val dimOut = ((dim._1 - field._1) / stride._1 + 1, (dim._2 - field._2) / stride._2 + 1)
    val fieldSq = field._1 * field._2
    val out = DenseMatrix.zeros[Float](fieldSq * dim._3, dimOut._1 * dimOut._2)
    val idc = if (withIndices) {
      ms.head.keysIterator.map { k =>
        k -> DenseMatrix.zeros[Int](field._1, field._2)
      }.toMap
    } else null
    var (w, h, i) = (0, 0, 0)
    while (w < ((dim._1 - field._1) / stride._1) + 1) {
      while (h < ((dim._2 - field._2) / stride._2) + 1) {
        var (x, y, z, wi) = (0, 0, 0, 0)
        while (x < field._1) {
          while (y < field._2) {
            while (z < dim._3) {
              val (a, b, c) = (x + (w * stride._1), y + (h * stride._2), z * fieldSq)
              val value = ms(z)(a, b)
              val lin = c + wi
              out.update(lin, i, value)
              if (withIndices) idc(a, b).update(x, y, i + 1)
              z += 1
            }
            z   = 0
            wi += 1
            y += 1
          }
          y  = 0
          x += 1
        }
        i += 1
        h += 1
      }
      h  = 0
      w += 1
    }
    (out, idc)
  }

  private def col2im(matrix: Matrix, dim: (Int, Int, Int)): Matrices = {
    var i = 0
    val out = new Array[Matrix](dim._3)
    while (i < dim._3) {
      val v = matrix.t(::, i).asDenseMatrix.reshape(dim._2, dim._1).t
      out(i) = v
      i += 1
    }
    out
  }

  private def adaptWeights(xs: Seq[Matrices], ys: Seq[Matrix], stepSize: Float): Matrix = {

    val xsys = xs.par.zip(ys)
    xsys.tasksupport = _forkJoinTaskSupport

    val _ds = (0 to _lastWlayerIdx).map { i =>
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
      case (i, _) => i -> weights(i)
    }

    val _errSum  = DenseMatrix.zeros[Float](1, _outputDim)
    val _square  = DenseMatrix.zeros[Float](1, _outputDim)
    _square := 2.0f

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
        val (c, x) = im2col(_in, l.field, l.stride`²`, withIndices = !seen)
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
        val l = _fullLayers(i)
        val p = _in * weights(i)
        val a = p.map(l)
        val b = p.map(l.derivative)
        fa += i -> a
        fb += i -> b
        if (i < _lastL) fully(a, i + 1)
      }

      @tailrec def derive(i: Int): Unit = {
        if (i == _lastWlayerIdx) {
          val yf = y - fa(i)
          val d = -yf *:* fb(i)
          val dw = fa(i - 1).t * d
          dws += i -> dw
          ds += i -> d
          e += yf
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
          val l1 = _convLayers(i + 1)
          val id = _indices(i + 1)
          val de = ds(i + 1)
          val fs = l1.field._1 * l1.field._2
          val dc = DenseMatrix.zeros[Float](fs * l1.filters, l1.dimIn._1 * l1.dimIn._2)
          var f  = 0
          while (f < de.rows) {
            var (x, y, q) = (0, 0, 0)
            while (x < l1.dimIn._1) {
              while (y < l1.dimIn._2) {
                var p = 0
                id(x, y).foreachPair { (_, v) =>
                  val t = (f * fs + p, q)
                  val d = if (v > 0) de(f, v - 1) else 0.0f
                  dc.update(t, d)
                  p += 1
                }
                y += 1
                q += 1
              }
              y = 0
              x += 1
            }
            f += 1
          }
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
      e :^= _square
      e *= 0.5f
      (dws, e)
    }.seq.foreach { ab =>
      _errSum += ab._2
      var i = 0
      while (i <= _lastWlayerIdx) {
        val m = _ds(i)
        val n = ab._1(i)
        m += n
        i += 1
      }
    }

    var i = 0
    while (i <= _lastWlayerIdx) {
      settings.updateRule(weights(i), _ds(i), stepSize, i)
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
      xsys.map { case (x, y) => 0.5f * pow(y - flow(x, _lastWlayerIdx), 2) }.reduce(_ + _)
    }

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

    errorFunc()

  }

}

//</editor-fold>
