package neuroflow.nets.gpu

import breeze.linalg.Options.{Dimensions2, Zero}
import breeze.linalg._
import breeze.stats._
import jcuda.jcublas.{JCublas2, cublasHandle}
import neuroflow.core.Activator._
import neuroflow.core.IllusionBreaker.SettingsNotSupportedException
import neuroflow.core.Network._
import neuroflow.core._
import neuroflow.nets.gpu.cuda.CuMatrix

import scala.annotation.tailrec
import scala.collection.Seq
import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

/**
  *
  * This is a convolutional feed-forward neural network running on CPU and CUDA.
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

// <editor-fold defaultstate="collapsed" desc="Double Precision Impl">

private[nets] case class ConvNetworkDouble(layers: Seq[Layer], settings: Settings[Double], weights: Weights[Double],
                                           identifier: String = "neuroflow.nets.gpu.ConvNetwork", numericPrecision: String = "Double")
  extends CNN[Double] with KeepBestLogic[Double] with WaypointLogic[Double] {

  implicit val handle = new cublasHandle
  JCublas2.cublasCreate(handle)

  type Vector   = Network.Vector[Double]
  type Vectors  = Network.Vectors[Double]
  type Matrix   = Network.Matrix[Double]
  type Matrices = Network.Matrices[Double]

  private val _forkJoinTaskSupport = new ForkJoinTaskSupport(new ForkJoinPool(settings.parallelism.getOrElse(1)))

  private val _allLayers = layers.map {
    case f: Focus[_]         => f.inner
    case d: Dense[_]         => d
    case c: Convolution[_]   => c
    case o: Output[_]        => o
  }.toArray

  private val _focusLayer         = layers.collect { case c: Focus[_] => c }.headOption
  private val _lastWlayerIdx      = weights.size - 1

  private val _activators         = _allLayers.map { l =>
    l.activator match {
      case ReLU    => CuMatrix.Activators.relu[Double]    ->  CuMatrix.Activators.relu_derivative[Double]
      case Linear  => CuMatrix.Activators.linear[Double]  ->  CuMatrix.Activators.linear_derivative[Double]
      case Sigmoid => CuMatrix.Activators.sigmoid[Double] ->  CuMatrix.Activators.sigmoid_derivative[Double]
      case Tanh    => CuMatrix.Activators.tanh[Double]    ->  CuMatrix.Activators.tanh_derivative[Double]
      case x       => throw new SettingsNotSupportedException(s"This activator is not implemented for CUDA: $x.")
    }
  }

  private val _convLayers         =  _allLayers.zipWithIndex.map(_.swap).filter {
    case (_, _: Convolution[_])   => true
    case _                        => false
  }.toMap.mapValues {
    case c: Convolution[Double]   => c
  }

  private val _outputDim = _allLayers.last.neurons
  private val _lastC     = _convLayers.maxBy(_._1)._1
  private val _lastL     = _allLayers.indices.last

  private val _cuWeights = weights.map(m => CuMatrix.fromDense(m))

  /**
    * Checks if the [[Settings]] are properly defined.
    * Might throw a [[SettingsNotSupportedException]].
    */
  override def checkSettings(): Unit = {
    super.checkSettings()
    if (settings.specifics.isDefined)
      warn("No specific settings supported. This has no effect.")
    settings.regularization.foreach {
      case KeepBest =>
      case _ => throw new SettingsNotSupportedException("This regularization is not supported.")
    }
  }

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
    if (settings.verbose) info(s"Training with ${xs.size} samples, batchSize = $batchSize ...")
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
      error
    }.reduce(_ + _)
    val errorPerS = _em / sampleSize
    val errorMean = mean(errorPerS)
    if (settings.verbose) info(f"Iteration $iteration - Loss $errorMean%.6g - Loss Vector $errorPerS")
    syncWeights()
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

    val fa = ArrayBuffer.empty[CuMatrix[Double]]

    @tailrec def conv(in: CuMatrix[Double], i: Int): Unit = {
      val l = _convLayers(i)
      val c = CuMatrix.ConvOps.im2col(in, i, l.dimIn, l.field, l.padding, l.stride)
      val p = _cuWeights(i) * c
      val a = _activators(i)._1(p)
      fa += a
      in.release()
      c.release()
      p.release()
      if (i < _lastC) conv(a, i + 1)
    }

    @tailrec def fully(in: CuMatrix[Double], i: Int): Unit = {
      val p = in * _cuWeights(i)
      val a = _activators(i)._1(p)
      fa += a
      p.release()
      if (i < _lastL) fully(a, i + 1)
    }

    conv(ms2CuMat(in), 0)
    val cuIn = fa(_lastC).reshape(1, _convLayers(_lastC).neurons)
    fully(cuIn, _lastC + 1)

    val o = fa(target).toDense
    fa.foreach(_.release())
    o

  }

  private def ms2CuMat(ms: Matrices): CuMatrix[Double] = {
    val dm = ms.map { m =>
      m.reshape(1, m.size)
    }.reduce(DenseMatrix.vertcat(_, _))
    CuMatrix.fromDense(dm)
  }

  private def adaptWeights(xs: Seq[Matrices], ys: Seq[Matrix], stepSize: Double): Matrix = {

    import settings.lossFunction

    val cuxs = xs.map(ms => ms2CuMat(ms))
    val cuys = ys.map(m => CuMatrix.fromDense(m))
    val xsys = cuxs.zip(cuys)

    val _dws = (0 to _lastWlayerIdx).map { i =>
      i -> CuMatrix.zeros[Double](weights(i).rows, weights(i).cols)
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
        i -> CuMatrix.fromDense(ww)
      case (i, _) => i -> null
    } - _lastC

    val _errSum  = CuMatrix.zeros[Double](1, _outputDim)

    xsys.foreach { xy =>

      val (x, y) = xy
      val fa  = collection.mutable.Map.empty[Int, CuMatrix[Double]]
      val fb  = collection.mutable.Map.empty[Int, CuMatrix[Double]]
      val fc  = collection.mutable.Map.empty[Int, CuMatrix[Double]]
      val ds  = collection.mutable.Map.empty[Int, CuMatrix[Double]]

      @tailrec def conv(in: CuMatrix[Double], i: Int): Unit = {
        val l = _convLayers(i)
        val c = CuMatrix.ConvOps.im2col(in, i, l.dimIn, l.field, l.padding, l.stride)
        val p = _cuWeights(i) * c
        var a = _activators(i)._1(p)
        var b = _activators(i)._2(p)
        if (i == _lastC) {
          a = a.reshape(1, l.neurons)
          b = b.reshape(1, l.neurons)
        }
        fa += i -> a
        fb += i -> b
        fc += i -> c
        in.release()
        p.release()
        if (i < _lastC) conv(a, i + 1)
      }

      @tailrec def fully(_in: CuMatrix[Double], i: Int): Unit = {
        val p = _in * _cuWeights(i)
        val a = _activators(i)._1(p)
        val b = _activators(i)._2(p)
        fa += i -> a
        fb += i -> b
        p.release()
        if (i < _lastL) fully(a, i + 1)
      }

      @tailrec def derive(i: Int): Unit = {
        if (i == _lastWlayerIdx) {
          val (err, grad) = lossFunction(y, fa(i))
          val d = grad *:* fb(i)
          val dw = fa(i - 1).t * d
          _dws(i) += dw
          _errSum += err
          ds += i -> d
          err.release()
          grad.release()
          dw.release()
          derive(i - 1)
        } else if (i < _lastWlayerIdx && i > _lastC) {
          val d1 = ds(i + 1) * _cuWeights(i + 1).t
          val d2 = d1 *:* fb(i)
          val dw = fa(i - 1).t * d2
          _dws(i) += dw
          ds += i -> d2
          d1.release()
          dw.release()
          derive(i - 1)
        } else if (i == _lastC) {
          val l = _convLayers(i)
          val d1 = ds(i + 1) * _cuWeights(i + 1).t
          val d2 = (d1 *:* fb(i)).reshape(l.filters, l.dimOut._1 * l.dimOut._2)
          val dw = d2 * fc(i).t
          _dws(i) += dw
          ds += i -> d2
          d1.release()
          dw.release()
          if (i > 0) derive(i - 1)
        } else {
          val l = _convLayers(i)
          val de = ds(i + 1)
          val dc = CuMatrix.ConvOps.im2col_backprop(de, l.dimOut, i + 1)
          val d = _ww(i) * dc *:* fb(i)
          val dw = d * fc(i).t
          _dws(i) += dw
          ds += i -> d
          dc.release()
          dw.release()
          if (i > 0) derive(i - 1)
        }
      }

      conv(x, 0)
      fully(fa(_lastC), _lastC + 1)
      derive(_lastWlayerIdx)

      fa.values.foreach(_.release())
      fb.values.foreach(_.release())
      fc.values.foreach(_.release())
      ds.values.foreach(_.release())

    }

    var i = 0
    while (i <= _lastWlayerIdx) {
      settings.updateRule(_cuWeights(i), _dws(i), stepSize, i)
      i += 1
    }

    xsys.foreach(_._2.release())
    _dws.values.foreach(_.release())
    _ww.values.foreach(_.release())
    val es = _errSum.toDense
    _errSum.release()

    es

  }

  private def syncWeights(): Unit = {
    weights.zip(_cuWeights).foreach {
      case (w, cw) => w := cw.toDense
    }
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

    def approximateErrorFuncDerivative(weightLayer: Int, weight: (Int, Int)): Matrix = {
      val Δ = settings.approximation.get.Δ
      val v = weights(weightLayer)(weight)
      weights(weightLayer).update(weight, v - Δ)
      syncWeightsBack()
      val a = errorFunc()
      weights(weightLayer).update(weight, v + Δ)
      syncWeightsBack()
      val b = errorFunc()
      weights(weightLayer).update(weight, v)
      syncWeightsBack()
      (b - a) / (2 * Δ)
    }

    def syncWeightsBack(): Unit = {
      weights.zip(_cuWeights).foreach {
        case (w, cw) => cw := w
      }
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

    syncWeightsBack()

    errorFunc()

  }

}

// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="Single Precision Impl">

private[nets] case class ConvNetworkSingle(layers: Seq[Layer], settings: Settings[Float], weights: Weights[Float],
                                           identifier: String = "neuroflow.nets.gpu.ConvNetwork", numericPrecision: String = "Single")
  extends CNN[Float] with KeepBestLogic[Float] with WaypointLogic[Float] {

  implicit val handle = new cublasHandle
  JCublas2.cublasCreate(handle)

  type Vector   = Network.Vector[Float]
  type Vectors  = Network.Vectors[Float]
  type Matrix   = Network.Matrix[Float]
  type Matrices = Network.Matrices[Float]

  private val _forkJoinTaskSupport = new ForkJoinTaskSupport(new ForkJoinPool(settings.parallelism.getOrElse(1)))

  private val _allLayers = layers.map {
    case f: Focus[_]         => f.inner
    case d: Dense[_]         => d
    case c: Convolution[_]   => c
    case o: Output[_]        => o
  }.toArray

  private val _focusLayer         = layers.collect { case c: Focus[_] => c }.headOption
  private val _lastWlayerIdx      = weights.size - 1

  private val _fullLayers         = _allLayers.map { l =>
    l.activator match {
      case ReLU    => CuMatrix.Activators.relu[Float]    ->  CuMatrix.Activators.relu_derivative[Float]
      case Linear  => CuMatrix.Activators.linear[Float]  ->  CuMatrix.Activators.linear_derivative[Float]
      case Sigmoid => CuMatrix.Activators.sigmoid[Float] ->  CuMatrix.Activators.sigmoid_derivative[Float]
      case Tanh    => CuMatrix.Activators.tanh[Float]    ->  CuMatrix.Activators.tanh_derivative[Float]
      case x       => throw new SettingsNotSupportedException(s"This activator is not implemented for CUDA: $x.")
    }
  }

  private val _convLayers         =  _allLayers.zipWithIndex.map(_.swap).filter {
    case (_, _: Convolution[_])   => true
    case _                        => false
  }.toMap.mapValues {
    case c: Convolution[_]   => c
  }

  private val _parArrayPool = {
    val range = _convLayers.flatMap(l => Seq(l._2.dimIn._3, l._2.dimOut._3))
    range.map { size =>
      val pa = (0 until size).toParArray
      pa.tasksupport = _forkJoinTaskSupport
      size -> pa
    }.toMap
  }

  private val _outputDim = _allLayers.last.neurons
  private val _lastC     = _convLayers.maxBy(_._1)._1
  private val _lastL     = _allLayers.indices.last

  private type Indices   = Map[(Int, Int), DenseMatrix[Int]]
  private val _indices   = collection.mutable.Map.empty[Int, Indices]
  private val _cuWeights = weights.map(m => CuMatrix.fromDense(m))

  /**
    * Checks if the [[Settings]] are properly defined.
    * Might throw a [[SettingsNotSupportedException]].
    */
  override def checkSettings(): Unit = {
    super.checkSettings()
    if (settings.specifics.isDefined)
      warn("No specific settings supported. This has no effect.")
    settings.regularization.foreach {
      case KeepBest =>
      case _ => throw new SettingsNotSupportedException("This regularization is not supported.")
    }
  }

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
    if (settings.verbose) info(s"Training with ${xs.size} samples, batchSize = $batchSize ...")
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
      error
    }.reduce(_ + _)
    val errorPerS = _em / sampleSize
    val errorMean = mean(errorPerS)
    if (settings.verbose) info(f"Iteration $iteration - Loss $errorMean%.6g - Loss Vector $errorPerS")
    syncWeights()
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

    val fa = ArrayBuffer.empty[CuMatrix[Float]]

    @tailrec def conv(in: Matrices, i: Int): Unit = {
      val l = _convLayers(i)
      val c = CuMatrix.fromDense(im2col(in, l.field, l.padding, l.stride)._1)
      val p = _cuWeights(i) * c
      val a = _fullLayers(i)._1(p)
      fa += a
      c.release()
      p.release()
      if (i < _lastC) conv(col2im(a.toDense, l.dimOut), i + 1)
    }

    @tailrec def fully(in: CuMatrix[Float], i: Int): Unit = {
      val p = in * _cuWeights(i)
      val a = _fullLayers(i)._1(p)
      fa += a
      p.release()
      if (i < _lastL) fully(a, i + 1)
    }

    conv(in, 0)
    val cuIn = fa(_lastC).reshape(1, _convLayers(_lastC).neurons)
    fully(cuIn, _lastC + 1)

    val o = fa(target).toDense
    fa.foreach(_.release())
    o

  }


  // TODO: move to CUDA.
  private def im2col(ms: Matrices, field: (Int, Int), padding: (Int, Int), stride: (Int, Int), withIndices: Boolean = false): (Matrix, Indices) = {
    val dim = (ms.head.rows, ms.head.cols, ms.size)
    val dimOut = (
      (dim._1 + 2 * padding._1 - field._1) / stride._1 + 1,
      (dim._2 + 2 * padding._2 - field._2) / stride._2 + 1
    )
    val fieldSq = field._1 * field._2
    val out = DenseMatrix.zeros[Float](fieldSq * dim._3, dimOut._1 * dimOut._2)
    val idc = if (withIndices) {
      ms.head.keysIterator.map { k =>
        k -> DenseMatrix.zeros[Int](field._1, field._2)
      }.toMap
    } else null
    _parArrayPool(dim._3).foreach { z =>
      var (w, h, i) = (-padding._1, -padding._2, 0)
      while (w < ((dim._1 + padding._1 - field._1) / stride._1) + 1) {
        while (h < ((dim._2 + padding._2 - field._2) / stride._2) + 1) {
          var (x, y, wi) = (0, 0, 0)
          while (x < field._1) {
            while (y < field._2) {
              val (a, b, c) = (x + (w * stride._1), y + (h * stride._2), z * fieldSq)
              if (a >= 0 && a < dim._1 && b >= 0 && b < dim._2) {
                val value = ms(z)(a, b)
                val lin = c + wi
                out.update(lin, i, value)
                if (withIndices) idc(a, b).update(x, y, i + 1)
              }
              wi += 1
              y += 1
            }
            y = 0
            x += 1
          }
          i += 1
          h += 1
        }
        h = 0
        w += 1
      }
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

    import settings.lossFunction

    val cuys = ys.map(m => CuMatrix.fromDense(m))
    val xsys = xs.zip(cuys)

    val _dws = (0 to _lastWlayerIdx).map { i =>
      i -> CuMatrix.zeros[Float](weights(i).rows, weights(i).cols)
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
        i -> CuMatrix.fromDense(ww)
      case (i, _) => i -> null
    } - _lastC

    val _errSum  = CuMatrix.zeros[Float](1, _outputDim)

    xsys.foreach { xy =>

      val (x, y) = xy
      val fa  = collection.mutable.Map.empty[Int, CuMatrix[Float]]
      val fb  = collection.mutable.Map.empty[Int, CuMatrix[Float]]
      val fc  = collection.mutable.Map.empty[Int, CuMatrix[Float]]
      val ds  = collection.mutable.Map.empty[Int, CuMatrix[Float]]

      @tailrec def conv(_in: Matrices, i: Int): Unit = {
        val l = _convLayers(i)
        val seen = _indices.isDefinedAt(i)
        val (cd, x) = im2col(_in, l.field, l.padding, l.stride, withIndices = !seen)
        val c = CuMatrix.fromDense(cd)
        val p = _cuWeights(i) * c
        var a = _fullLayers(i)._1(p)
        var b = _fullLayers(i)._2(p)
        if (i == _lastC) {
          a = a.reshape(1, l.neurons)
          b = b.reshape(1, l.neurons)
        }
        fa += i -> a
        fb += i -> b
        fc += i -> c
        p.release()
        if (!seen) _indices += i -> x
        if (i < _lastC) conv(col2im(a.toDense, l.dimOut), i + 1)
      }

      @tailrec def fully(_in: CuMatrix[Float], i: Int): Unit = {
        val p  = _in * _cuWeights(i)
        val a  = _fullLayers(i)._1(p)
        val b  = _fullLayers(i)._2(p)
        fa += i -> a
        fb += i -> b
        p.release()
        if (i < _lastL) fully(a, i + 1)
      }

      @tailrec def derive(i: Int): Unit = {
        if (i == _lastWlayerIdx) {
          val (err, grad) = lossFunction(y, fa(i))
          val d = grad *:* fb(i)
          val dw = fa(i - 1).t * d
          _dws(i) += dw
          _errSum += err
          ds += i -> d
          err.release()
          grad.release()
          dw.release()
          derive(i - 1)
        } else if (i < _lastWlayerIdx && i > _lastC) {
          val d1 = ds(i + 1) * _cuWeights(i + 1).t
          val d2 = d1 *:* fb(i)
          val dw = fa(i - 1).t * d2
          _dws(i) += dw
          ds += i -> d2
          d1.release()
          dw.release()
          derive(i - 1)
        } else if (i == _lastC) {
          val l = _convLayers(i)
          val d1 = ds(i + 1) * _cuWeights(i + 1).t
          val d2 = (d1 *:* fb(i)).reshape(l.filters, l.dimOut._1 * l.dimOut._2)
          val dw = d2 * fc(i).t
          _dws(i) += dw
          ds += i -> d2
          d1.release()
          dw.release()
          if (i > 0) derive(i - 1)
        } else {
          val l1 = _convLayers(i + 1)
          val id = _indices(i + 1)
          val de = ds(i + 1).toDense
          val dp = padLeft(de, Dimensions2(de.rows, de.cols + 1), Zero)
          val fs = l1.field._1 * l1.field._2
          val dc = DenseMatrix.zeros[Float](fs * l1.filters, l1.dimIn._1 * l1.dimIn._2)
          // TODO: move to CUDA.
          _parArrayPool(de.rows).foreach { f =>
            val _de = dp(f, ::)
            var (x, y, q) = (0, 0, 0)
            while (x < l1.dimIn._1) {
              while (y < l1.dimIn._2) {
                var p = 0
                id(x, y).foreachPair { (_, v) =>
                  val t = (f * fs + p, q)
                  dc.update(t, _de(v))
                  p += 1
                }
                y += 1
                q += 1
              }
              y = 0
              x += 1
            }
          }
          val cdc = CuMatrix.fromDense(dc)
          val d = _ww(i) * cdc *:* fb(i)
          val dw = d * fc(i).t
          _dws(i) += dw
          ds += i -> d
          cdc.release()
          dw.release()
          if (i > 0) derive(i - 1)
        }
      }

      conv(x, 0)
      fully(fa(_lastC), _lastC + 1)
      derive(_lastWlayerIdx)

      fa.values.foreach(_.release())
      fb.values.foreach(_.release())
      fc.values.foreach(_.release())
      ds.values.foreach(_.release())

    }

    var i = 0
    while (i <= _lastWlayerIdx) {
      settings.updateRule(_cuWeights(i), _dws(i), stepSize, i)
      i += 1
    }

    xsys.foreach(_._2.release())
    _dws.values.foreach(_.release())
    _ww.values.foreach(_.release())
    val es = _errSum.toDense
    _errSum.release()

    es

  }

  private def syncWeights(): Unit = {
    weights.zip(_cuWeights).foreach {
      case (w, cw) => w := cw.toDense
    }
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

    def approximateErrorFuncDerivative(weightLayer: Int, weight: (Int, Int)): Matrix = {
      val Δ = settings.approximation.get.Δ.toFloat
      val v = weights(weightLayer)(weight)
      weights(weightLayer).update(weight, v - Δ)
      syncWeightsBack()
      val a = errorFunc()
      weights(weightLayer).update(weight, v + Δ)
      syncWeightsBack()
      val b = errorFunc()
      weights(weightLayer).update(weight, v)
      syncWeightsBack()
      (b - a) / (2 * Δ)
    }

    def syncWeightsBack(): Unit = {
      weights.zip(_cuWeights).foreach {
        case (w, cw) => cw := w
      }
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

    syncWeightsBack()

    errorFunc()

  }

}

// </editor-fold>
