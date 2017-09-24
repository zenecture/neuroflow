package neuroflow.nets.gpu

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import jcuda.jcublas.{JCublas2, cublasHandle}
import neuroflow.core.EarlyStoppingLogic.CanAverage
import neuroflow.core.IllusionBreaker.SettingsNotSupportedException
import neuroflow.core.Network._
import neuroflow.core._
import neuroflow.nets.Registry
import neuroflow.nets.gpu.cuda.CuMatrix

import scala.annotation.tailrec
import scala.collection.Seq
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool


/**
  *
  * This is a feed-forward neural network with fully connected layers running on CUDA.
  * It uses gradient descent to optimize the error function Σ1/2(y - net(x))².
  *
  * Use the parallelism parameter with care, as it greatly affects memory usage.
  *
  * @author bogdanski
  * @since 15.01.16
  *
  */


object DenseNetwork {
  
  implicit val double: Constructor[Double, DenseNetworkDouble] = new Constructor[Double, DenseNetworkDouble] {
    def apply(ls: Seq[Layer], settings: Settings[Double])(implicit weightProvider: WeightProvider[Double]): DenseNetworkDouble = {
      DenseNetworkDouble(ls, settings, weightProvider(ls))
    }
  }

  implicit val single: Constructor[Float, DenseNetworkSingle] = new Constructor[Float, DenseNetworkSingle] {
    def apply(ls: Seq[Layer], settings: Settings[Float])(implicit weightProvider: WeightProvider[Float]): DenseNetworkSingle = {
      DenseNetworkSingle(ls, settings, weightProvider(ls))
    }
  }
  
}

//// Double Precision Impl

private[nets] case class DenseNetworkDouble(layers: Seq[Layer], settings: Settings[Double], weights: Weights[Double],
                                      identifier: String = Registry.register())
  extends FFN[Double] with EarlyStoppingLogic[Double] with KeepBestLogic[Double] with WaypointLogic[Double] {

  implicit val handle = new cublasHandle
  JCublas2.cublasCreate(handle)

  type Vector   = Network.Vector[Double]
  type Vectors  = Network.Vectors[Double]
  type Matrix   = Network.Matrix[Double]
  type Matrices = Network.Matrices[Double]

  private val _layers = layers.map {
    case Focus(inner) => inner
    case layer: Layer => layer
  }.toArray

  private val _clusterLayer   = layers.collect { case c: Focus => c }.headOption

  private val _layersNI       = _layers.tail.map { case h: HasActivator[Double] => h }
  private val _outputDim      = _layers.last.neurons
  private val _lastWlayerIdx  = weights.size - 1
  private val _cuWeights      = weights.map(m => CuMatrix.fromDense(m))

  private val _forkJoinTaskSupport = new ForkJoinTaskSupport(new ForkJoinPool(settings.parallelism))

  private implicit object Average extends CanAverage[Double, DenseNetworkDouble, Vector, Vector] {
    def averagedError(xs: Vectors, ys: Vectors): Double = {
      val errors = xs.map(evaluate).zip(ys).map {
        case (a, b) => mean(abs(a - b))
      }
      mean(errors)
    }
  }

  /**
    * Checks if the [[Settings]] are properly defined.
    * Might throw a [[SettingsNotSupportedException]].
    */
  override def checkSettings(): Unit = {
    super.checkSettings()
    if (settings.specifics.isDefined)
      warn("No specific settings supported. This has no effect.")
    settings.regularization.foreach {
      case _: EarlyStopping[_, _] | KeepBest =>
      case _ => throw new SettingsNotSupportedException("This regularization is not supported.")
    }
  }

  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Vectors, ys: Vectors): Unit = {
    require(xs.size == ys.size, "Mismatch between sample sizes!")
    import settings._
    val batchSize = settings.batchSize.getOrElse(xs.size)
    if (settings.verbose) info(s"Training with ${xs.size} samples, batchize = $batchSize ...")
    val xsys = xs.map(_.asDenseMatrix).zip(ys.map(_.asDenseMatrix)).grouped(batchSize).toSeq
    run(xsys, learningRate(0 -> 1.0), xs.size, batchSize, precision, 1, iterations)
  }

  /**
    * Computes output for `x`.
    */
  def apply(x: Vector): Vector = {
    val input = DenseMatrix.create[Double](1, x.size, x.toArray)
    _clusterLayer.map { cl =>
      flow(input, layers.indexOf(cl) - 1).toDenseVector
    }.getOrElse {
      flow(input, _lastWlayerIdx).toDenseVector
    }
  }

  /**
    * The training loop.
    */
  @tailrec private def run(xsys: Seq[Seq[(Matrix, Matrix)]], stepSize: Double, sampleSize: Int, batchSize: Int, precision: Double,
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
    if (errorMean > precision && iteration < maxIterations && !shouldStopEarly) {
      run(xsys, settings.learningRate(iteration + 1 -> stepSize), sampleSize, batchSize, precision, iteration + 1, maxIterations)
    } else {
      if (settings.verbose) info(f"Took $iteration iterations of $maxIterations with Mean Error = $errorMean%.6g")
      weights.zip(_cuWeights).foreach {
        case (w, cw) => w := cw.toDense
      }
      takeBest()
    }
  }

  /**
    * Computes the network recursively.
    */
  private def flow(in: Matrix, outLayer: Int): Matrix = {
    val _in = CuMatrix.fromDense(in)
    val fa  = collection.mutable.Map.empty[Int, CuMatrix[Double]]
    @tailrec def forward(in: CuMatrix[Double], i: Int): Unit = {
      val p  = in * _cuWeights(i)
      val pd = p.toDense
      p.release()
      val ad = pd.map(_layersNI(i).activator)
      val a  = CuMatrix.fromDense(ad)
      fa += i -> a
      if (i < outLayer) forward(a, i + 1)
    }
    forward(_in, 0)
    val o = fa(outLayer).toDense
    _in.release()
    fa.values.foreach(_.release())
    o
  }

  /**
    * Computes gradient for all weights in parallel,
    * adapts their value using gradient descent and returns the error matrix.
    */
  private def adaptWeights(xs: Matrices, ys: Matrices, stepSize: Double): Matrix = {
    val cuxs = xs.map(m => CuMatrix.fromDense(m))
    val cuys = ys.map(m => CuMatrix.fromDense(m))
    val xsys = cuxs.par.zip(cuys)
    xsys.tasksupport = _forkJoinTaskSupport

    val _ds = (0 to _lastWlayerIdx).map { i =>
      i -> CuMatrix.zeros[Double](weights(i).rows, weights(i).cols)
    }.toMap

    val _errSum = CuMatrix.zeros[Double](1, _outputDim)
    val _square = CuMatrix.zeros[Double](1, _outputDim)
    _square := 2.0

    xsys.map { xy =>

      val (x, y) = xy
      val fa  = collection.mutable.Map.empty[Int, CuMatrix[Double]]
      val fb  = collection.mutable.Map.empty[Int, CuMatrix[Double]]
      val dws = collection.mutable.Map.empty[Int, CuMatrix[Double]]
      val ds  = collection.mutable.Map.empty[Int, CuMatrix[Double]]
      val e   = CuMatrix.zeros[Double](1, _outputDim)

      @tailrec def forward(in: CuMatrix[Double], i: Int): Unit = {
        val p = in * _cuWeights(i)
        val pd = p.toDense // activator can be arbitrary, CPU.
        p.release()
        val ad = pd.map(_layersNI(i).activator)
        val bd = pd.map(_layersNI(i).activator.derivative)
        val a = CuMatrix.fromDense(ad)
        val b = CuMatrix.fromDense(bd)
        fa += i -> a
        fb += i -> b
        if (i < _lastWlayerIdx) forward(a, i + 1)
      }

      @tailrec def derive(i: Int): Unit = {
        if (i == 0 && _lastWlayerIdx == 0) {
          val yf = y - fa(0)
          val nyf = -yf
          val d = nyf *:* fb(0)
          val dw = x.t * d
          dws += 0 -> dw
          e += yf
          nyf.release()
          yf.release()
          d.release()
        } else if (i == _lastWlayerIdx) {
          val yf = y - fa(i)
          val nyf = -yf
          val d = nyf *:* fb(i)
          val dw = fa(i - 1).t * d
          dws += i -> dw
          ds += i -> d
          e += yf
          nyf.release()
          yf.release()
          derive(i - 1)
        } else if (i < _lastWlayerIdx && i > 0) {
          val d1 = ds(i + 1) * _cuWeights(i + 1).t
          val d2 = d1 *:* fb(i)
          val dw = fa(i - 1).t * d2
          dws += i -> dw
          ds += i -> d2
          d1.release()
          derive(i - 1)
        } else if (i == 0) {
          val d1 = ds(i + 1) * _cuWeights(i + 1).t
          val d2 = d1 *:* fb(i)
          val dw = x.t * d2
          dws += i -> dw
          d1.release()
        }
      }

      forward(x, 0)
      derive(_lastWlayerIdx)
      e :^= _square
      e *= 0.5

      ds.values.foreach(_.release())
      fa.values.foreach(_.release())
      fb.values.foreach(_.release())

      (dws, e)

    }.seq.foreach { ab =>
      _errSum += ab._2
      ab._2.release()
      var i = 0
      while (i <= _lastWlayerIdx) {
        val m = _ds(i)
        val n = ab._1(i)
        m += n
        i += 1
        n.release()
      }
    }

    var i = 0
    while (i <= _lastWlayerIdx) {
      settings.updateRule(_cuWeights(i), _ds(i), stepSize, i)
      i += 1
    }

    xsys.foreach { xy =>
      xy._1.release()
      xy._2.release()
    }

    _ds.values.foreach(_.release())
    val es = _errSum.toDense
    _errSum.release()
    _square.release()

    es
  }

  /** Approximates the gradient based on finite central differences. (For debugging) */
  private def adaptWeightsApprox(xs: Matrices, ys: Matrices, stepSize: Double): Matrix = {

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
      weights.zip(_cuWeights).foreach {case (w, cw) => cw := w }
      val a = errorFunc()
      weights(weightLayer).update(weight, v + Δ)
      weights.zip(_cuWeights).foreach {case (w, cw) => cw := w }
      val b = errorFunc()
      weights(weightLayer).update(weight, v)
      weights.zip(_cuWeights).foreach {case (w, cw) => cw := w }
      (b - a) / (2 * Δ)
    }

    val updates = collection.mutable.HashMap.empty[(Int, (Int, Int)), Double]
    val grads   = collection.mutable.HashMap.empty[(Int, (Int, Int)), Double]
    val debug   = collection.mutable.HashMap.empty[Int, Matrix]

    weights.zipWithIndex.foreach {
      case (l, idx) =>
        debug += idx -> l.copy
        l.foreachPair { (k, v) =>
          val efd  = approximateErrorFuncDerivative(idx, k)
          val grad = sum(efd)
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

//// Single Precision Impl

private[nets] case class DenseNetworkSingle(layers: Seq[Layer], settings: Settings[Float], weights: Weights[Float],
                                            identifier: String = Registry.register())
  extends FFN[Float] with EarlyStoppingLogic[Float] with KeepBestLogic[Float] with WaypointLogic[Float] {

  implicit val handle = new cublasHandle
  JCublas2.cublasCreate(handle)

  type Vector   = Network.Vector[Float]
  type Vectors  = Network.Vectors[Float]
  type Matrix   = Network.Matrix[Float]
  type Matrices = Network.Matrices[Float]

  private val _layers = layers.map {
    case Focus(inner) => inner
    case layer: Layer => layer
  }.toArray

  private val _clusterLayer   = layers.collect { case c: Focus => c }.headOption

  private val _activators     = _layers.tail.map { case h: HasActivator[Double] => h.activator.map[Float](_.toDouble, _.toFloat) }
  private val _outputDim      = _layers.last.neurons
  private val _lastWlayerIdx  = weights.size - 1
  private val _cuWeights      = weights.map(m => CuMatrix.fromDense(m))

  private val _forkJoinTaskSupport = new ForkJoinTaskSupport(new ForkJoinPool(settings.parallelism))

  private implicit object Average extends CanAverage[Float, DenseNetworkSingle, Vector, Vector] {
    def averagedError(xs: Vectors, ys: Vectors): Double = {
      val errors = xs.map(evaluate).zip(ys).map {
        case (a, b) => mean(abs(a - b))
      }
      mean(errors)
    }
  }

  /**
    * Checks if the [[Settings]] are properly defined.
    * Might throw a [[SettingsNotSupportedException]].
    */
  override def checkSettings(): Unit = {
    super.checkSettings()
    if (settings.specifics.isDefined)
      warn("No specific settings supported. This has no effect.")
    settings.regularization.foreach {
      case _: EarlyStopping[_, _] | KeepBest =>
      case _ => throw new SettingsNotSupportedException("This regularization is not supported.")
    }
  }

  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Vectors, ys: Vectors): Unit = {
    require(xs.size == ys.size, "Mismatch between sample sizes!")
    import settings._
    val batchSize = settings.batchSize.getOrElse(xs.size)
    if (settings.verbose) info(s"Training with ${xs.size} samples, batchize = $batchSize ...")
    val xsys = xs.map(_.asDenseMatrix).zip(ys.map(_.asDenseMatrix)).grouped(batchSize).toSeq
    run(xsys, learningRate(0 -> 1.0).toFloat, xs.size, batchSize, precision, 1, iterations)
  }

  /**
    * Computes output for `x`.
    */
  def apply(x: Vector): Vector = {
    val input = DenseMatrix.create[Float](1, x.size, x.toArray)
    _clusterLayer.map { cl =>
      flow(input, layers.indexOf(cl) - 1).toDenseVector
    }.getOrElse {
      flow(input, _lastWlayerIdx).toDenseVector
    }
  }

  /**
    * The training loop.
    */
  @tailrec private def run(xsys: Seq[Seq[(Matrix, Matrix)]], stepSize: Float, sampleSize: Int, batchSize: Int, precision: Double,
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
    if (errorMean > precision && iteration < maxIterations && !shouldStopEarly) {
      run(xsys, settings.learningRate(iteration + 1 -> stepSize).toFloat, sampleSize, batchSize, precision, iteration + 1, maxIterations)
    } else {
      if (settings.verbose) info(f"Took $iteration iterations of $maxIterations with Mean Error = $errorMean%.6g")
      weights.zip(_cuWeights).foreach {
        case (w, cw) => w := cw.toDense
      }
      takeBest()
    }
  }

  /**
    * Computes the network recursively.
    */
  private def flow(in: Matrix, outLayer: Int): Matrix = {
    val _in = CuMatrix.fromDense(in)
    val fa  = collection.mutable.Map.empty[Int, CuMatrix[Float]]
    @tailrec def forward(in: CuMatrix[Float], i: Int): Unit = {
      val p  = in * _cuWeights(i)
      val pd = p.toDense
      p.release()
      val ad = pd.map(_activators(i))
      val a  = CuMatrix.fromDense(ad)
      fa += i -> a
      if (i < outLayer) forward(a, i + 1)
    }
    forward(_in, 0)
    val o = fa(outLayer).toDense
    _in.release()
    fa.values.foreach(_.release())
    o
  }

  /**
    * Computes gradient for all weights in parallel,
    * adapts their value using gradient descent and returns the error matrix.
    */
  private def adaptWeights(xs: Matrices, ys: Matrices, stepSize: Float): Matrix = {
    val cuxs = xs.map(m => CuMatrix.fromDense(m))
    val cuys = ys.map(m => CuMatrix.fromDense(m))
    val xsys = cuxs.par.zip(cuys)
    xsys.tasksupport = _forkJoinTaskSupport

    val _ds = (0 to _lastWlayerIdx).map { i =>
      i -> CuMatrix.zeros[Float](weights(i).rows, weights(i).cols)
    }.toMap

    val _errSum = CuMatrix.zeros[Float](1, _outputDim)
    val _square = CuMatrix.zeros[Float](1, _outputDim)
    _square := 2.0f

    xsys.map { xy =>

      val (x, y) = xy
      val fa  = collection.mutable.Map.empty[Int, CuMatrix[Float]]
      val fb  = collection.mutable.Map.empty[Int, CuMatrix[Float]]
      val dws = collection.mutable.Map.empty[Int, CuMatrix[Float]]
      val ds  = collection.mutable.Map.empty[Int, CuMatrix[Float]]
      val e   = CuMatrix.zeros[Float](1, _outputDim)

      @tailrec def forward(in: CuMatrix[Float], i: Int): Unit = {
        val p = in * _cuWeights(i)
        val pd = p.toDense // activator can be arbitrary, CPU.
        p.release()
        val ad = pd.map(_activators(i))
        val bd = pd.map(_activators(i).derivative)
        val a = CuMatrix.fromDense(ad)
        val b = CuMatrix.fromDense(bd)
        fa += i -> a
        fb += i -> b
        if (i < _lastWlayerIdx) forward(a, i + 1)
      }

      @tailrec def derive(i: Int): Unit = {
        if (i == 0 && _lastWlayerIdx == 0) {
          val yf = y - fa(0)
          val nyf = -yf
          val d = nyf *:* fb(0)
          val dw = x.t * d
          dws += 0 -> dw
          e += yf
          nyf.release()
          yf.release()
          d.release()
        } else if (i == _lastWlayerIdx) {
          val yf = y - fa(i)
          val nyf = -yf
          val d = nyf *:* fb(i)
          val dw = fa(i - 1).t * d
          dws += i -> dw
          ds += i -> d
          e += yf
          nyf.release()
          yf.release()
          derive(i - 1)
        } else if (i < _lastWlayerIdx && i > 0) {
          val d1 = ds(i + 1) * _cuWeights(i + 1).t
          val d2 = d1 *:* fb(i)
          val dw = fa(i - 1).t * d2
          dws += i -> dw
          ds += i -> d2
          d1.release()
          derive(i - 1)
        } else if (i == 0) {
          val d1 = ds(i + 1) * _cuWeights(i + 1).t
          val d2 = d1 *:* fb(i)
          val dw = x.t * d2
          dws += i -> dw
          d1.release()
        }
      }

      forward(x, 0)
      derive(_lastWlayerIdx)
      e :^= _square
      e *= 0.5f

      ds.values.foreach(_.release())
      fa.values.foreach(_.release())
      fb.values.foreach(_.release())

      (dws, e)

    }.seq.foreach { ab =>
      _errSum += ab._2
      ab._2.release()
      var i = 0
      while (i <= _lastWlayerIdx) {
        val m = _ds(i)
        val n = ab._1(i)
        m += n
        i += 1
        n.release()
      }
    }

    var i = 0
    while (i <= _lastWlayerIdx) {
      settings.updateRule(_cuWeights(i), _ds(i), stepSize, i)
      i += 1
    }

    xsys.foreach { xy =>
      xy._1.release()
      xy._2.release()
    }

    _ds.values.foreach(_.release())
    val es = _errSum.toDense
    _errSum.release()
    _square.release()

    es
  }

  /** Approximates the gradient based on finite central differences. (For debugging) */
  private def adaptWeightsApprox(xs: Matrices, ys: Matrices, stepSize: Float): Matrix = {

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