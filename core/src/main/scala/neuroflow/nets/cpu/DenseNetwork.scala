package neuroflow.nets.cpu

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import neuroflow.core.EarlyStoppingLogic.CanAverage
import neuroflow.core.IllusionBreaker.SettingsNotSupportedException
import neuroflow.core.Network._
import neuroflow.core._
import neuroflow.nets.Registry

import scala.annotation.tailrec
import scala.collection.Seq
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool


/**
  *
  * This is a feed-forward neural network with fully connected layers.
  * It uses gradient descent to optimize the error function Σ1/2(y - net(x))².
  *
  * Use the parallelism parameter with care, as it greatly affects memory usage.
  *
  * @author bogdanski
  * @since 15.01.16
  *
  */


object DenseNetwork {
  implicit val constructor: Constructor[DenseNetwork] = new Constructor[DenseNetwork] {
    def apply(ls: Seq[Layer], settings: Settings)(implicit weightProvider: WeightProvider): DenseNetwork = {
      DenseNetwork(ls, settings, weightProvider(ls))
    }
  }
}


private[nets] case class DenseNetwork(layers: Seq[Layer], settings: Settings, weights: Weights,
                                      identifier: String = Registry.register())
  extends FeedForwardNetwork with EarlyStoppingLogic with KeepBestLogic with WaypointLogic {

  import neuroflow.core.Network._

  private val _layers = layers.map {
    case Focus(inner) => inner
    case layer: Layer   => layer
  }.toArray

  private val _clusterLayer   = layers.collect { case c: Focus => c }.headOption

  private val _layersNI       = _layers.tail.map { case h: HasActivator[Double] => h }
  private val _outputDim      = _layers.last.neurons
  private val _lastWlayerIdx  = weights.size - 1

  private val _forkJoinTaskSupport = new ForkJoinTaskSupport(new ForkJoinPool(settings.parallelism))

  private implicit object Average extends CanAverage[DenseNetwork, Vector, Vector] {
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
      takeBest()
    }
  }

  /**
    * Computes the network recursively.
    */
  private def flow(in: Matrix, outLayer: Int): Matrix = {
    val fa  = collection.mutable.Map.empty[Int, Matrix]
    @tailrec def forward(in: Matrix, i: Int): Unit = {
      val p = in * weights(i)
      val a = p.map(_layersNI(i).activator)
      fa += i -> a
      if (i < outLayer) forward(a, i + 1)
    }
    forward(in, 0)
    fa(outLayer)
  }

  /**
    * Computes gradient for all weights in parallel,
    * adapts their value using gradient descent and returns the error matrix.
    */
  private def adaptWeights(xs: Matrices, ys: Matrices, stepSize: Double): Matrix = {
    val xsys = xs.par.zip(ys)
    xsys.tasksupport = _forkJoinTaskSupport

    val _ds = (0 to _lastWlayerIdx).map { i =>
      i -> DenseMatrix.zeros[Double](weights(i).rows, weights(i).cols)
    }.toMap

    val _errSum = DenseMatrix.zeros[Double](1, _outputDim)
    val _square = DenseMatrix.zeros[Double](1, _outputDim)
    _square := 2.0

    xsys.map { xy =>
      val (x, y) = xy
      val fa  = collection.mutable.Map.empty[Int, Matrix]
      val fb  = collection.mutable.Map.empty[Int, Matrix]
      val dws = collection.mutable.Map.empty[Int, Matrix]
      val ds  = collection.mutable.Map.empty[Int, Matrix]
      val e   = DenseMatrix.zeros[Double](1, _outputDim)

      @tailrec def forward(in: Matrix, i: Int): Unit = {
        val p = in * weights(i)
        val a = p.map(_layersNI(i).activator)
        val b = p.map(_layersNI(i).activator.derivative)
        fa += i -> a
        fb += i -> b
        if (i < _lastWlayerIdx) forward(a, i + 1)
      }

      @tailrec def derive(i: Int): Unit = {
        if (i == 0 && _lastWlayerIdx == 0) {
          val yf = y - fa(0)
          val d = -yf *:* fb(0)
          val dw = x.t * d
          dws += 0 -> dw
          e += yf
        } else if (i == _lastWlayerIdx) {
          val yf = y - fa(i)
          val d = -yf *:* fb(i)
          val dw = fa(i - 1).t * d
          dws += i -> dw
          ds += i -> d
          e += yf
          derive(i - 1)
        } else if (i < _lastWlayerIdx && i > 0) {
          val d = (ds(i + 1) * weights(i + 1).t) *:* fb(i)
          val dw = fa(i - 1).t * d
          dws += i -> dw
          ds += i -> d
          derive(i - 1)
        } else if (i == 0) {
          val d = (ds(i + 1) * weights(i + 1).t) *:* fb(i)
          val dw = x.t * d
          dws += i -> dw
        }
      }

      forward(x, 0)
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
  private def adaptWeightsApprox(xs: Matrices, ys: Matrices, stepSize: Double): Matrix = {

    require(settings.updateRule.isInstanceOf[Debuggable])
    val _rule: Debuggable = settings.updateRule.asInstanceOf[Debuggable]

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
