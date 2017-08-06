package neuroflow.nets

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import neuroflow.core.EarlyStoppingLogic.CanAverage
import neuroflow.core.IllusionBreaker.SettingsNotSupportedException
import neuroflow.core.Network._
import neuroflow.core._

import scala.annotation.tailrec
import scala.collection.Seq
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool


/**
  *
  * This is a standard artificial neural network, using gradient descent,
  * fully connected weights.
  *
  * @author bogdanski
  * @since 15.01.16
  *
  */


object DefaultNetwork {
  implicit val constructor: Constructor[DefaultNetwork] = new Constructor[DefaultNetwork] {
    def apply(ls: Seq[Layer], settings: Settings)(implicit weightProvider: WeightProvider): DefaultNetwork = {
      DefaultNetwork(ls, settings, weightProvider(ls))
    }
  }
}


private[nets] case class DefaultNetwork(layers: Seq[Layer], settings: Settings, weights: Weights,
                                        identifier: String = Registry.register())
  extends FeedForwardNetwork with SupervisedTraining with EarlyStoppingLogic with KeepBestLogic {

  import neuroflow.core.Network._

  private val _layers = layers.map {
    case Cluster(inner) => inner
    case layer: Layer   => layer
  }.toArray

  private val _layersNI = _layers.tail.map { case h: HasActivator[Double] => h }
  private val _outputDim = _layers.last.neurons

  private val _forkJoinTaskSupport = new ForkJoinTaskSupport(new ForkJoinPool(settings.parallelism))

  private val layerSize  = _layers.size - 1
  private val weightLayers = weights.size - 1

  private implicit object Average extends CanAverage[DefaultNetwork] {
    import neuroflow.common.VectorTranslation._
    def averagedError(xs: Seq[Vector], ys: Seq[Vector]): Double = {
      val errors = xs.map(evaluate).zip(ys).toVector.map {
        case (a, b) => mean(abs(a.dv - b.dv))
      }.dv
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
      case _: EarlyStopping | KeepBest =>
      case _ => throw new SettingsNotSupportedException("This regularization is not supported.")
    }
  }

  /**
    * Takes a sequence of input vectors `xs` and trains this
    * network against the corresponding output vectors `ys`.
    */
  def train(xs: Array[Data], ys: Array[Data]): Unit = {
    import settings._
    val in = xs.map(x => DenseMatrix.create[Double](1, x.size, x))
    val out = ys.map(y => DenseMatrix.create[Double](1, y.size, y))
    if (settings.verbose) info(s"Training with ${in.size} samples ...")
    run(in, out, learningRate(0), precision, 1, iterations)
  }

  /**
    * Takes the input vector `x` to compute the output vector.
    */
  def apply(x: Vector): Vector = {
    val input = DenseMatrix.create[Double](1, x.size, x.toArray)
    layers.collect {
      case c: Cluster => c
    }.headOption.map { cl =>
      flow(input, 0, layers.indexOf(cl) - 1).map(cl.inner.activator).toArray.toVector
    }.getOrElse {
      flow(input, 0, layers.size - 1).toArray.toVector
    }
  }

  /**
    * The training loop.
    */
  @tailrec private def run(xs: Matrices, ys: Matrices, stepSize: Double, precision: Double,
                           iteration: Int, maxIterations: Int): Unit = {
    val error = adaptWeights(xs, ys, stepSize)
    val errorMean = mean(error)
    if (settings.verbose) info(f"Iteration $iteration - Mean Error $errorMean%.6g - Error Vector $error")
    maybeGraph(errorMean)
    keepBest(errorMean, weights)
    if (errorMean > precision && iteration < maxIterations && !shouldStopEarly) {
      run(xs, ys, settings.learningRate(iteration + 1), precision, iteration + 1, maxIterations)
    } else {
      if (settings.verbose) info(f"Took $iteration iterations of $maxIterations with Mean Error = $errorMean%.6g")
      takeBest()
    }
  }

  /**
    * Evaluates the error function Σ1/2(y - net(x))² in parallel.
    */
  private def errorFunc(xs: Matrices, ys: Matrices): Matrix = {
    val xsys = xs.zip(ys).par
    xsys.tasksupport = _forkJoinTaskSupport
    xsys.map { xy => 0.5 * pow(xy._1 - flow(xy._2, 0, layerSize), 2) }.reduce(_ + _)
  }

    /**
    * Computes the network recursively from `cursor` until `target`.
    */
  @tailrec private def flow(in: Matrix, cursor: Int, target: Int): Matrix = {
    if (target < 0) in
    else {
      val processed = _layers(cursor) match {
        case h: HasActivator[Double] =>
          if (cursor <= weightLayers) in.map(h.activator) * weights(cursor)
          else in.map(h.activator)
        case _ => in * weights(cursor)
      }
      if (cursor < target) flow(processed, cursor + 1, target) else processed
    }
  }

  /**
    * Computes gradient for all weights in parallel,
    * adapts their value using gradient descent and returns the error matrix.
    */
  private def adaptWeights(xs: Matrices, ys: Matrices, stepSize: Double): Matrix =
    if (settings.approximation.isDefined) {
      weights.zipWithIndex.foreach { 
        case (l, idx) =>
          l.foreachPair { (k, v) =>
            val grad = approximateErrorFuncDerivative(xs, ys, idx, k)
            l.update(k, v - stepSize * sum(grad))
          }
      }
      errorFunc(xs, ys)
    } else {
      val xsys = xs.par.zip(ys)
      xsys.tasksupport = _forkJoinTaskSupport

      val _ds = (0 to weightLayers).map { i =>
        i -> DenseMatrix.zeros[Double](weights(i).rows, weights(i).cols)
      }.toMap

      val _errSum = DenseMatrix.zeros[Double](1, _outputDim)
      val _square = DenseMatrix.zeros[Double](1, _outputDim)
      _square := 2.0

      xsys.map { xy =>
        val (x, y) = xy
        val ps  = collection.mutable.Map.empty[Int, Matrix]
        val fa  = collection.mutable.Map.empty[Int, Matrix]
        val fb  = collection.mutable.Map.empty[Int, Matrix]
        val dws = collection.mutable.Map.empty[Int, Matrix]
        val ds  = collection.mutable.Map.empty[Int, Matrix]
        val e   = DenseMatrix.zeros[Double](1, _outputDim)

        @tailrec def forward(in: Matrix, i: Int): Unit = {
          val p = in * weights(i)
          val a = p.map(_layersNI(i).activator)
          val b = p.map(_layersNI(i).activator.derivative)
          ps += i -> p
          fa += i -> a
          fb += i -> b
          if (i < weightLayers) forward(a, i + 1)
        }

        @tailrec def derive(i: Int): Unit = {
          if (i == 0 && weightLayers == 0) {
            val yf = y - fa(0)
            val d = -yf *:* fb(0)
            val dw = x.t * d
            dws += 0 -> dw
            e += yf
          } else if (i == weightLayers) {
            val yf = y - fa(i)
            val d = -yf *:* fb(i)
            val dw = fa(i - 1).t * d
            dws += i -> dw
            ds += i -> d
            e += yf
            derive(i - 1)
          } else if (i < weightLayers && i > 0) {
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
        derive(weightLayers)
        e :^= _square
        e *= 0.5
        (dws, e)
      }.seq.foreach { ab =>
        _errSum += ab._2
        var i = 0
        while (i <= weightLayers) {
          val m = _ds(i)
          val n = ab._1(i)
          m += (n *= stepSize)
          i += 1
        }
      }
      var i = 0
      while (i <= weightLayers) {
        weights(i) -= _ds(i)
        i += 1
      }
      _errSum
    }

  /**
    * Approximates the gradient based on finite central differences.
    */
  private def approximateErrorFuncDerivative(xs: Matrices, ys: Matrices,
                              weightLayer: Int, weight: (Int, Int)): Matrix = {
    val Δ = settings.approximation.get.Δ
    val v = weights(weightLayer)(weight)
    weights(weightLayer).update(weight, v - Δ)
    val a = errorFunc(xs, ys)
    weights(weightLayer).update(weight, v + Δ)
    val b = errorFunc(xs, ys)
    (b - a) / (2 * Δ)
  }

}
