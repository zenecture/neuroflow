package neuroflow.nets

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import neuroflow.common.Registry
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

  private val _forkJoinTaskSupport = new ForkJoinTaskSupport(new ForkJoinPool(settings.parallelism))

  private val layerSize  = _layers.size - 1
  private val weightLayerSize = weights.size - 1

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
  def train(xs: Seq[Vector], ys: Seq[Vector]): Unit = {
    import settings._
    val in = xs.map(x => DenseMatrix.create[Double](1, x.size, x.toArray)).toVector
    val out = ys.map(y => DenseMatrix.create[Double](1, y.size, y.toArray)).toVector
    run(in, out, learningRate(0), precision, 0, iterations)
  }

  /**
    * Takes the input vector `x` to compute the output vector.
    */
  def evaluate(x: Vector): Vector = {
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
    val error = errorFunc(xs, ys)
    val errorMean = mean(error)
    if (errorMean > precision && iteration < maxIterations && !shouldStopEarly) {
      if (settings.verbose) info(f"Taking step $iteration - Mean Error $errorMean%.6g - Error Vector $error")
      maybeGraph(errorMean)
      adaptWeights(xs, ys, stepSize)
      keepBest(errorMean, weights)
      run(xs, ys, settings.learningRate(iteration + 1), precision, iteration + 1, maxIterations)
    } else {
      if (settings.verbose) info(f"Took $iteration iterations of $maxIterations with Mean Error = $errorMean%.6g")
      takeBest()
    }
  }

  /**
    * Evaluates the error function Σ1/2(target - prediction(x))² in parallel.
    */
  private def errorFunc(xs: Matrices, ys: Matrices): Matrix = {
    xs.zip(ys).par.map {
      case (x, y) =>
        0.5 * pow(y - flow(x, 0, layerSize), 2)
    }.reduce(_ + _)
  }

    /**
    * Computes the network recursively from `cursor` until `target`.
    */
  @tailrec private def flow(in: Matrix, cursor: Int, target: Int): Matrix = {
    if (target < 0) in
    else {
      val processed = _layers(cursor) match {
        case h: HasActivator[Double] =>
          if (cursor <= weightLayerSize) in.map(h.activator) * weights(cursor)
          else in.map(h.activator)
        case _ => in * weights(cursor)
      }
      if (cursor < target) flow(processed, cursor + 1, target) else processed
    }
  }

  /**
    * Computes gradient for all weights in parallel,
    * and adapts their value using gradient descent.
    */
  private def adaptWeights(xs: Matrices, ys: Matrices, stepSize: Double): Unit = 
    if (settings.approximation.isDefined) {
      weights.zipWithIndex.foreach { case (l, idx) =>
        l.foreachPair { (k, v) =>
          val grad = approximateErrorFuncDerivative(xs, ys, idx, k)
          l.update(k, v - stepSize * sum(grad))
        }
      }
    } else {
      val xsys = xs.par.zip(ys)
      xsys.tasksupport = _forkJoinTaskSupport
      val derivatives = xsys.map {
        case (x, y) =>
          val  ps = collection.mutable.Map.empty[Int, Matrix]
          val  fa = collection.mutable.Map.empty[Int, Matrix]
          val  fb = collection.mutable.Map.empty[Int, Matrix]
          val dws = collection.mutable.Map.empty[Int, Matrix]
          val  ds = collection.mutable.Map.empty[Int, Matrix]

          @tailrec def forward(in: Matrix, i: Int): Unit = {
            val p = in * weights(i)
            val a = p.map(_layersNI(i).activator)
            val b = p.map(_layersNI(i).activator.derivative)
            ps += i -> p
            fa += i -> a
            fb += i -> b
            if (i < weightLayerSize) forward(a, i + 1)
          }

          @tailrec def derive(j: Int): Unit = j match {
            case i if i == 0 && weightLayerSize == 0 =>
              val d = -(y - fa(0)) *:* fb(0)
              val dw = x.t * d
              dws += 0 -> dw
            case i if i == weightLayerSize =>
              val d = -(y - fa(i)) *:* fb(i)
              val dw = fa(i - 1).t * d
              dws += i -> dw
              ds += i -> d
              derive(i - 1)
            case i if i < weightLayerSize && i > 0 =>
              val d = (ds(i + 1) * weights(i + 1).t) *:* fb(i)
              val dw = fa(i - 1).t * d
              dws += i -> dw
              ds += i -> d
              derive(i - 1)
            case i if i == 0 =>
              val d = (ds(i + 1) * weights(i + 1).t) *:* fb(i)
              val dw = x.t * d
              dws += i -> dw
          }

          forward(x, 0)
          derive(weightLayerSize)
          dws
      }.reduce { (a, b) =>
        val ds = collection.mutable.Map.empty[Int, Matrix]
        (0 to weightLayerSize).map { i =>
          ds += i -> (a(i) + b(i)).map(_ * stepSize)
        }
        ds
      }

      (0 to weightLayerSize).foreach { i =>
        val n = weights(i) - derivatives(i)
        weights(i).foreachPair { (k, v) =>
          weights(i).update(k, n(k))
        }
      }
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
