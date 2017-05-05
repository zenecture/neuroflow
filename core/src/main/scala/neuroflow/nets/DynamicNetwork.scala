package neuroflow.nets

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import neuroflow.core.EarlyStoppingLogic.CanAverage
import neuroflow.core._
import neuroflow.core.Network._

import scala.annotation.tailrec
import scala.collection.Seq
import scala.util.Random

/**
  *
  * Same as DefaultNetwork, but it uses the Armijo–Goldstein condition
  * to adapt the learning rate to an optimal value. This promises faster convergence,
  * but comes with more computational overhead than DefaultNetwork.
  *
  * Here, the learning parameter should be a large starting value, e.g. 10.
  *
  * @author bogdanski
  * @since 20.01.16
  *
  */


object DynamicNetwork {
  implicit val constructor: Constructor[DynamicNetwork] = new Constructor[DynamicNetwork] {
    def apply(ls: Seq[Layer], settings: Settings)(implicit weightProvider: WeightProvider): DynamicNetwork = {
      DynamicNetwork(ls, settings, weightProvider(ls))
    }
  }
}


private[nets] case class DynamicNetwork(layers: Seq[Layer], settings: Settings, weights: Weights,
                                        identifier: String = Random.alphanumeric.take(3).mkString)
  extends FeedForwardNetwork with SupervisedTraining with EarlyStoppingLogic with KeepBestLogic {

  import neuroflow.core.Network._

  private val _layers = layers.map {
    case Cluster(inner) => inner
    case layer: Layer   => layer
  }.toArray

  private val fastLayersSize1  = _layers.size - 1
  private val fastWeightsSize1 = weights.size - 1

  private implicit object Average extends CanAverage[DynamicNetwork] {
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
    run(in, out, learningRate, precision, 0, iterations)
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
      info("Couldn't find Cluster Layer. Using Output Layer.")
      flow(input, 0, layers.size - 1).toArray.toVector
    }
  }

  /**
    * The eval loop.
    */
  @tailrec private def run(xs: Matrices, ys: Matrices, stepSize: Double, precision: Double,
                           iteration: Int, maxIterations: Int): Unit = {
    val error = errorFunc(xs, ys)
    val errorMean = mean(error)
    if (errorMean > precision && iteration < maxIterations && !shouldStopEarly) {
      if (settings.verbose) info(f"Taking step $iteration - Mean Error $errorMean%.6g - Error Vector $error")
      maybeGraph(errorMean)
      adaptWeights(xs, ys, stepSize)
      update(errorMean, weights)
      run(xs, ys, stepSize, precision, iteration + 1, maxIterations)
    } else {
      if (settings.verbose) info(f"Took $iteration iterations of $maxIterations with Mean Error = $errorMean%.6g")
      take()
    }
  }

  /**
    * Evaluates the error function Σ1/2(prediction(x) - observation)² in parallel.
    */
  private def errorFunc(xs: Matrices, ys: Matrices): Matrix = {
    xs.zip(ys).par.map {
      case (x, y) =>
        0.5 * pow(flow(x, 0, fastLayersSize1) - y, 2)
    }.reduce(_ + _)
  }

  /**
    * Computes gradient via `errorFuncDerivative` for all weights,
    * and adapts their value using gradient descent.
    */
  private def adaptWeights(xs: Matrices, ys: Matrices, stepSize: Double): Unit = {
    weights.foreach { l =>
      l.foreachPair { (k, v) =>
        val weightLayer = weights.indexOf(l)
        val firstOrder =
          if (settings.approximation.isDefined) approximateErrorFuncDerivative(xs, ys, weightLayer, k)
          else errorFuncDerivative(xs, ys, weightLayer, k)
        val direction = mean(-firstOrder)
        l.update(k, v + α(stepSize, direction, xs, ys, weightLayer, k) * direction)
      }
    }
  }

  /**
    * Computes the network recursively from `cursor` until `target`.
    */
  @tailrec final protected def flow(in: Matrix, cursor: Int, target: Int): Matrix = {
    if (target < 0) in
    else {
      val processed = _layers(cursor) match {
        case h: HasActivator[Double] =>
          if (cursor <= fastWeightsSize1) in.map(h.activator) * weights(cursor)
          else in.map(h.activator)
        case _ => in * weights(cursor)
      }
      if (cursor < target) flow(processed, cursor + 1, target) else processed
    }
  }

  /**
    * Computes the error function derivative with respect to `weight` in `weightLayer`.
    */
  private def errorFuncDerivative(xs: Matrices, ys: Matrices,
                              weightLayer: Int, weight: (Int, Int)): Matrix = {
    xs.zip(ys).map {
      case (x, y) =>
        val ws = weights.map(_.copy)
        ws(weightLayer).update(weight, 1.0)
        ws(weightLayer).foreachKey(k => if (k != weight) ws(weightLayer).update(k, 0.0))
        val in = flow(x, 0, weightLayer - 1).map { i =>
          _layers(weightLayer) match {
            case h: HasActivator[Double] => h.activator(i)
            case _ => i
          }
        }
        val ds = _layers.drop(weightLayer + 1).map {
          case h: HasActivator[Double] =>
            val i = _layers.indexOf(h) - 1
            flow(x, 0, i).map(h.activator.derivative)
        }
        (flow(x, 0, fastLayersSize1) - y) *:* chain(ds, ws, in, weightLayer, 0)
    }.reduce(_ + _)
  }

  /**
    * Constructs overall chain rule derivative based on single derivatives `ds` recursively.
    */
  @tailrec private def chain(ds: Matrices, ws: Matrices, in: Matrix, cursor: Int, cursorDs: Int): Matrix = {
    if (cursor < ws.size - 1) chain(ds, ws, ds(cursorDs) *:* (in * ws(cursor)), cursor + 1, cursorDs + 1)
    else ds(cursorDs) *:* (in * ws(cursor))
  }

  /**
    * Approximates the gradient based on finite central differences.
    */
  private def approximateErrorFuncDerivative(xs: Matrices, ys: Matrices,
                                         layer: Int, weight: (Int, Int)): Matrix = {
    finiteCentralDiff(xs, ys, layer, weight, order = 1)
  }

  /**
    * Computes the finite central diff for respective `order`.
    */
  private def finiteCentralDiff(xs: Matrices, ys: Matrices,
                                layer: Int, weight: (Int, Int), order: Int): Matrix = {
    val Δ = settings.approximation.getOrElse(Approximation(1E-5)).Δ
    val f = () => if (order == 1) errorFunc(xs, ys) else approximateErrorFuncDerivative(xs, ys, layer, weight)
    val v = weights(layer)(weight)
    weights(layer).update(weight, v - Δ)
    val a = f.apply
    weights(layer).update(weight, v + Δ)
    val b = f.apply
    (b - a) / (2 * Δ)
  }

  /**
    * Tries to find the optimal step size α through backtracking line search.
    */

  val (τ, c) = settings.specifics.map(s => (s.getOrElse("τ", 0.5), s.getOrElse("c", 0.5))).getOrElse((0.5, 0.5))

  @tailrec private def α(stepSize: Double, direction: Double, xs: Matrices, ys: Matrices,
                weightLayer: Int, weight: (Int, Int)): Double = {
    val v = weights(weightLayer)(weight)
    val t = -c * (-direction * direction)
    val a = mean(errorFunc(xs, ys))
    weights(weightLayer).update(weight, v + (stepSize * direction))
    val b = mean(errorFunc(xs, ys))
    weights(weightLayer).update(weight, v)
    if ((a - b) < (stepSize * t)) α(stepSize * τ, direction, xs, ys, weightLayer, weight) else stepSize
  }

}
