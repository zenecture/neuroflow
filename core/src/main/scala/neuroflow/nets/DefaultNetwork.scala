package neuroflow.nets

import breeze.linalg.{DenseMatrix, sum}
import breeze.numerics._
import breeze.stats.mean
import neuroflow.core.Network.Weights
import neuroflow.core._

import scala.annotation.tailrec


/**
  *
  * This is a standard artificial neural network, using gradient descent,
  * fully connected weights, no sharing.
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


case class DefaultNetwork(layers: Seq[Layer], settings: Settings, weights: Weights) extends Network {

  /**
    * Input `xs` and output `ys` will be the mold for the weights.
    * Returns this `Network`, with new weights.
    */
  def train(xs: Seq[Seq[Double]], ys: Seq[Seq[Double]]): Unit = {
    import settings._
    run(xs, ys, learningRate, precision, 0, maxIterations)
  }

  /**
    * Input `xs` will be evaluated based on current weights
    */
  def evaluate(xs: Seq[Double]): Seq[Double] = {
    val input = DenseMatrix.create[Double](1, xs.size, xs.toArray)
    flow(input, 0, layers.size - 1).toArray.toVector
  }

  /**
    * Trains this `Network` with optimal weights based on `xs` and `ys`
    */
  @tailrec private def run(xs: Seq[Seq[Double]], ys: Seq[Seq[Double]], stepSize: Double, precision: Double,
                           iteration: Int, maxIterations: Int): Unit = {
    val input = xs map (x => DenseMatrix.create[Double](1, x.size, x.toArray))
    val output = ys map (y => DenseMatrix.create[Double](1, y.size, y.toArray))
    val error = errorFunc(input, output)
    if (mean(error) > precision && iteration < maxIterations) {
      if (settings.verbose) info(s"Taking step $iteration - error: $error, error per sample: ${sum(error) / input.size}")
      adaptWeights(input, output, stepSize)
      run(xs, ys, stepSize, precision, iteration + 1, maxIterations)
    } else {
      if (settings.verbose) info(s"Took $iteration iterations of $maxIterations with error $error")
    }
  }

  /**
    * Computes gradient via `deriveErrorFunc` for all weights,
    * and adapts their value using gradient descent.
    */
  private def adaptWeights(xs: Seq[DenseMatrix[Double]], ys: Seq[DenseMatrix[Double]], stepSize: Double): Unit = {
    weights.foreach { l =>
      l.foreachPair { (k, v) =>
        val layer = weights.indexOf(l)
        val grad = if (settings.approximation.isDefined) approximateErrorFuncDerivative(xs, ys, layer, k) else deriveErrorFunc(xs, ys, layer, k)
        l.update(k, v - stepSize * mean(grad))
      }
    }
  }

  /**
    * Computes the network recursively from `cursor` until `target` (both representing the 'layer indices')
    */
  @tailrec private def flow(in: DenseMatrix[Double], cursor: Int, target: Int): DenseMatrix[Double] = {
    if (target < 0) in
    else {
      val processed = layers(cursor) match {
        case h: HasActivator[Double] =>
          if (cursor <= (weights.size - 1)) in.map(h.activator) * weights(cursor)
          else in.map(h.activator)
        case i => in * weights(cursor)
      }
      if (cursor < target) flow(processed, cursor + 1, target) else processed
    }
  }

  /**
    * Computes the error function derivative with respect to `weight` in `weightLayer`
    */
  private def deriveErrorFunc(xs: Seq[DenseMatrix[Double]], ys: Seq[DenseMatrix[Double]],
                              weightLayer: Int, weight: (Int, Int)): DenseMatrix[Double] = {
    xs.zip(ys).map { t =>
      val (x, y) = t
      val ws = weights.map(_.copy)
      ws(weightLayer).update(weight, 1.0)
      ws(weightLayer).foreachKey(k => if (k != weight) ws(weightLayer).update(k, 0.0))
      val in = flow(x, 0, weightLayer - 1).map { i =>
        layers(weightLayer) match {
          case h: HasActivator[Double] => h.activator(i)
          case _ => i
        }
      }
      val ds = layers.drop(weightLayer + 1).map {
        case h: HasActivator[Double] =>
          val i = layers.indexOf(h) - 1
          flow(x, 0, i).map(h.activator.derivative)
      }
      (flow(x, 0, layers.size - 1) - y) :* chain(ds, ws, in, weightLayer, 0)
    }.reduce(_ + _)
  }

  /**
    * Constructs overall chain rule derivative based on single derivatives `ds` recursively.
    */
  @tailrec private def chain(ds: Seq[DenseMatrix[Double]], ws: Seq[DenseMatrix[Double]], in: DenseMatrix[Double], cursor: Int, cursorDs: Int): DenseMatrix[Double] = {
    if (cursor < ws.size - 1) chain(ds, ws, ds(cursorDs) :* (in * ws(cursor)), cursor + 1, cursorDs + 1) else ds(cursorDs) :* (in * ws(cursor))
  }

  /**
    * Approximates the gradient based on finite central differences.
    */
  private def approximateErrorFuncDerivative(xs: Seq[DenseMatrix[Double]], ys: Seq[DenseMatrix[Double]],
                              weightLayer: Int, weight: (Int, Int)): DenseMatrix[Double] = {
    val Δ = settings.approximation.get.Δ
    val v = weights(weightLayer)(weight)
    weights(weightLayer).update(weight, v - Δ)
    val a = errorFunc(xs, ys)
    weights(weightLayer).update(weight, v + Δ)
    val b = errorFunc(xs, ys)
    (b - a) / (2 * Δ)
  }

  /**
    * Evaluates the error function Σ1/2(prediction(x) - observation)²
    */
  private def errorFunc(xs: Seq[DenseMatrix[Double]], ys: Seq[DenseMatrix[Double]]): DenseMatrix[Double] = {
    xs.zip(ys).map { t =>
      val (x, y) = t
      0.5 * pow(flow(x, 0, layers.size - 1) - y, 2)
    }.reduce(_ + _)
  }

}
