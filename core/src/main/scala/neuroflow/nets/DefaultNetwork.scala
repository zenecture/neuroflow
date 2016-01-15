package neuroflow.nets

import breeze.linalg.{DenseMatrix, sum}
import breeze.numerics._
import neuroflow.core.{HasActivator, Network}

import scala.annotation.tailrec

/**
  *
  * This is a standard artificial neural network, using gradient descent,
  * fully connected weights (no sharing).
  *
  *
  * @author bogdanski
  * @since 15.01.16
  */

trait DefaultNetwork extends Network {

  /**
    * Computes the network recursively from cursor until target
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
    * Constructs overall chain rule derivative based on single derivatives `ds` recursively.
    */
  @tailrec private def derive(ds: Seq[DenseMatrix[Double]], ws: Seq[DenseMatrix[Double]], in: DenseMatrix[Double], cursor: Int, cursorDs: Int): DenseMatrix[Double] = {
    if (cursor < ws.size - 1) derive(ds, ws, ds(cursorDs) :* (in * ws(cursor)), cursor + 1, cursorDs + 1) else ds(cursorDs) :* (in * ws(cursor))
  }

  /**
    * Computes the error function derivative with respect to `weight` in `layer`
    */
  private def deriveErrorFunc(xs: Seq[DenseMatrix[Double]], ys: Seq[DenseMatrix[Double]],
                              layer: Int, weight: (Int, Int)): DenseMatrix[Double] = {
    xs.zip(ys).map { t =>
      val (x, y) = t
      val ws = weights.map(_.copy)
      ws(layer).update(weight, 1.0)
      ws(layer).foreachKey(k => if (k != weight) ws(layer).update(k, 0.0))
      val in = flow(x, 0, layer - 1).map { i =>
        layers(layer) match {
          case h: HasActivator[Double] => h.activator(i)
          case _ => i
        }
      }
      val ds = layers.drop(layer + 1).map { k => k match {
        case h: HasActivator[Double] =>
          val i = layers.indexOf(k) - 1
          flow(x, 0, i).map(h.activator.derivative)
      }
      }
      (flow(x, 0, layers.size - 1) - y) :* derive(ds, ws, in, layer, 0)
    }.reduce(_ + _)
  }

  /**
    * Computes the gradient numerically based on finite differences.
    */
  private def numericGradient(xs: Seq[DenseMatrix[Double]], ys: Seq[DenseMatrix[Double]],
                              layer: Int, weight: (Int, Int)): DenseMatrix[Double] = {
    import settings.Δ
    val v = weights(layer)(weight)
    val a = errorFunc(xs, ys)
    weights(layer).update(weight, v + Δ)
    val b = errorFunc(xs, ys)
    (b - a) / Δ
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

  /**
    * Computes gradient via `deriveErrorFunc` for all weights,
    * and adapts their value using gradient descent.
    */
  private def adaptWeights(xs: Seq[DenseMatrix[Double]], ys: Seq[DenseMatrix[Double]], stepSize: Double): Unit = {
    weights.foreach { l =>
      l.foreachPair { (k, v) =>
        val layer = weights.indexOf(l)
        val grad = if (settings.numericGradient) numericGradient(xs, ys, layer, k) else deriveErrorFunc(xs, ys, layer, k)
        val mean = stepSize * sum(grad) / grad.rows
        l.update(k, v - mean)
      }
    }
  }

  /**
    * Trains this `Network` with optimal weights based on `xs` and `ys`
    */
  @tailrec private def run(xs: Seq[Seq[Double]], ys: Seq[Seq[Double]], stepSize: Double, precision: Double,
                           iteration: Int, maxIterations: Int): Unit = {
    val input = xs map (x => DenseMatrix.create[Double](1, x.size, x.toArray))
    val output = ys map (y => DenseMatrix.create[Double](1, y.size, y.toArray))
    val error = errorFunc(input, output)
    if (error.toArray.exists(_ > precision) && iteration < maxIterations) {
      if (settings.verbose) info(s"Taking step $iteration - error: $error, error per sample: ${sum(error) / input.size}")
      adaptWeights(input, output, stepSize)
      run(xs, ys, stepSize, precision, iteration + 1, maxIterations)
    } else {
      if (settings.verbose) info(s"Took $iteration iterations of $maxIterations with error $error")
    }
  }

  /**
    * Input `xs` and output `ys` will be the mold for the weights.
    * Returns this `Network`, with new weights.
    */
  def train(xs: Seq[Seq[Double]], ys: Seq[Seq[Double]]): Unit =
    run(xs, ys, 0.01, 0.001, 0, 1000)
  def train(xs: Seq[Seq[Double]], ys: Seq[Seq[Double]], stepSize: Double, precision: Double, maxIterations: Int): Unit =
    run(xs, ys, stepSize, precision, 0, maxIterations)

  /**
    * Input `xs` will be evaluated based on current weights
    */
  def evaluate(xs: Seq[Double]): List[Double] = {
    val input = DenseMatrix.create[Double](1, xs.size, xs.toArray)
    flow(input, 0, layers.size - 1).toArray.toList
  }
}
