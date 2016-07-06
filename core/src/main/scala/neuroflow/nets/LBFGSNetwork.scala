package neuroflow.nets

import breeze.linalg._
import breeze.numerics._
import breeze.optimize._
import breeze.stats._
import neuroflow.core.Network._
import neuroflow.core._

import scala.annotation.tailrec


/**
  *
  * This is a fully connected Neural Network that uses LBFGS,
  * a quasi-Newton method to find optimal weights.
  *
  * @author bogdanski
  * @since 12.06.16
  *
  */



object LBFGSNetwork {
  implicit val constructor: Constructor[LBFGSNetwork] = new Constructor[LBFGSNetwork] {
    def apply(ls: Seq[Layer], settings: Settings)(implicit weightProvider: WeightProvider): LBFGSNetwork = {
      LBFGSNetwork(ls, settings, weightProvider(ls))
    }
  }
}


private[nets] case class LBFGSNetwork(layers: Seq[Layer], settings: Settings, weights: Weights) extends FeedForwardNetwork {

  /**
    * Checks if the [[Settings]] are properly defined.
    * Might throw a [[SettingsNotSupportedException]].
    */
  override def check(): Unit =
    if (settings.regularization.isDefined)
      throw new SettingsNotSupportedException("No regularization other than built-in LBFGS supported.")

  /**
    * Takes a sequence of input vectors `xs` and trains this
    * feed forward network against the corresponding output vectors `ys`.
    */
  def train(xs: Seq[Seq[Double]], ys: Seq[Seq[Double]]): Unit = {

    import settings._

    val in = xs map (x => DenseMatrix.create[Double](1, x.size, x.toArray))
    val out = ys map (y => DenseMatrix.create[Double](1, y.size, y.toArray))

    /**
      * Maps from V to W_i
      */
    def ws(v: DenseVector[Double], i: Int): Weights = {
      val (neuronsLeft, neuronsRight) = (layers(i).neurons, layers(i + 1).neurons)
      val product = neuronsLeft * neuronsRight
      val weightValues = v.slice(0, product).toArray
      val partialWeights = Seq(DenseMatrix.create[Double](neuronsLeft, neuronsRight, weightValues))
      if (i < layers.size - 2) partialWeights ++ ws(v.slice(product, v.length), i + 1)
      else partialWeights
    }

    /**
      * Evaluates the error function Σ1/2(prediction(x) - observation)²
      */
    def errorFunc(v: DenseVector[Double]): Double = {
      val err = {
        in.zip(out).par.map {
          case (xx, yy) => 0.5 * pow(flow(ws(v, 0), xx, 0, layers.size - 1) - yy, 2)
        }.reduce(_ + _)
      }
      mean(err)
    }

    /**
      * Maps from W_i to V
      */
    def flatten: DenseVector[Double] = DenseVector(weights.foldLeft(Array.empty[Double])((l, r) => l ++ r.data))

    /**
      * Updates W_i using V
      */
    def update(v: DenseVector[Double]): Unit = {
      (ws(v, 0) zip weights) foreach {
        case (n, o) => n.foreachPair {
          case ((r, c), nv) => o.update(r, c, nv)
        }
      }
    }

    val mem = settings.specifics.flatMap(_.get("m").map(_.toInt)).getOrElse(3)
    val mzi = settings.specifics.flatMap(_.get("maxZoomIterations").map(_.toInt)).getOrElse(10)
    val mlsi = settings.specifics.flatMap(_.get("maxLineSearchIterations").map(_.toInt)).getOrElse(10)
    val approx = approximation.getOrElse(Approximation(1E-5)).Δ

    val gradientFunction = new ApproximateGradientFunction[Int, DenseVector[Double]](errorFunc, approx)
    val lbfgs = new NFLBFGS(maxIter = maxIterations, m = mem, maxZoomIter = mzi, maxLineSearchIter = mlsi, tolerance = settings.precision)
    val optimum = lbfgs.minimize(gradientFunction, flatten)

    update(optimum)

  }

  /**
    * Takes the input vector `x` to compute their output.
    */
  def evaluate(xs: Seq[Double]): Seq[Double] = {
    val input = DenseMatrix.create[Double](1, xs.size, xs.toArray)
    flow(weights, input, 0, layers.size - 1).toArray.toVector
  }

  /**
    * Computes the network recursively from `cursor` until `target` (both representing the 'layer indices')
    */
  @tailrec private def flow(weights: Weights, in: DenseMatrix[Double], cursor: Int, target: Int): DenseMatrix[Double] = {
    if (target < 0) in
    else {
      val processed = layers(cursor) match {
        case h: HasActivator[Double] =>
          if (cursor <= (weights.size - 1)) in.map(h.activator) * weights(cursor)
          else in.map(h.activator)
        case _ => in * weights(cursor)
      }
      if (cursor < target) flow(weights, processed, cursor + 1, target) else processed
    }
  }

}
