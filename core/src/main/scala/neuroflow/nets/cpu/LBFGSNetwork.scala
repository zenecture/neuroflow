package neuroflow.nets.cpu

import breeze.linalg._
import breeze.numerics._
import breeze.optimize._
import breeze.stats._
import neuroflow.core.IllusionBreaker.SettingsNotSupportedException
import neuroflow.core.Network._
import neuroflow.core._

import scala.annotation.tailrec
import scala.collection.Seq


/**
  *
  * This is a fully connected Neural Network that uses Breeze's LBFGS,
  * a quasi-Newton method to find optimal weights.
  *
  * Gradients are approximated, don't use it with big data.
  *
  * @author bogdanski
  * @since 12.06.16
  *
  */


object LBFGSNetwork {
  implicit val double: Constructor[Double, LBFGSNetwork] = new Constructor[Double, LBFGSNetwork] {
    def apply(ls: Seq[Layer], settings: Settings[Double])(implicit weightProvider: WeightProvider[Double]): LBFGSNetwork = {
      LBFGSNetwork(ls, settings, weightProvider(ls))
    }
  }
}


private[nets] case class LBFGSNetwork(layers: Seq[Layer], settings: Settings[Double], weights: Weights[Double],
                                      identifier: String = "neuroflow.nets.cpu.LBFGSNetwork", numericPrecision: String = "Double") extends FFN[Double] {

  type Vector   = Network.Vector[Double]
  type Vectors  = Network.Vectors[Double]
  type Matrix   = Network.Matrix[Double]
  type Matrices = Network.Matrices[Double]

  private val fastLayers = layers.map {
    case Focus(inner) => inner
    case layer: Layer   => layer
  }.toArray
  private val fastLayerSize1 = layers.size - 1
  private val fastLayerSize2 = layers.size - 2

  private val fastWeightSize1 = weights.size - 1

  /**
    * Checks if the [[Settings]] are properly defined.
    * Might throw a [[SettingsNotSupportedException]].
    */
  override def checkSettings(): Unit = {
    super.checkSettings()
    if (settings.regularization.isDefined)
      throw new SettingsNotSupportedException("No regularization supported.")
  }

  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Vectors, ys: Vectors): Unit = {

    import settings._

    val in = xs.map(x => x.asDenseMatrix)
    val out = ys.map(y => y.asDenseMatrix)
    val neuronProduct = (0 until fastLayerSize1).map(i => (i, i + 1) -> fastLayers(i).neurons * fastLayers(i + 1).neurons).toMap

    /**
      * Maps from V to W_i.
      */
    @tailrec def ws(pw: Array[Matrix], v: Vector, i: Int): Array[Matrix] = {
      val (neuronsLeft, neuronsRight) = (fastLayers(i).neurons, fastLayers(i + 1).neurons)
      val product = neuronProduct(i, i + 1)
      val weightValues = v.slice(0, product).toArray
      val partialWeights = DenseMatrix.create[Double](neuronsLeft, neuronsRight, weightValues)
      if (i < fastLayerSize2)
        ws(pw :+ partialWeights, v.slice(product, v.length), i + 1)
      else pw :+ partialWeights
    }

    /**
      * Evaluates the error function Σ1/2(prediction(x) - observation)² in parallel.
      */
    def errorFunc(v: Vector): Double = {
      val err = mean {
        in.zip(out).map {
          case (xx, yy) => pow(flow(ws(Array.empty[Matrix], v, 0), xx, 0, fastLayerSize1) - yy, 2)
        }.reduce(_ + _)
      }
      err
    }

    /**
      * Maps from W_i to V.
      */
    def flatten: Vector = DenseVector(weights.foldLeft(Array.empty[Double])((l, r) => l ++ r.toArray))

    /**
      * Updates W_i using V.
      */
    def update(v: Vector): Unit = {
      (ws(Array.empty[Matrix], v, 0) zip weights) foreach {
        case (n, o) => n.foreachPair {
          case ((r, c), nv) => o.update(r, c, nv)
        }
      }
    }

    val mem = settings.specifics.flatMap(_.get("m").map(_.toInt)).getOrElse(3)
    val mzi = settings.specifics.flatMap(_.get("maxZoomIterations").map(_.toInt)).getOrElse(10)
    val mlsi = settings.specifics.flatMap(_.get("maxLineSearchIterations").map(_.toInt)).getOrElse(10)
    val approx = approximation.getOrElse(Approximation(1E-5)).Δ

    val gradientFunction = new ApproximateGradientFunction[Int, Vector](errorFunc, approx)
    val lbfgs = new NFLBFGS(verbose = settings.verbose, maxIter = iterations, m = mem, maxZoomIter = mzi,
      maxLineSearchIter = mlsi, tolerance = settings.precision, maybeGraph = maybeGraph)
    val optimum = lbfgs.minimize(gradientFunction, flatten)

    update(optimum)

  }

  /**
    * Computes output for `x`.
    */
  def apply(x: Vector): Vector = {
    val input = DenseMatrix.create[Double](1, x.size, x.toArray)
    layers.collect {
      case c: Focus[Double] => c
    }.headOption.map { cl =>
      flow(weights, input, 0, layers.indexOf(cl) - 1).map(cl.inner.activator).toDenseVector
    }.getOrElse {
      flow(weights, input, 0, layers.size - 1).toDenseVector
    }
  }

  /**
    * Computes the network recursively from `cursor` until `target`.
    */
  @tailrec final protected def flow(weights: Weights[Double], in: Matrix, cursor: Int, target: Int): Matrix = {
    if (target < 0) in
    else {
      val processed = fastLayers(cursor) match {
        case h: HasActivator[Double] =>
          if (cursor <= fastWeightSize1) in.map(h.activator) * weights(cursor)
          else in.map(h.activator)
        case _ => in * weights(cursor)
      }
      if (cursor < target) flow(weights, processed, cursor + 1, target) else processed
    }
  }

}
