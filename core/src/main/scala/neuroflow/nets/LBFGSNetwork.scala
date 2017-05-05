package neuroflow.nets

import breeze.linalg._
import breeze.numerics._
import breeze.optimize._
import breeze.stats._
import neuroflow.core.Network._
import neuroflow.core._

import scala.annotation.tailrec
import scala.collection.Seq
import scala.util.Random


/**
  *
  * This is a fully connected Neural Network that uses Breeze's LBFGS,
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


private[nets] case class LBFGSNetwork(layers: Seq[Layer], settings: Settings, weights: Weights,
                                      identifier: String = Random.alphanumeric.take(3).mkString) extends FeedForwardNetwork with SupervisedTraining {

  import neuroflow.core.Network._

  private val fastLayers = layers.map {
    case Cluster(inner) => inner
    case layer: Layer   => layer
  }.toArray
  private val fastLayerSize1 = layers.size - 1
  private val fastLayerSize2 = layers.size - 2

  private val fastWeights = weights.toArray
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
    * Takes a sequence of input vectors `xs` and trains this
    * network against the corresponding output vectors `ys`.
    */
  def train(xs: Seq[Vector], ys: Seq[Vector]): Unit = {

    import settings._

    val in = xs.map(x => DenseMatrix.create[Double](1, x.size, x.toArray)).toParArray
    val out = ys.map(y => DenseMatrix.create[Double](1, y.size, y.toArray)).toParArray
    val neuronProduct = (0 until fastLayerSize1).map(i => (i, i + 1) -> fastLayers(i).neurons * fastLayers(i + 1).neurons).toMap

    /**
      * Maps from V to W_i.
      */
    @tailrec def ws(pw: Seq[Matrix], v: DVector, i: Int): Seq[Matrix] = {
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
    def errorFunc(v: DVector): Double = {
      val err = mean {
        in.zip(out).map {
          case (xx, yy) => pow(flow(ws(Nil, v, 0), xx, 0, fastLayerSize1) - yy, 2)
        }.reduce(_ + _)
      }
      err
    }

    /**
      * Maps from W_i to V.
      */
    def flatten: DVector = DenseVector(weights.foldLeft(Array.empty[Double])((l, r) => l ++ r.toArray))

    /**
      * Updates W_i using V.
      */
    def update(v: DVector): Unit = {
      (ws(Nil, v, 0) zip weights) foreach {
        case (n, o) => n.foreachPair {
          case ((r, c), nv) => o.update(r, c, nv)
        }
      }
    }

    val mem = settings.specifics.flatMap(_.get("m").map(_.toInt)).getOrElse(3)
    val mzi = settings.specifics.flatMap(_.get("maxZoomIterations").map(_.toInt)).getOrElse(10)
    val mlsi = settings.specifics.flatMap(_.get("maxLineSearchIterations").map(_.toInt)).getOrElse(10)
    val approx = approximation.getOrElse(Approximation(1E-5)).Δ

    val gradientFunction = new ApproximateGradientFunction[Int, DVector](errorFunc, approx)
    val lbfgs = new NFLBFGS(verbose = settings.verbose, maxIter = iterations, m = mem, maxZoomIter = mzi,
      maxLineSearchIter = mlsi, tolerance = settings.precision, maybeGraph = maybeGraph)
    val optimum = lbfgs.minimize(gradientFunction, flatten)

    update(optimum)

  }

  /**
    * Takes the input vector `x` to compute the output vector.
    */
  def evaluate(x: Vector): Vector = {
    val input = DenseMatrix.create[Double](1, x.size, x.toArray)
    layers.collect {
      case c: Cluster => c
    }.headOption.map { cl =>
      flow(fastWeights, input, 0, layers.indexOf(cl) - 1).map(cl.inner.activator).toArray.toVector
    }.getOrElse {
      info("Couldn't find Cluster Layer. Using Output Layer.")
      flow(fastWeights, input, 0, layers.size - 1).toArray.toVector
    }
  }

  /**
    * Computes the network recursively from `cursor` until `target`.
    */
  @tailrec final protected def flow(weights: Weights, in: Matrix, cursor: Int, target: Int): Matrix = {
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
