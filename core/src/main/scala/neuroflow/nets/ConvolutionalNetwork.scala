package neuroflow.nets

import breeze.linalg._
import breeze.numerics._
import breeze.optimize._
import breeze.stats._
import neuroflow.common.Registry
import neuroflow.core.IllusionBreaker.SettingsNotSupportedException
import neuroflow.core.Network._
import neuroflow.core._

import scala.annotation.tailrec

/**
  *
  * This is a somewhat experimental conv net.
  *
  * @author bogdanski
  * @since 03.05.17
  *
  */


object ConvolutionalNetwork {
  implicit val constructor: Constructor[ConvolutionalNetwork] = new Constructor[ConvolutionalNetwork] {
    def apply(ls: Seq[Layer], settings: Settings)(implicit weightProvider: WeightProvider): ConvolutionalNetwork = {
      ConvolutionalNetwork(ls, settings, weightProvider(ls))
    }
  }
}

private[nets] case class ConvolutionalNetwork(layers: Seq[Layer], settings: Settings, weights: Weights,
                                              identifier: String = Registry.register()) extends FeedForwardNetwork with SupervisedTraining {

  import neuroflow.core.Network._

  private val fastLayers      = layers.toArray
  private val fastLayerSize1  = layers.size - 1
  private val fastLayerSize2  = layers.size - 2

  private val fastWeights     = weights.toArray
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
    * Takes the input vector `x` to compute the output vector.
    */
  def evaluate(x: Vector): Vector = {
    val input = DenseMatrix.create[Double](1, x.size, x.toArray)
    flow(fastWeights, input, 0, layers.size - 1).toArray.toVector
  }


  /**
    * Takes a sequence of input vectors `xs` and trains this
    * network against the corresponding output vectors `ys`.
    */
  def train(xs: Seq[Vector], ys: Seq[Vector]): Unit = {

    import settings._

    val in  = xs.map(x => DenseMatrix.create[Double](1, x.size, x.toArray)).toParArray
    val out = ys.map(y => DenseMatrix.create[Double](1, y.size, y.toArray)).toParArray

    /**
      * Maps from V to W_i.
      */
    @tailrec def ws(pw: Seq[Matrix], v: DVector, i: Int): Seq[Matrix] = {

      val (nu, prod) = (fastLayers(i), fastLayers(i + 1)) match {

        case (_: Layer, r: Convolutable) =>
          val p = r.filters * r.fieldSize
          val wsv = v.slice(0, p).toArray
          (DenseMatrix.create[Double](r.filters, r.fieldSize, wsv), p)

        case (l: Layer, r: Layer)       =>
          val p = l.neurons * r.neurons
          val wsv = v.slice(0, p).toArray
          (DenseMatrix.create[Double](l.neurons, r.neurons, wsv), p)

      }

      if (i < fastLayerSize2)
        ws(pw :+ nu, v.slice(prod, v.length), i + 1)
      else pw :+ nu

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
    * Computes the network recursively from `cursor` until `target`.
    */
  @tailrec final protected def flow(weights: Weights, in: Matrix, cursor: Int, target: Int): Matrix = {

    if (target < 0) in else {

      val processed = fastLayers(cursor) match {

        case c: Convolutable =>

          val rf = c.receptiveField(in)

          val convoluted = rf.map {
            field => weights(cursor - 1) * field
          }.reduce((l, r) => DenseMatrix.vertcat(l, r))

          fastLayers(cursor + 1) match {
            case _: Convolutable => convoluted.map(c.activator)
            case _: Layer        =>
              val cm = convoluted.map(c.activator).t
              val ms = (0 until cm.cols).map {
                c => mean(cm(::, c))
              }
              val ma = DenseMatrix.create[Double](1, ms.size, ms.toArray)
              ma * weights(cursor)
          }

        case l: Layer with HasActivator[Double] if cursor <= fastWeightSize1 =>
          fastLayers(cursor + 1) match {
            case _: Convolutable => in.map(l.activator)
            case _: Layer        => in.map(l.activator) * weights(cursor)
          }

        case o: Output => in.map(o.activator)

        case _: Input  => fastLayers(cursor + 1) match {
          case _: Convolutable   => in
          case _: Layer          => in * weights(cursor)
        }

      }

      if (cursor < target) flow(weights, processed, cursor + 1, target) else processed

    }

  }

}
