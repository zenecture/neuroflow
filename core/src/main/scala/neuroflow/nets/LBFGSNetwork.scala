package neuroflow.nets

import breeze.linalg._
import breeze.math.{MutableInnerProductModule, NormedModule}
import breeze.numerics._
import breeze.optimize.FirstOrderMinimizer.{ConvergenceCheck, ConvergenceReason, State}
import breeze.optimize._
import breeze.stats.mean
import neuroflow.core.Network.Weights
import neuroflow.core._
import neuroflow.nets.NFLBFGS.ErrorFunctionMin

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
    * Input `xs` and desired output `ys` will be the mold for the weights.
    * Returns this `Network`, with new weights.
    */
  def train(xs: Seq[Seq[Double]], ys: Seq[Seq[Double]]): Unit = {

    import settings._

    val in = xs map (x => DenseMatrix.create[Double](1, x.size, x.toArray))
    val out = ys map (y => DenseMatrix.create[Double](1, y.size, y.toArray))

    /**
      * Builds Net(W_i) from V.
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
        in.zip(out).par.map { t =>
          val (xx, yy) = t
          0.5 * pow(flow(ws(v, 0), xx, 0, layers.size - 1) - yy, 2)
        }.reduce(_ + _)
      }
      mean(err)
    }

    /**
      * Maps from W_i to V
      */
    def flatten: DenseVector[Double] = DenseVector(weights.foldLeft(Array.empty[Double])((l, r) => l ++ r.data))

    /**
      * Updates W_i using V.
      */
    def update(v: DenseVector[Double]): Unit = {
      (ws(v, 0) zip weights) foreach { nw =>
        val (n, o) = nw
        n.foreachPair { case ((r, c), nv) => o.update(r, c, nv) }
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
    * Input `xs` will be evaluated based on current weights
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
        case i => in * weights(cursor)
      }
      if (cursor < target) flow(weights, processed, cursor + 1, target) else processed
    }
  }

}

private[nets] class NFLBFGS(cc: ConvergenceCheck[DenseVector[Double]], m: Int, maxZoomIter: Int, maxLineSearchIter: Int)
                           (implicit space: MutableInnerProductModule[DenseVector[Double], Double]) extends LBFGS[DenseVector[Double]](cc, m)(space) {

  def this(maxIter: Int = -1, m: Int = 7, tolerance: Double = 1E-5, maxZoomIter: Int, maxLineSearchIter: Int)(implicit space: MutableInnerProductModule[DenseVector[Double], Double]) =
    this(NFLBFGS.defaultConvergenceCheck(maxIter, tolerance), m, maxZoomIter, maxLineSearchIter)

  override protected def determineStepSize(state: State, f: DiffFunction[DenseVector[Double]], dir: DenseVector[Double]): Double = {
    val x = state.x
    val ff = LineSearch.functionFromSearchDirection(f, x, dir)
    val search = new StrongWolfeLineSearch(maxZoomIter, maxLineSearchIter)
    search.minimize(ff, if(state.iter == 0.0) 1.0 / norm(dir) else 1.0)
  }

}

private[nets] object NFLBFGS {

  import FirstOrderMinimizer._

  def defaultConvergenceCheck[T](maxIter: Int, tolerance: Double, relative: Boolean = false, fvalMemory: Int = 20)
                                (implicit space: NormedModule[T, Double]): ConvergenceCheck[T] =
      maxIterationsReached[T](maxIter) ||
      ErrorFunctionValue(lessThan = tolerance, historyLength = 10) ||
      searchFailed

  case object ErrorFunctionMin extends ConvergenceReason {
    def reason = "error function is sufficiently minimal."
  }

}


private[nets] case class ErrorFunctionValue[T](lessThan: Double, historyLength: Int) extends ConvergenceCheck[T] {

  type Info = IndexedSeq[Double]

  def update(newX: T, newGrad: T, newVal: Double, oldState: State[T, _, _], oldInfo: Info): Info =
    (oldInfo :+ newVal).takeRight(historyLength)

  def apply(state: State[T, _, _], info: IndexedSeq[Double]): Option[ConvergenceReason] =
    if (info.length >= 2 && (state.adjustedValue <= lessThan)) Some(ErrorFunctionMin) else None

  def initialInfo: Info = IndexedSeq(Double.PositiveInfinity)

}
