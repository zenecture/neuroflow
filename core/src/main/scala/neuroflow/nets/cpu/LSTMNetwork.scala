package neuroflow.nets.cpu

import java.lang.System.identityHashCode

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import neuroflow.common.~>
import neuroflow.core.Activator._
import neuroflow.core.Network._
import neuroflow.core._

import scala.annotation.tailrec
import scala.collection._


/**
  *
  * This is a Long-Short-Term Memory Network. The standard model is implemented.
  * It comes with recurrent connections and a dedicated memory cell with input-, output-
  * and forget-gates for each neuron. Multiple layers can be stacked horizontally,
  * where the current layer gets input from the lower layers at the same time step
  * and from itself at the previous time step.
  *
  * Gradients are approximated, don't use it with big data.
  *
  *
  * @author bogdanski
  * @since 07.07.16
  *
  */


object LSTMNetwork {
  implicit val double: Constructor[Double, LSTMNetworkDouble] = new Constructor[Double, LSTMNetworkDouble] {
    def apply(ls: Seq[Layer], settings: Settings[Double])(implicit weightProvider: WeightProvider[Double]): LSTMNetworkDouble = {
      LSTMNetworkDouble(ls, settings, weightProvider(ls))
    }
  }
}


private[nets] case class LSTMNetworkDouble(layers: Seq[Layer], settings: Settings[Double], weights: Weights[Double],
                                     identifier: String = "neuroflow.nets.cpu.LSTMNetwork", numericPrecision: String = "Double")
  extends RNN[Double] with KeepBestLogic[Double] with WaypointLogic[Double] {

  type Vector   = Network.Vector[Double]
  type Vectors  = Network.Vectors[Double]
  type Matrix   = Network.Matrix[Double]
  type Matrices = Network.Matrices[Double]

  private val hiddenLayers = layers.drop(1).dropRight(1)
  private val initialOut   = hiddenLayers.map(l => DenseMatrix.zeros[Double](1, l.neurons)).toArray
  private val     _ANil    = Array.empty[Matrix]
  private val separators   = settings.partitions.getOrElse(Set.empty)
  private val zeroOutput   = DenseMatrix.zeros[Double](1, layers.last.neurons)

  // mutable state
  private val memoryCells  = hiddenLayers.map(l => DenseMatrix.zeros[Double](1, l.neurons))
  private var   xIndices   = Map.empty[Int, Int]
  private var   yIndices   = Map.empty[Int, Int]
  private var   noTargets  = Set.empty[Int]


  /**
    * Checks if the [[Settings]] are properly defined for this network.
    * Throws a [[neuroflow.core.IllusionBreaker.SettingsNotSupportedException]] if not. Default behavior is no op.
    */
  override def checkSettings(): Unit = {
    super.checkSettings()
    if (settings.specifics.isDefined)
      warn("No specific settings supported. This setting object has no effect.")
  }

  /**
    * Computes output sequence for `xs`.
    */
  def apply(xs: Vectors): Vectors = {
    val in = xs.map(x => x.asDenseMatrix)
    xIndices = in.map(identityHashCode).zipWithIndex.toMap
    ~> (reset()) next unfoldingFlow(in, initialOut, _ANil, _ANil, 0) map (_.map(_.toDenseVector))
  }

  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Vectors, ys: Vectors): Unit = {
    import settings._
    val in = xs.map(x => x.asDenseMatrix).toArray
    val out = ys.map(y => y.asDenseMatrix).toArray
    noTargets = ys.zipWithIndex.filter { case (vec, idx) => vec.forall(_ == Double.PositiveInfinity) }.map(_._2).toSet
    xIndices = in.map(identityHashCode).zipWithIndex.toMap
    yIndices = out.map(identityHashCode).zipWithIndex.toMap
    run(in, out, learningRate(1 -> 1.0), precision, 0, iterations)
  }

  /**
    * The eval loop.
    */
  @tailrec private def run(xs: Matrices, ys: Matrices, stepSize: Double, precision: Double,
                           iteration: Int, maxIterations: Int): Unit = {
    val error = lossFunction(xs, ys)
    val errorMean = mean(error)
    if (errorMean > precision && iteration < maxIterations) {
      if (settings.verbose) info(f"Iteration $iteration, Avg. Loss = $errorMean%.6g, Vector: $error")
      maybeGraph(errorMean)
      adaptWeights(xs, ys, stepSize)
      keepBest(errorMean)
      waypoint(iteration)
      run(xs, ys, settings.learningRate(iteration + 1 -> stepSize), precision, iteration + 1, maxIterations)
    } else {
      info(f"Took $iteration of $maxIterations iterations.")
      takeBest()
      reset() // finally reset one more time
    }
  }

  /**
    * Evaluates the loss function Σ1/2(out(x) - target)² over time.
    */
  private def lossFunction(xs: Matrices, ys: Matrices): Matrix = {
    reset()
    val errs = unfoldingFlow(xs, initialOut, Array.empty[Matrix], Array.empty[Matrix], 0)
    errs.zip(ys).map {
      case (_, t) if noTargets.contains(yIndices(identityHashCode(t))) => zeroOutput
      case (y, t) => 0.5 * pow(y - t, 2)
    }.reduce(_ + _)
  }

  /**
    * Adapts the weights with truncated back prop through time or finite differences.
    */
  private def adaptWeights(xs: Matrices, ys: Matrices, stepSize: Double): Unit = {
    weights.foreach { l =>
      l.foreachPair { (k, v) =>
        val layer = weights.indexOf(l)
        val grad =
          if (settings.approximation.isDefined) approximateGradient(xs, ys, layer, k)
          else ??? // this is TODO
        l.update(k, v - stepSize * grad)
      }
    }
  }

  /**
    * Approximates the gradient based on finite central differences.
    */
  private def approximateGradient(xs: Matrices, ys: Matrices, layer: Int, weight: (Int, Int)): Double = {
    sum(settings.approximation.get.apply(weights, () => lossFunction(xs, ys), () => (), layer, weight))
  }

  /**
    * Resets internal state of this network.
    */
  private def reset(): Unit = memoryCells.foreach(cell => cell.foreachPair { case ((r, c), v) => cell.update(r, c, 0.0) })

  /**
    * Unfolds this network through time and space.
    */
  @tailrec private def unfoldingFlow(xs: Matrices, lastOuts: Matrices,
                                     newOuts: Matrices, res: Matrices, cur: Int): Matrices = {
    val in = cur < xs.length
    if (in) {
      val hd = xs(cur)
      if (isSeparator(hd)) {
        val (ri, _) = flow(hd, lastOuts, _ANil)
        reset()
        unfoldingFlow(xs, lastOuts = initialOut, newOuts = _ANil, res = res :+ ri, cur = cur + 1)
      } else {
        val (ri, newOut) = flow(hd, lastOuts, _ANil)
        unfoldingFlow(xs, lastOuts = newOut, newOuts = _ANil, res = res :+ ri, cur = cur + 1)
      }
    } else { res }
  }

  /**
    * Checks if given matrix `m` is the end of logical input partition.
    */
  private def isSeparator(m: Matrix): Boolean = separators.contains(xIndices(identityHashCode(m)))

  /**
    * Computes this network for a single time step.
    */
  @tailrec private def flow(in: Matrix, lastOuts: Matrices, newOuts: Matrices,
                   cursor: Int = 0, target: Int = layers.size - 1): (Matrix, Matrices) = {
    val c = cursor - 1
    val (processed, newOut) = layers(cursor) match {
      case h: HasActivator[Double] if cursor < target =>
        val yOut = lastOuts(c)
        val getWs = (wt: Int) => weights(target + ((c * 7) + wt))
        val (wsNetGateIn, wsNetGateOut, wsNetForget) = (getWs(0), getWs(1), getWs(2))
        val (wsNetIn, wsGateIn, wsGateOut, wsForget) = (getWs(3), getWs(4), getWs(5), getWs(6))
        val (in1, in2, in3, in4) = (in * weights(c), in * wsNetGateIn, in * wsNetGateOut, in * wsNetForget)
        val netIn = (in1 + (yOut * wsNetIn)).map(h.activator)
        val gateIn = (in2 + (yOut * wsGateIn)).map(Sigmoid)
        val gateOut = (in3 + (yOut * wsGateOut)).map(Sigmoid)
        val forget = (in4 + (yOut * wsForget)).map(Sigmoid)
        val state = (netIn *:* gateIn) + (forget *:* memoryCells(c))
        val netOut = state.map(h.activator) *:* gateOut
        state.foreachPair { case ((row, col), i) => memoryCells(c).update(row, col, i) }
        (netOut, Some(netOut))
      case h: HasActivator[Double] =>
        val netOut = (in * weights(c)).map(h.activator)
        (netOut, None)
      case _ =>
        (in, None)
    }
    if (cursor < target) flow(processed, lastOuts, newOuts ++ newOut, cursor + 1) else (processed, newOuts)
  }

}
