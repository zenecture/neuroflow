package neuroflow.nets

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
  * This is a Long Short-Term Memory Network. The standard LSTM model is implemented.
  * It comes with recurrent connections and a dedicated memory cell with input-, output-
  * and forget-gates for each neuron. Multiple layers can be stacked horizontally,
  * where the current layer gets input from the lower layers at the same time step
  * and from itself at the previous time step.
  *
  *   Remarks:
  *
  *      - Use the positive infinity vector ∞ for training tuples without a target.
  *      The error will be zero for this particular time step. Example:
  *         (Coltrane, ∞), (plays, ∞), (the, Blues)
  *
  *
  *   This is work in progress. Things may change. Use at own risk.
  *
  * @author bogdanski
  * @since 07.07.16
  *
  */


object LSTMNetwork {
  implicit val constructor: Constructor[LSTMNetwork] = new Constructor[LSTMNetwork] {
    def apply(ls: Seq[Layer], settings: Settings)(implicit weightProvider: WeightProvider): LSTMNetwork = {
      LSTMNetwork(ls, settings, weightProvider(ls))
    }
  }
}


private[nets] case class LSTMNetwork(layers: Seq[Layer], settings: Settings, weights: Weights) extends RecurrentNetwork {

  import neuroflow.core.Network._

  private val hiddenLayers = layers.drop(1).dropRight(1)
  private val initialOut = hiddenLayers.map(l => DenseMatrix.zeros[Double](1, l.neurons))
  private val separators = settings.partitions.getOrElse(Set.empty)
  private val zeroOutput = DenseMatrix.zeros[Double](1, layers.last.neurons)

  // mutable state
  private val memoryCells = hiddenLayers.map(l => DenseMatrix.zeros[Double](1, l.neurons))
  private var xIndices = Map.empty[Int, Int]
  private var yIndices = Map.empty[Int, Int]
  private var noTargets = Set.empty[Int]


  /**
    * Checks if the [[Settings]] are properly defined for this network.
    * Throws a [[SettingsNotSupportedException]] if not. Default behavior is no op.
    */
  override def checkSettings(): Unit = {
    super.checkSettings()
    if (settings.specifics.isDefined)
      warn("No specific settings supported. This setting object has no effect.")
  }

  /**
    * Takes the input vector sequence `xs` to compute the output vector sequence.
    */
  def evaluate(xs: Seq[Vector]): Seq[Vector] = {
    val in = xs.map(x => DenseMatrix.create[Double](1, x.size, x.toArray)).toList
    xIndices = in.map(identityHashCode).zipWithIndex.toMap
    ~> (reset) next unfoldingFlow(in, initialOut, Nil, Nil) map (_.map(_.toArray.toVector))
  }

  /**
    * Takes a sequence of input vectors `xs` and trains this
    * network against the corresponding output vectors `ys`.
    */
  def train(xs: Seq[Vector], ys: Seq[Vector]): Unit = {
    import settings._
    val in = xs.map(x => DenseMatrix.create[Double](1, x.size, x.toArray)).toList
    val out = ys.map(y => DenseMatrix.create[Double](1, y.size, y.toArray)).toList
    noTargets = ys.zipWithIndex.filter { case (vec, idx) => vec.forall(_ == Double.PositiveInfinity) }.map(_._2).toSet
    xIndices = in.map(identityHashCode).zipWithIndex.toMap
    yIndices = out.map(identityHashCode).zipWithIndex.toMap
    run(in, out, learningRate, precision, 0, iterations)
  }

  /**
    * The eval loop.
    */
  @tailrec private def run(xs: Matrices, ys: Matrices, stepSize: Double, precision: Double,
                           iteration: Int, maxIterations: Int): Unit = {
    val error = errorFunc(xs, ys)
    val errorMean = mean(error)
    if (errorMean > precision && iteration < maxIterations) {
      if (settings.verbose) info(f"Taking step $iteration - Mean Error $errorMean%.6g - Error Vector $error")
      maybeGraph(errorMean)
      adaptWeights(xs, ys, stepSize)
      run(xs, ys, stepSize, precision, iteration + 1, maxIterations)
    } else {
      if (settings.verbose) info(f"Took $iteration iterations of $maxIterations with Mean Error = $errorMean%.3g")
      reset() // finally reset one more time
    }
  }

  /**
    * Evaluates the error function Σ1/2(out(x) - target)² over time.
    */
  private def errorFunc(xs: Matrices, ys: Matrices): Matrix = {
    reset()
    val errs = unfoldingFlow(xs, initialOut, Nil, Nil)
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
          if (settings.approximation.isDefined) approximateErrorFuncDerivative(xs, ys, layer, k)
          else ??? // this is TODO
        l.update(k, v - stepSize * mean(grad))
      }
    }
  }

  /**
    * Approximates the gradient based on finite central differences.
    */
  private def approximateErrorFuncDerivative(xs: Matrices, ys: Matrices, layer: Int, weight: (Int, Int)): Matrix = {
    val Δ = settings.approximation.getOrElse(Approximation(1E-5)).Δ
    val v = weights(layer)(weight)
    weights(layer).update(weight, v - Δ)
    val a = errorFunc(xs, ys)
    weights(layer).update(weight, v + Δ)
    val b = errorFunc(xs, ys)
    (b - a) / (2 * Δ)
  }

  /**
    * Resets internal state of this network.
    */
  private def reset(): Unit = memoryCells.foreach(cell => cell.foreachPair { case ((r, c), v) => cell.update(r, c, 0.0) })

  /**
    * Unfolds this network through time and space.
    */
  @tailrec private def unfoldingFlow(xs: Matrices, lastOuts: Matrices,
                                     newOuts: Matrices, res: Matrices): Matrices =
    xs match {
      case hd :: tl if isSeparator(hd) =>
        val (ri, _) = flow(hd, lastOuts, Nil)
        reset()
        unfoldingFlow(xs = tl, lastOuts = initialOut, newOuts = Nil, res = res :+ ri)
      case hd :: tl =>
        val (ri, newOut) = flow(hd, lastOuts, Nil)
        unfoldingFlow(xs = tl, lastOuts = newOut, newOuts = Nil, res = res :+ ri)
      case Nil => res
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
        val state = (netIn :* gateIn) + (forget :* memoryCells(c))
        val netOut = state.map(h.activator) :* gateOut
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
