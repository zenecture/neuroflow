package neuroflow.nets

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import neuroflow.common.~>
import neuroflow.core.Activator._
import neuroflow.core.Network._
import neuroflow.core._

import scala.Seq
import scala.annotation.tailrec
import scala.collection._


/**
  * @author bogdanski
  * @since 07.07.16
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
  private val storageDelta = weights.size - hiddenLayers.size
  private val memCells = hiddenLayers.map(l => DenseMatrix.zeros[Double](1, l.neurons)) // mutable
  private val initialOut = hiddenLayers.map(l => DenseMatrix.zeros[Double](1, l.neurons))

  /**
    * Takes the input vector sequence `xs` to compute the output vector sequence.
    */
  def evaluate(xs: Seq[Vector]): Seq[Vector] = {
    val in = xs.map(x => DenseMatrix.create[Double](1, x.size, x.toArray)).toList
    ~> (reset) next unfoldingFlow(in, initialOut, Nil, Nil) map (_.map(_.toArray.toVector))
  }

  /**
    * Takes a sequence of input vectors `xs` and trains
    * this network against the corresponding output vectors `ys`.
    */
  def train(xs: Seq[Vector], ys: Seq[Vector]): Unit = {
    import settings._
    val in = xs.map(x => DenseMatrix.create[Double](1, x.size, x.toArray)).toList
    val out = ys.map(y => DenseMatrix.create[Double](1, y.size, y.toArray)).toList
    run(in, out, learningRate, precision, 0, iterations)
  }

  /**
    * The eval loop.
    */
  @tailrec private def run(xs: Matrices, ys: Matrices, stepSize: Double, precision: Double,
                           iteration: Int, maxIterations: Int): Unit = {
    val error = mean(errorFunc(xs, ys))
    if (error > precision && iteration < maxIterations) {
      if (settings.verbose) info(s"Taking step $iteration - Error: $error")
      adaptWeights(xs, ys, stepSize)
      run(xs, ys, stepSize, precision, iteration + 1, maxIterations)
    } else {
      if (settings.verbose) info(s"Took $iteration iterations of $maxIterations with error $error")
      reset // finally reset one more time
    }
  }

  /**
    * Evaluates the error function Σ1/2(prediction(x) - observation)² over time.
    */
  private def errorFunc(xs: Matrices, ys: Matrices): Matrix = {
    reset()
    val errs = unfoldingFlow(xs, initialOut, Nil, Nil)
    errs.zip(ys).map {
      case (y, t) => 0.5 * pow(y - t, 2)
    }.reduce(_ + _)
  }

  /**
    * Adapts the weight with truncated back prop through time or finite differences.
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
    weights(layer).update(weight, v)
    (b - a) / (2 * Δ)
  }

  /**
    * Resets internal state of this network.
    */
  private def reset(): Unit = memCells.foreach(cell => cell.foreachPair { case ((r, c), v) => cell.update(r, c, 0.0) })

  /**
    * Unfolds this network through time and space.
    */
  @tailrec private def unfoldingFlow(xs: Matrices, lastOuts: Matrices,
                                     newOuts: Matrices, res: Seq[Matrix]): Seq[Matrix] =
    xs match {
      case hd :: tl =>
        val (ri, newOut) = flow(hd, lastOuts, Nil)
        unfoldingFlow(xs = tl, lastOuts = newOut, newOuts = Nil, res = res :+ ri)
      case Nil => res
    }

  /**
    * Computes this network for a single time step.
    */
  @tailrec private def flow(in: Matrix, lastOuts: Matrices, newOuts: Matrices,
                   cursor: Int = 0, target: Int = layers.size - 1): (Matrix, Matrices) = {
    if (target < 0) (in, newOuts)
    else {
      val (processed, no) = layers(cursor) match {
        case h: HasActivator[Double] if cursor < storageDelta =>
          val c = cursor - 1
          val yOut = lastOuts(c)
          val wl = weights(target + c)
          val (wsNetIn, wsGateIn, wsGateOut) = reconstructWeights(wl, h)
          val netIn = (in + (yOut * wsNetIn)).map(h.activator)
          val gateIn = (in + (yOut * wsGateIn)).map(Sigmoid)
          val gateOut = (in + (yOut * wsGateOut)).map(Sigmoid)
          val state = (netIn :* gateIn) + memCells(c)
          val netOut = state.map(h.activator) :* gateOut
          state.foreachPair { case ((row, col), i) => memCells(c).update(row, col, i) }
          (netOut * weights(cursor), Seq(netOut))
        case h: HasActivator[Double] => (in.map(h.activator), Nil)
        case _ => (in * weights(cursor), Nil)
      }
      if (cursor < target) flow(processed, lastOuts, newOuts ++ no, cursor + 1) else (processed, newOuts)
    }
  }

  /**
    * Reconstructs the recurrent weights of layer `l` from a nested matrix `m`.
    */
  private def reconstructWeights(m: Matrix, l: Layer): (Matrix, Matrix, Matrix) = {
    val f = l.neurons * l.neurons
    val netIn = DenseMatrix.create[Double](l.neurons, l.neurons, m.data.slice(0, f))
    val gateIn = DenseMatrix.create[Double](l.neurons, l.neurons, m.data.slice(f, 2 * f))
    val gateOut = DenseMatrix.create[Double](l.neurons, l.neurons, m.data.slice(2 * f, 3 * f))
    (netIn, gateIn, gateOut)
  }

}
