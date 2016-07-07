package neuroflow.nets

import breeze.linalg._
import neuroflow.common.~>
import neuroflow.core.Activator._
import neuroflow.core.Network.{Vector, _}
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

  type Matrix = DenseMatrix[Double]
  type Matrices = Seq[Matrix]

  val hiddenLayers = layers.drop(1).dropRight(1)
  val storageDelta = weights.size - hiddenLayers.size
  val memCells = hiddenLayers.map(l => DenseMatrix.zeros[Double](1, l.neurons))

  /**
    * Takes the input vector sequence `xs` to compute the output vector sequence.
    */
  def evaluate(xs: Seq[Vector]): Seq[Vector] = {
    val in = xs.map(x => DenseMatrix.create[Double](1, x.size, x.toArray)).toList
    ~> (reset) next unfoldingFlow(in, initialOut(in), Nil, Nil) map (_.map(_.toArray.toVector))
  }

  /**
    * Takes a sequence of input vectors `xs` and trains this
    * network against the corresponding output vectors `ys`.
    */
  def train(xs: Seq[Vector], ys: Seq[Vector]): Unit = ???

  /**
    * Resets internal state of this network.
    */
  private def reset(): Unit = memCells.foreach(cell => cell.foreachPair { case ((r, c), v) => cell.update(r, c, 0.0) })

  /**
    * Unfolds this network through time and space.
    */
  @tailrec private def unfoldingFlow(xs: Matrices, lastOuts: Matrices,
                                     newOuts: Matrices, res: Seq[Matrix]): Seq[Matrix] = xs match {
    case hd :: tl =>
      val (ri, no) = flow(hd, lastOuts, Nil)
      unfoldingFlow(tl, lastOuts, no, res :+ ri)
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
          val los = lastOuts(c)
          val wl = weights(target + c)
          val (wsNetIn, wsGateIn, wsGateOut) = reconstructWeights(wl, h)
          val netIn = (in + (los * wsNetIn)).map(h.activator)
          val gateIn = (in + (los * wsGateIn)).map(Sigmoid)
          val gateOut = (in + (los * wsGateOut)).map(Sigmoid)
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
    * Reconstructs the recurrent weights of layer `l` from a compressed matrix `m`.
    */
  private def reconstructWeights(m: Matrix, l: Layer): (Matrix, Matrix, Matrix) = {
    val f = l.neurons * l.neurons
    val netIn = DenseMatrix.create[Double](l.neurons, l.neurons, m.data.slice(0, f))
    val gateIn = DenseMatrix.create[Double](l.neurons, l.neurons, m.data.slice(f, 2 * f))
    val gateOut = DenseMatrix.create[Double](l.neurons, l.neurons, m.data.slice(2 * f, 3 * f))
    (netIn, gateIn, gateOut)
  }

  /**
    * Gives the initial results for recurrent connections.
    */
  private def initialOut(in: Matrices): Matrices = hiddenLayers.map(l => DenseMatrix.zeros[Double](1, l.neurons))

}
