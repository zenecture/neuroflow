package neuroflow.nets

import breeze.linalg._
import breeze.numerics._
import breeze.optimize._
import breeze.stats._
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

  val hiddenLayers = layers.drop(1).dropRight(1)
  val cells = hiddenLayers.map(l => DenseMatrix.zeros[Double](1, l.neurons))

  /**
    * Takes the input sequence `xs` to compute its output.
    */
  def evaluate(xs: Seq[Vector]): Vector = {
    val in = xs.map(x => DenseMatrix.create[Double](1, x.size, x.toArray)).toList
    ~> (reset) next unfoldingFlow
  }

  /**
    * Takes a sequence of input vectors `xs` and trains this
    * network against the corresponding output vectors `ys`.
    */
  def train(xs: Seq[Vector], ys: Seq[Vector]): Unit = ???

  /**
    * Resets internal state of this network.
    */
  private def reset(): Unit = cells.foreach(cell => cell.foreachPair { case ((r, c), v) => cell.update(r, c, 0.0) })

  private def unfoldingFlow: Vector = ???

}
