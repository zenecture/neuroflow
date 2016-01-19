package neuroflow.core

import breeze.linalg.DenseMatrix
import neuroflow.common.Logs
import neuroflow.core.Network.Weights

/**
  * @author bogdanski
  * @since 03.01.16
  */

object Network {

  type Weights = Seq[DenseMatrix[Double]]

  /**
    * Constructs a new `Network` depending on `Constructor` with layers `ls` and settings `sets`.
    */
  def apply[T <: Network](ls: Seq[Layer])(implicit constructor: Constructor[T], weightProvider: WeightProvider): T = constructor(ls, Settings(false, 0.0, true))
  def apply[T <: Network](ls: Seq[Layer], sets: Settings)(implicit constructor: Constructor[T], weightProvider: WeightProvider): T = constructor(ls, sets)
}

/**
  * Constructor for nets
  */
trait Constructor[+T <: Network] {
  def apply(ls: Seq[Layer], settings: Settings)(implicit weightProvider: WeightProvider): T
}

/**
  * If `numericGradient` is true, the gradient will be approximated using step size `Δ`, which is alot faster
  * than actually deriving the whole net. The `verbose` flag indicates logging behavior.
  */
case class Settings(numericGradient: Boolean, Δ: Double, verbose: Boolean) extends Serializable

trait Network extends Logs with Serializable {

  /**
    * Settings of this neural network
    */
  val settings: Settings

  /**
    * Layers of this neural network
    */
  val layers: Seq[Layer]

  /**
    * The weights packed as a list of matrices
    */
  val weights: Weights

  /**
    * Input `xs` and output `ys` will be the mold for the weights.
    * Returns this `Network`, with new weights.
    */
  def train(xs: Seq[Seq[Double]], ys: Seq[Seq[Double]]): Unit
  def train(xs: Seq[Seq[Double]], ys: Seq[Seq[Double]], stepSize: Double, precision: Double, maxIterations: Int): Unit

  /**
    * Input `xs` will be evaluated based on current weights
    */
  def evaluate(xs: Seq[Double]): Seq[Double]

  override def toString: String = weights.foldLeft("")(_ + "\n---\n" + _)
}
