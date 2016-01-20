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
  def apply[T <: Network](ls: Seq[Layer])(implicit constructor: Constructor[T], weightProvider: WeightProvider): T = constructor(ls, NetSettings(false, 0.0, true))
  def apply[T <: Network](ls: Seq[Layer], sets: NetSettings)(implicit constructor: Constructor[T], weightProvider: WeightProvider): T = constructor(ls, sets)
}

/**
  * Constructor for nets
  */
trait Constructor[+T <: Network] {
  def apply(ls: Seq[Layer], settings: NetSettings)(implicit weightProvider: WeightProvider): T
}

/**
  * If `numericGradient` is true, the gradient will be approximated using step size `Δ`, which is a lot faster
  * than actually deriving the whole net. The `verbose` flag indicates logging behavior.
  */
case class NetSettings(numericGradient: Boolean, Δ: Double, verbose: Boolean) extends Serializable

/**
  * The `learningRate` determines the amplification of the gradients. The network will terminate
  * either if `precision` is high enough or `maxIterations` is reached. If `regularization` is provided,
  * during training the respective regulator will try to avoid over-fitting.
  */
case class TrainSettings(learningRate: Double, precision: Double, maxIterations: Int, regularization: Option[Regularization]) extends Serializable

trait Network extends Logs with Serializable {

  /**
    * Settings of this neural network
    */
  val settings: NetSettings

  /**
    * Layers of this neural network
    */
  val layers: Seq[Layer]

  /**
    * The weights packed as a list of matrices
    */
  val weights: Weights

  /**
    * Input `xs` and desired output `ys` will be the mold for the weights.
    * Returns this `Network`, with new weights. Use `trainSettings` for fine tuning.
    */
  def train(xs: Seq[Seq[Double]], ys: Seq[Seq[Double]], trainSettings: TrainSettings): Unit

  /**
    * Input `xs` will be evaluated based on current weights
    */
  def evaluate(xs: Seq[Double]): Seq[Double]

  override def toString: String = weights.foldLeft("")(_ + "\n---\n" + _)
}
