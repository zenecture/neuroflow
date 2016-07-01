package neuroflow.core

import breeze.linalg.DenseMatrix
import neuroflow.common.Logs
import neuroflow.core.Network.Weights
import neuroflow.core.Network.Vector
import shapeless._
import shapeless.ops.hlist._

import scala.annotation.implicitNotFound

/**
  * @author bogdanski
  * @since 03.01.16
  */

object Network {

  type Weights = Seq[DenseMatrix[Double]]
  type Vector = Seq[Double]

  /**
    * Constructs a new [[Network]] with the respective [[Constructor]] in scope.
    * Additionally, it will prove that the architecture of the net is sound.
    */
  def apply[T <: Network, L <: HList](ls: L, settings: Settings)(implicit
                                                                 startsWith: L StartsWith Input,
                                                                 endsWith: L EndsWith Output,
                                                                 weightProvider: WeightProvider,
                                                                 constructor: Constructor[T],
                                                                 toList: L ToList Layer): T = {
    constructor(ls.toList, settings)
  }

}


/**
  * A minimal constructor for a [[Network]].
  */
@implicitNotFound("No network constructor in scope. Import your desired network or try: import neuroflow.nets.DefaultNetwork._")
trait Constructor[+T <: Network] {
  def apply(ls: Seq[Layer], settings: Settings)(implicit weightProvider: WeightProvider): T
}


/**
  * The `verbose` flag indicates logging behavior. The `learningRate` determines the amplification of the gradients.
  * The network will terminate either if `precision` is high enough or `maxIterations` is reached. If `regularization` is provided,
  * during training the respective regulator will try to avoid over-fitting. If `approximation` is provided, gradients will be approximated numerically.
  * Some nets require specific parameters which can be mapped with `specifics`.
  */
case class Settings(verbose: Boolean = true, learningRate: Double = 1.0, precision: Double = 1E-5, maxIterations: Int = 10,
                    regularization: Option[Regularization] = None, approximation: Option[Approximation] = None,
                    specifics: Option[Map[String, Double]] = None) extends Serializable


trait Network extends Logs with Serializable {

  /**
    * Settings of this neural network.
    */
  val settings: Settings

  /**
    * Layers of this neural network.
    */
  val layers: Seq[Layer]

  /**
    * The weights packed as a sequence of matrices.
    */
  val weights: Weights

  override def toString: String = weights.foldLeft("")(_ + "\n---\n" + _)

}


trait FeedForwardNetwork extends Network {

  /**
    * Takes a sequence of input vectors `xs` and trains this
    * feed forward network against the corresponding output vectors `ys`.
    */
  def train(xs: Seq[Vector], ys: Seq[Vector]): Unit

  /**
    * Takes the input vector `x` to compute their output.
    */
  def evaluate(x: Vector): Vector

}



trait RecurrentNetwork extends Network {

  /**
    * Takes a time sequence of input vector sequences `xs` and trains
    * this recurrent network against the corresponding output time sequence `ys`.
    */
  def train(xs: Seq[Seq[Vector]], ys: Seq[Seq[Vector]]): Unit

  /**
    * Takes the input vector sequences `xs` to compute their output.
    */
  def evaluate(xs: Seq[Vector]): Seq[Vector]

  /**
    * Resets the internal state of this recurrent network.
    */
  def reset: Unit

}
