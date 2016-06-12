package neuroflow.core

import breeze.linalg.DenseMatrix
import neuroflow.common.Logs
import neuroflow.core.Network.Weights
import shapeless._
import shapeless.ops.hlist._

import scala.annotation.implicitNotFound

/**
  * @author bogdanski
  * @since 03.01.16
  */

object Network {

  type Weights = Seq[DenseMatrix[Double]]

  /**
    * Constructs a new [[Network]] with the respective [[Constructor]] in scope.
    * Additionally, it will prove that the architecture of the net is sound.
    */
  def apply[T <: Network, H, L <: HList](ls: H :: L, settings: Settings)(implicit
                                                                 startsWith: (H :: L) StartsWith Input,
                                                                 endsWith: L EndsWith Output,
                                                                 weightProvider: WeightProvider,
                                                                 constructor: Constructor[T],
                                                                 toList: (H :: L) ToList Layer): T = {
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
case class Settings(verbose: Boolean, learningRate: Double, precision: Double, maxIterations: Int,
                    regularization: Option[Regularization], approximation: Option[Approximation],
                    specifics: Option[Map[String, Double]]) extends Serializable


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
    * Input `xs` and desired output `ys` will be the mold for the weights.
    * Returns this `Network`, with new weights.
    */
  def train(xs: Seq[Seq[Double]], ys: Seq[Seq[Double]]): Unit

  /**
    * Input `xs` will be evaluated based on current weights
    */
  def evaluate(xs: Seq[Double]): Seq[Double]

  override def toString: String = weights.foldLeft("")(_ + "\n---\n" + _)
}
