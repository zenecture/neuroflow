package neuroflow.core

import breeze.linalg.DenseMatrix
import neuroflow.common.{Logs, ~>}
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
  * The network will terminate either if `precision` is high enough or `iterations` is reached. If `regularization`
  * is provided, during training the respective regulator will try to avoid over-fitting. If `approximation` is provided,
  * gradients will be approximated numerically, which can be fast yet more unprecise. Some nets require specific parameters specified
  * in the `specifics` mapping.
  */
case class Settings(verbose: Boolean = true, learningRate: Double = 0.1, precision: Double = 1E-5, iterations: Int = 10,
                    regularization: Option[Regularization] = None, approximation: Option[Approximation] = None,
                    specifics: Option[Map[String, Double]] = None) extends Serializable


trait IllusionBreaker { self: Network =>

  class SettingsNotSupportedException(message: String) extends Exception(message)

  /**
    * Checks if the [[Settings]] are properly defined for this network.
    * Throws a [[SettingsNotSupportedException]] if not. Default behavior is no op.
    */
  def checkSettings(): Unit = ()

}


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

  /**
    * Takes a sequence of input vectors `xs` and trains this
    * network against the corresponding output vectors `ys`.
    */
  def train(xs: Seq[Vector], ys: Seq[Vector]): Unit

  override def toString: String = weights.foldLeft("")(_ + "\n---\n" + _)

}


trait FeedForwardNetwork extends Network with IllusionBreaker {

  checkSettings()

  /**
    * Takes the input vector `x` to compute the output vector.
    */
  def evaluate(x: Vector): Vector

}


trait RecurrentNetwork extends Network with IllusionBreaker {

  checkSettings()


  /**
    * Takes the input vector sequence `xs` to compute the mean output vector.
    */
  def evaluateMean(xs: Seq[Vector]): Vector =
    ~> (evaluate(xs)) map(res => res.reduce { (r, v) => r.zip(v).map { case (a, b) => a + b } } map { _ / res.size })

  /**
    * Takes the input vector sequence `xs` to compute the output vector sequence.
    */
  def evaluate(xs: Seq[Vector]): Seq[Vector]

}
