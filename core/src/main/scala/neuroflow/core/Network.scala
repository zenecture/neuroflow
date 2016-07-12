package neuroflow.core

import breeze.linalg.{DenseMatrix, DenseVector}
import neuroflow.common._
import neuroflow.core.Network._
import shapeless._
import shapeless.ops.hlist._

import scala.annotation.implicitNotFound
import scala.collection._

/**
  * @author bogdanski
  * @since 03.01.16
  */

object Network extends TypeAliases {

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
  * For the sake of beauty.
  */
trait TypeAliases {

  type Vector = Seq[Double]
  type DVector = DenseVector[Double]
  type Matrix = DenseMatrix[Double]
  type Matrices = Seq[Matrix]
  type Weights = Seq[Matrix]

}


/**
  * A minimal constructor for a [[Network]].
  */
@implicitNotFound("No network constructor in scope. Import your desired network or try: import neuroflow.nets.DefaultNetwork._")
trait Constructor[+T <: Network] {
  def apply(ls: Seq[Layer], settings: Settings)(implicit weightProvider: WeightProvider): T
}


/**
  * The `verbose` flag indicates logging behavior.
  * The `learningRate` determines the amplification of the gradients.
  * The network will terminate either if `precision` is high enough or `iterations` is reached.
  * The `errorFuncOutput` option prints the error func graph to the specified file/closure
  * When `regularization` is provided, the respective regulator will try to avoid over-fitting.
  * With `approximation`  the gradients will be approximated numerically.
  * With `partitions` a training sequence for RNNs can be logically partitioned (0 index-based).
  * Some nets require specific parameters in the `specifics` mapping.
  */
case class Settings(verbose: Boolean = true,
                    learningRate: Double = 0.1,
                    precision: Double = 1E-5,
                    iterations: Int = 100,
                    errorFuncOutput: Option[ErrorFuncOutput] = None,
                    regularization: Option[Regularization] = None,
                    approximation: Option[Approximation] = None,
                    partitions: Option[Vector] = None,
                    specifics: Option[Map[String, Double]] = None) extends Serializable


trait IllusionBreaker { self: Network =>

  class SettingsNotSupportedException(message: String) extends Exception(message)

  /**
    * Checks if the [[Settings]] are properly defined for this network.
    * Throws a [[SettingsNotSupportedException]] if not. Default behavior is no op.
    */
  def checkSettings(): Unit = ()

}


trait Network extends Logs with ErrorFuncGrapher with IllusionBreaker with Welcoming with Serializable {

  checkSettings()

  /**
    * Settings of this neural network.
    */
  val settings: Settings

  /**
    * Layers of this neural network.
    */
  val layers: Seq[Layer]

  /**
    * The weights are a bunch of matrices.
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

  /**
    * Takes the input vector `x` to compute the output vector.
    */
  def evaluate(x: Vector): Vector

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("FFNs don't support partitions. This setting has no effect. You may remove it?")
  }

}


trait RecurrentNetwork extends Network with IllusionBreaker {

  /**
    * Takes the input vector sequence `xs` to compute the output vector sequence.
    */
  def evaluate(xs: Seq[Vector]): Seq[Vector]

  /**
    * Takes the input vector sequence `xs` to compute the mean output vector.
    */
  def evaluateMean(xs: Seq[Vector]): Vector =
    ~> (evaluate(xs)) map(res => res.reduce { (r, v) => r.zip(v).map { case (a, b) => a + b } } map { _ / res.size })

}
