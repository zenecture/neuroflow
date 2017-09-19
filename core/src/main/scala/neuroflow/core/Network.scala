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
  def apply[T <: Network[_, _], L <: HList](ls: L, settings: Settings = Settings())
                                     (implicit
                                      startsWith: L StartsWith In,
                                      endsWith: L EndsWith Out,
                                      weightProvider: WeightProvider,
                                      constructor: Constructor[T],
                                      toList: L ToList Layer): T = {
    constructor(ls.toList, settings)
  }

}


/** For the sake of beauty. */
trait TypeAliases {

  type SVector  = scala.Vector[Double]
  type Vector   = DenseVector[Double]
  type Matrix   = DenseMatrix[Double]
  type Vectors  = Seq[Vector]
  type Matrices = Seq[Matrix]
  type Weights  = IndexedSeq[Matrix]
  type Learning = PartialFunction[(Int, Double), Double]

}


/** A minimal constructor for a [[Network]]. */
@implicitNotFound("No network constructor in scope. Import your desired network or try: import neuroflow.nets.DefaultNetwork._")
trait Constructor[+T <: Network[_, _]] {
  def apply(ls: Seq[Layer], settings: Settings)(implicit weightProvider: WeightProvider): T
}


/**
  * The `verbose` flag indicates logging behavior.
  * The `learningRate` is a function from current iteration and learning rate, producing a new learning rate.
  * The `updateRule` defines the relationship between gradient, weights and learning rate during training.
  * The network will terminate either if `precision` is high enough or `iterations` is reached.
  * If `prettyPrint` is true, the layout will be rendered graphically.
  * The level of `parallelism` controls how many threads will be used for training.
  * For distributed training, `coordinator` and `transport` specific settings may be configured.
  * The `batchSize` controls how many samples are presented per weight update. (1=on-line, ..., n=full-batch)
  * The `errorFuncOutput` option prints the error func graph to the specified file/closure.
  * When `regularization` is provided, the respective regulator will try to avoid over-fitting.
  * A `waypoint` action can be specified, e.g. saving the weights along the way.
  * With `approximation`  the gradients will be approximated numerically.
  * With `partitions` a sequential training sequence can be partitioned for RNNs (0 index-based).
  * Some nets use specific parameters set in the `specifics` map.
  */
case class Settings(verbose: Boolean                            = true,
                    learningRate: Learning                      = { case (_, _) => 1E-4 },
                    updateRule: Update                          = Vanilla,
                    precision: Double                           = 1E-5,
                    iterations: Int                             = 100,
                    prettyPrint: Boolean                        = false,
                    parallelism: Int                            = Runtime.getRuntime.availableProcessors,
                    coordinator: Node                           = Node("0.0.0.0", 2552),
                    transport: Transport                        = Transport(100000, "128 MiB"),
                    batchSize: Option[Int]                      = None,
                    errorFuncOutput: Option[ErrorFuncOutput]    = None,
                    regularization: Option[Regularization]      = None,
                    waypoint: Option[Waypoint]                  = None,
                    approximation: Option[Approximation]        = None,
                    partitions: Option[Set[Int]]                = None,
                    specifics: Option[Map[String, Double]]      = None) extends Serializable


trait IllusionBreaker { self: Network[_, _] =>

  /**
    * Checks if the [[Settings]] are properly defined for this network.
    * Throws a [[neuroflow.core.IllusionBreaker.SettingsNotSupportedException]] if not. Default behavior is no op.
    */
  def checkSettings(): Unit = ()

}


object IllusionBreaker {

  class SettingsNotSupportedException(message: String) extends Exception(message)
  class NotSoundException(message: String) extends Exception(message)

}


trait Network[In, Out] extends (In => Out) with Logs with ErrorFuncGrapher with IllusionBreaker with Welcoming with Serializable {

  checkSettings()

  sayHi()

  val identifier: String

  /** Settings of this neural network. */
  val settings: Settings

  /** Layers of this neural network. */
  val layers: Seq[Layer]

  /** The weights are a bunch of matrices. */
  val weights: Weights

  /**
    * Computes output for given input `in`.
    * Alias for `net(x)` syntax.
    */
  def evaluate(in: In): Out = apply(in)

  override def toString: String = weights.foldLeft("")(_ + "\n---\n" + _)

}


trait FeedForwardNetwork extends Network[Vector, Vector] {

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("FFNs don't support partitions. This setting has no effect.")
  }

  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Vectors, ys: Vectors): Unit

}


trait ConvolutionalNetwork extends Network[Matrices, Vector] {

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("CNNs don't support partitions. This setting has no effect.")
  }

  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Seq[Matrices], ys: Vectors): Unit

}


trait RecurrentNetwork extends Network[Vectors, Vectors] {

  /**
    * Takes input `xs` and trains this network against output `ys`.
    */
  def train(xs: Vectors, ys: Vectors): Unit

}


trait DistributedTraining {

  /**
    * Triggers execution of training for nodes `ns`.
    */
  def train(ns: collection.Set[Node]): Unit

}


trait DistributedFeedForwardNetwork extends Network[Vector, Vector] with DistributedTraining {

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("FFNs don't support partitions. This setting has no effect.")
    if (settings.batchSize.isDefined)
      warn("Setting the batch size has no effect in distributed training.")
  }

}


trait DistributedConvolutionalNetwork extends Network[Matrices, Vector] with DistributedTraining {

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("CNNs don't support partitions. This setting has no effect.")
    if (settings.batchSize.isDefined)
      warn("Setting the batch size has no effect in distributed training.")
  }

}
