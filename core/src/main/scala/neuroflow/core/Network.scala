package neuroflow.core

import breeze.linalg.{DenseMatrix, DenseVector}
import neuroflow.common._
import neuroflow.core.Network._
import neuroflow.dsl.{ Vector => _, _ }

import scala.annotation.implicitNotFound
import scala.collection._

/**
  * @author bogdanski
  * @since 03.01.16
  */

object Network extends Lexicon {

  /**
    * Constructs a new [[Network]] with the respective [[Constructor]] in scope.
    * Additionally, it proves that the [[Layout]] graph is a valid composition.
    */
  def apply[V, L <: Layout, N <: Network[_, _, _]]
                                     (layout: L,
                                      settings: Settings[V] = Settings[V]())
                                     (implicit
                                      startsWith: L StartsWith In,
                                      endsWith: L EndsWith Out,
                                      weightBreeder: WeightBreeder[V],
                                      constructor: Constructor[V, N],
                                      extractor: Extractor[L, Layer, V]): N = {
    val (layers, loss) = extractor(layout)
    constructor(layers, loss, settings)
  }

}


trait Lexicon {

  type Vector[V]     =  DenseVector[V]
  type Matrix[V]     =  DenseMatrix[V]
  type Vectors[V]    =  Seq[Vector[V]]
  type Matrices[V]   =  Seq[Matrix[V]]
  type Tensors[V]    =  Seq[Tensor3D[V]]
  type Weights[V]    =  IndexedSeq[Matrix[V]]
  type LearningRate  =  PartialFunction[(Int, Double), Double]

}


trait Network[V, In, Out] extends (In => Out) with Logs with LossFuncGrapher with IllusionBreaker with Welcoming with Serializable {

  checkSettings()

  sayHi()

  val identifier: String

  val numericPrecision: String

  /** Layers of this neural network. */
  val layers: Seq[Layer]

  /** The attached loss function. */
  val lossFunction: LossFunction[V]

  /** Settings of this neural network. */
  val settings: Settings[V]

  /** The weights are a bunch of matrices. */
  val weights: Weights[V]

  /**
    * Computes output for given input `in`.
    * Alias for `net(x)` syntax.
    */
  def evaluate(in: In): Out = apply(in)

  /**
    * A focus (Ω) is used if the desired model output is not the [[neuroflow.dsl.Out]] layer, but a layer `l` in between.
    * The result is reshaped from the underworlds raw matrix to the layers `algebraicType`.
    */
  def focus[L <: Layer](l: L)(implicit cp: (Matrix[V], L) CanProduce l.algebraicType): In => l.algebraicType
  def Ω[L <: Layer](l: L)(implicit cp: (Matrix[V], L) CanProduce l.algebraicType): In => l.algebraicType = focus(l)

  override def toString: String = weights.foldLeft("")(_ + "\n---\n" + _)

}


/** A minimal constructor for a [[Network]]. */
@implicitNotFound("No `Constructor` in scope. Import your desired network or try: import neuroflow.nets.cpu.DenseNetwork._")
trait Constructor[V, +N <: Network[_, _, _]] {
  def apply(ls: Seq[Layer], loss: LossFunction[V], settings: Settings[V])(implicit weightBreeder: WeightBreeder[V]): N
}


trait FFN[V] extends Network[V, Vector[V], Vector[V]] {

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("FFNs don't support partitions. This setting has no effect.")
  }

  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Vectors[V], ys: Vectors[V]): Unit

}


trait CNN[V] extends Network[V, Tensor3D[V], Vector[V]] {

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("CNNs don't support partitions. This setting has no effect.")
  }

  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Tensors[V], ys: Vectors[V]): Unit

}


trait RNN[V] extends Network[V, Vectors[V], Vectors[V]] {

  /**
    * Takes input `xs` and trains this network against output `ys`.
    */
  def train(xs: Vectors[V], ys: Vectors[V]): Unit

}


trait DistributedTraining {

  /**
    * Triggers execution of training for nodes `ns`.
    */
  def train(ns: collection.Set[Node]): Unit

}


trait DistFFN[V] extends Network[V, Vector[V], Vector[V]] with DistributedTraining {

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("FFNs don't support partitions. This setting has no effect.")
    if (settings.batchSize.isDefined)
      warn("Setting the batch size has no effect in distributed training.")
  }

}


trait DistCNN[V] extends Network[V, Tensor3D[V], Vector[V]] with DistributedTraining {

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("CNNs don't support partitions. This setting has no effect.")
    if (settings.batchSize.isDefined)
      warn("Setting the batch size has no effect in distributed training.")
  }

}
