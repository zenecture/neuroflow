package neuroflow.core

import breeze.linalg.{DenseMatrix, DenseVector}
import neuroflow.common._
import neuroflow.core.Network._
import neuroflow.dsl.{Vector => _, _}

import scala.annotation.implicitNotFound
import scala.collection._
import scala.util.Try

/**
  * @author bogdanski
  * @since 03.01.16
  */

object Network extends Lexicon {

  /**
    * Constructs a new [[Network]] with the respective [[Constructor]] in scope.
    * Additionally, it gives `evidence` that the graph `L` is a valid composition.
    */
  def apply[V, L <: Layout, N <: Network[_, _, _]]
                                     (layout: L,
                                      settings: Settings[V] = Settings[V]())
                                     (implicit
                                      weightBreeder: WeightBreeder[V],
                                      constructor: Constructor[V, N],
                                      evidence: L IsValidLayoutFor N): N = {
    val (layers, loss) = (layout.toSeq, layout.toLossFunction[V])
    constructor(layers, loss, settings)
  }

}


trait Lexicon {

  type Vector[V]        =  DenseVector[V]
  type Matrix[V]        =  DenseMatrix[V]
  type Vectors[V]       =  Seq[Vector[V]]
  type Matrices[V]      =  Seq[Matrix[V]]
  type Tensors[V]       =  Seq[Tensor3D[V]]
  type Weights[V]       =  IndexedSeq[Matrix[V]]
  type LearningRate[V]  =  PartialFunction[(Int, V), V]

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
    */
  def apply(in: In): Out

  /**
    * Computes output for given input `in`.
    * Alias for `net(x)` syntax.
    */
  def evaluate(in: In): Out = apply(in)

  /**
    * Computes output for given inputs `in`
    * using efficient batch mode.
    */
  def batchApply(xs: Seq[In]): Seq[Out]

  /**
    * A focus (Ω) is used if the desired model output is not the [[neuroflow.dsl.Out]] layer, but a layer `l` in between.
    * The result is reshaped from the underworlds raw matrix to the layers `algebraicType`.
    */
  def focus[L <: Layer](l: L)(implicit cp: (Matrix[V], L) CanProduce l.algebraicType): In => l.algebraicType
  def Ω[L <: Layer](l: L)(implicit cp: (Matrix[V], L) CanProduce l.algebraicType): In => l.algebraicType = focus(l)

  override def toString: String = weights.foldLeft("")(_ + "\n---\n" + _)

}


@implicitNotFound(
  "No constructor in scope for ${N}. Import your desired network or try: " +
  "import neuroflow.nets.cpu.DenseNetwork._"
)
trait Constructor[V, +N <: Network[_, _, _]] {
  def apply(ls: Seq[Layer], loss: LossFunction[V], settings: Settings[V])(implicit weightBreeder: WeightBreeder[V]): N
}


trait Training[V, M[_], N[_]] {

  case class Run(startTime: Long, endTime: Long, iterations: Int)

  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: M[V], ys: N[V]): Try[Run]

}

trait FFN[V] extends Network[V, Vector[V], Vector[V]] with Training[V, Vectors, Vectors] {

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("FFNs don't support partitions. This has no effect.")
  }

}


trait CNN[V] extends Network[V, Tensor3D[V], Vector[V]] with Training[V, Tensors, Vectors] {

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("CNNs don't support partitions. This has no effect.")
  }

}


trait RNN[V] extends Network[V, Vectors[V], Vectors[V]] with Training[V, Vectors, Vectors]

