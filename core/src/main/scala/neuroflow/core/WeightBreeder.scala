package neuroflow.core

import java.util.concurrent.ThreadLocalRandom

import breeze.linalg.DenseMatrix
import breeze.storage.Zero
import neuroflow.common.CanProduce
import neuroflow.core.Network.Weights
import neuroflow.dsl._

import scala.annotation.implicitNotFound
import scala.reflect.ClassTag

/**
  * @author bogdanski
  * @since 18.01.16
  */

/**
  * A [[WeightBreeder]] produces a weight matrix for each [[Layer]].
  */
@implicitNotFound(
  "No `WeightBreeder` in scope for type ${V}. Add your own or use a " +
    "predefined breeder: implicit val wp = neuroflow.core.WeightBreeder[${V}]")
trait WeightBreeder[V] extends (Seq[Layer] => Weights[V])

object WeightBreeder {

  /**
    * Constructs `Breeder` in scope.
    */
  def apply[V](implicit breeder: Breeder[V]): Breeder[V] = breeder

  implicit object breeder_double extends Breeder[Double]
  implicit object breeder_float extends Breeder[Float]

  trait Breeder[V] {

    /**
      * Weights drawn from normal distribution with parameters `μ` and `σ`.
      */
    def normal[N <: Network[V, _, _]](μ: Double, σ: Double)(implicit ct: ClassTag[V], zero: Zero[V], bwf: BuildWeightsFor[V, N], cp: Double CanProduce V): WeightBreeder[V] = bwf.normal(μ, σ)

    /**
      * Weights drawn from normal distribution with parameters,
      * specified individually for each weight layer using `config`,
      * which maps from index to `μ` and `σ`.
      */
    def normal[N <: Network[V, _, _]](config: Map[Int, (Double, Double)])(implicit ct: ClassTag[V], zero: Zero[V], bwf: BuildWeightsFor[V, N], cp: Double CanProduce V): WeightBreeder[V] = bwf.normal(config)

    /**
      * Weights drawn from random distribution in range `r`.
      */
    def random[N <: Network[V, _, _]](r: (Double, Double))(implicit ct: ClassTag[V], zero: Zero[V], bwf: BuildWeightsFor[V, N], cp: Double CanProduce V): WeightBreeder[V] = bwf.random(r)

    /**
      * Weights drawn from random distribution,
      * specified individually for each weight layer using `config`,
      * which maps from index to range `r`.
      */
    def random[N <: Network[V, _, _]](config: Map[Int, (Double, Double)])(implicit ct: ClassTag[V], zero: Zero[V], bwf: BuildWeightsFor[V, N], cp: Double CanProduce V): WeightBreeder[V] = bwf.random(config)

    /**
      * Weights are all equal to `seed`.
      */
    def static[N <: Network[V, _, _]](seed: V)(implicit ct: ClassTag[V], zero: Zero[V], bwf: BuildWeightsFor[V, N], cp: Double CanProduce V): WeightBreeder[V] = bwf.static(seed)

  }

  trait BuildWeightsFor[V, N <: Network[V, _, _]] {
    def normal(μ: Double, σ: Double)(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V]
    def normal(config: Map[Int, (Double, Double)])(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V]
    def random(r: (Double, Double))(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V]
    def random(config: Map[Int, (Double, Double)])(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V]
    def static(seed: V)(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V]
  }

  trait FFN[V] extends BaseOps[V] with BuildWeightsFor[V, neuroflow.core.FFN[V]] {

    def normal(μ: Double, σ: Double)(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, normalSeed(μ, σ))
    }

    def normal(config: Map[Int, (Double, Double)])(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, config.mapValues { case (μ, σ) => normalSeed(μ, σ) } )
    }

    def random(r: (Double, Double))(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, randomSeed(r))
    }

    def random(config: Map[Int, (Double, Double)])(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, config.mapValues { r => randomSeed(r) } )
    }

    def static(seed: V)(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, () => seed)
    }

  }

  trait CNN[V] extends BaseOps[V] with BuildWeightsFor[V, neuroflow.core.CNN[V]] {

    def normal(μ: Double, σ: Double)(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(layers: Seq[Layer]): Weights = convoluted(layers, normalSeed(μ, σ))
    }

    def normal(config: Map[Int, (Double, Double)])(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(layers: Seq[Layer]): Weights = convoluted(layers, config.mapValues { case (μ, σ) => normalSeed(μ, σ) } )
    }

    def random(r: (Double, Double))(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(layers: Seq[Layer]): Weights = convoluted(layers, randomSeed(r))
    }

    def random(config: Map[Int, (Double, Double)])(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(layers: Seq[Layer]): Weights = convoluted(layers, config.mapValues { r => randomSeed(r) } )
    }

    def static(seed: V)(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(layers: Seq[Layer]): Weights = convoluted(layers, () => seed)
    }

  }

  trait RNN[V] extends BaseOps[V] with BuildWeightsFor[V, neuroflow.core.RNN[V]] {

    def random(r: (Double, Double))(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(layers: Seq[Layer]): Weights = {
        val fc = fullyConnected(layers, layers.indices.map((_, randomSeed(r))).toMap)
        fc ++ recurrentEnrichment(layers, fc, layers.indices.map((_, randomSeed(r))).toMap)
      }
    }

    // TODO.
    def normal(μ: Double, σ: Double)(implicit ct: ClassTag[V], zero: Zero[V], cp: CanProduce[Double, V]): WeightBreeder[V] = ???
    def normal(config: Map[Int, (Double, Double)])(implicit ct: ClassTag[V], zero: Zero[V], cp: CanProduce[Double, V]): WeightBreeder[V] = ???
    def random(config: Map[Int, (Double, Double)])(implicit ct: ClassTag[V], zero: Zero[V], cp: CanProduce[Double, V]): WeightBreeder[V] = ???
    def static(seed: V)(implicit ct: ClassTag[V], zero: Zero[V], cp: CanProduce[Double, V]): WeightBreeder[V] = ???

  }

  trait BaseOps[V] {

    type Weights = Network.Weights[V]

    /**
      * All `layers` are fully connected such that their weight matrices can
      * flow from left to right by regular matrix multiplication.
      * The `seed` determines the initial weight values.
      */
    def fullyConnected(layers: Seq[Layer], seed: Map[Int, () => V])(implicit ct: ClassTag[V], z: Zero[V]): Weights =
      layers.dropRight(1).zipWithIndex.toArray.map {
        case (layer, index) =>
          val (neuronsLeft, neuronsRight) = (layer.neurons, layers(index + 1).neurons)
          val product = neuronsLeft * neuronsRight
          DenseMatrix.create[V](neuronsLeft, neuronsRight, Array.fill(product)(seed(index)()))
      }

    def fullyConnected(layers: Seq[Layer], seed: () => V)(implicit ct: ClassTag[V], z: Zero[V]): Weights = {
      val m = layers.indices.map((_, seed)).toMap
      fullyConnected(layers, m)
    }

    /**
      * Produces weights for convolution operator.
      */
    def convoluted(layers: Seq[Layer], seed: Map[Int, () => V])(implicit ct: ClassTag[V], z: Zero[V]): Weights =
      layers.zipWithIndex.toArray.map {
        case (Convolution(dimIn, _, field, _, filters, _), idx) =>
          val depth = dimIn._3
          val receptive = field._1 * field._2 * depth
          DenseMatrix.create[V](filters, receptive, Array.fill(receptive * filters)(seed(idx)()))
        case (Focus(inner), idx) =>
          inner match {
            case Convolution(dimIn, _, field, _, filters, _) =>
              val depth = dimIn._3
              val receptive = field._1 * field._2 * depth
              DenseMatrix.create[V](filters, receptive, Array.fill(receptive * filters)(seed(idx)()))
            case layer =>
              val (neuronsLeft, neuronsRight) = (layers(idx - 1).neurons, layer.neurons)
              val product = neuronsLeft * neuronsRight
              DenseMatrix.create[V](neuronsLeft, neuronsRight, Array.fill(product)(seed(idx)()))
          }
        case (layer, idx)  =>
          val (neuronsLeft, neuronsRight) = (layers(idx - 1).neurons, layer.neurons)
          val product = neuronsLeft * neuronsRight
          DenseMatrix.create[V](neuronsLeft, neuronsRight, Array.fill(product)(seed(idx)()))
      }

    def convoluted(layers: Seq[Layer], seed: () => V)(implicit ct: ClassTag[V], z: Zero[V]): Weights = {
      val m = layers.indices.map((_, seed)).toMap
      convoluted(layers, m)
    }

    /**
      * Enriches the given `layers` and their `weights` with recurrent LSTM connections.
      */
    def recurrentEnrichment(layers: Seq[Layer], weights: Weights, seed: Map[Int, () => V])(implicit ct: ClassTag[V], z: Zero[V]): Weights =
      weights.dropRight(1).zipWithIndex.flatMap {
        case (ws, index) =>
          val ns = layers(index + 1).neurons
          val in = (1 to 3) map { _ => DenseMatrix.create[V](ws.rows, ws.cols, (1 to ws.rows * ws.cols).map(_ => seed(index)()).toArray) }
          val cells = (1 to 4) map { _ => DenseMatrix.create[V](ns, ns, (1 to ns * ns).map(_ => seed(index)()).toArray) }
          in ++ cells
      }

    /**
      * Gives a seed function to generate weights in range `i`.
      */
    def randomSeed(i: (Double, Double))(implicit cp: Double CanProduce V): () => V = () => cp(ThreadLocalRandom.current.nextDouble(i._1, i._2))
    def normalSeed(μ: Double, σ: Double)(implicit cp: Double CanProduce V): () => V = () => cp(breeze.stats.distributions.Gaussian(μ, σ).draw())

  }

}

