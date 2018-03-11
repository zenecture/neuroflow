package neuroflow.core

import java.util.concurrent.ThreadLocalRandom

import breeze.linalg.DenseMatrix
import breeze.storage.Zero
import neuroflow.common.{CanProduce, Logs}
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
  "\nNo `WeightBreeder` in scope for type ${V}. Example to add a random " +
    "breeder:\nimplicit val breeder = neuroflow.core.WeightBreeder[${V}].random(0, 1)")
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
      * Weights drawn from normal distribution with parameters
      * specified individually for each layer using `config`, which maps
      * from index to `μ` and `σ`. The index is zero-based, e. g:
      *   Layout = 0 :: 1 :: 2 :: Loss
      */
    def normal[N <: Network[V, _, _]](config: Map[Int, (Double, Double)])(implicit ct: ClassTag[V], zero: Zero[V], bwf: BuildWeightsFor[V, N], cp: Double CanProduce V): WeightBreeder[V] = bwf.normal(config)

    /**
      * Weights drawn from random distribution in range `r`.
      */
    def random[N <: Network[V, _, _]](r: (Double, Double))(implicit ct: ClassTag[V], zero: Zero[V], bwf: BuildWeightsFor[V, N], cp: Double CanProduce V): WeightBreeder[V] = bwf.random(r)

    /**
      * Weights drawn from random distribution with range
      * specified individually for each layer using `config`, which maps
      * from index to range. The index is zero-based, e. g:
      *   Layout = 0 :: 1 :: 2 :: Loss
      */
    def random[N <: Network[V, _, _]](config: Map[Int, (Double, Double)])(implicit ct: ClassTag[V], zero: Zero[V], bwf: BuildWeightsFor[V, N], cp: Double CanProduce V): WeightBreeder[V] = bwf.random(config)

    /**
      * Weights are all equal to `seed`.
      */
    def static[N <: Network[V, _, _]](seed: V)(implicit ct: ClassTag[V], zero: Zero[V], bwf: BuildWeightsFor[V, N], cp: Double CanProduce V): WeightBreeder[V] = bwf.static(seed)

    /**
      * Weights are the same for each layer, specified individually using `config`,
      * which maps from index to seed. The index is zero-based, e. g:
      *   Layout = 0 :: 1 :: 2 :: Loss
      */
    def static[N <: Network[V, _, _]](config: Map[Int, V])(implicit ct: ClassTag[V], zero: Zero[V], bwf: BuildWeightsFor[V, N], cp: Double CanProduce V): WeightBreeder[V] = bwf.static(config)

  }



  trait BuildWeightsFor[V, N <: Network[V, _, _]] extends BaseOps[V] {

    def normal(μ: Double, σ: Double)(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(layers: Seq[Layer]): Weights = traverseAndBuild(layers, normalSeed(μ, σ))
    }

    def normal(config: Map[Int, (Double, Double)])(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(layers: Seq[Layer]): Weights = traverseAndBuild(layers, config.mapValues { case (μ, σ) => normalSeed(μ, σ) } )
    }

    def random(r: (Double, Double))(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(layers: Seq[Layer]): Weights = traverseAndBuild(layers, randomSeed(r))
    }

    def random(config: Map[Int, (Double, Double)])(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(layers: Seq[Layer]): Weights = traverseAndBuild(layers, config.mapValues { r => randomSeed(r) } )
    }

    def static(seed: V)(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(layers: Seq[Layer]): Weights = traverseAndBuild(layers, () => seed)
    }

    def static(config: Map[Int, V])(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(layers: Seq[Layer]): Weights = traverseAndBuild(layers, config.mapValues(seed => () => seed))
    }

  }



  trait BaseOps[V] extends Logs {

    type Weights = Network.Weights[V]

    def traverseAndBuild(layers: Seq[Layer], seed: () => V)(implicit ct: ClassTag[V], z: Zero[V]): Weights = {
      val m = layers.indices.map((_, seed)).toMap
      traverseAndBuild(layers, m)
    }

    def traverseAndBuild(layers: Seq[Layer], seed: Map[Int, () => V])(implicit ct: ClassTag[V], z: Zero[V]): Weights = {

      def build(l: Layer, idx: Int): Option[DenseMatrix[V]] = l match {

        case v: Vector[_]      =>
          if (seed.isDefinedAt(idx))
            warn(s"A plain Vector layer does not have weights. Ignoring config index: $idx.")
          None

        case c: Convolution[_] =>
          val depth = c.dimIn._3
          val receptive = c.field._1 * c.field._2 * depth
          val w = DenseMatrix.create[V](c.filters, receptive, Array.fill(receptive * c.filters)(seed(idx)()))
          Some(w)

        case d: Dense[_]       =>
          val (neuronsLeft, neuronsRight) = (layers(idx - 1).neurons, d.neurons)
          val product = neuronsLeft * neuronsRight
          val w = DenseMatrix.create[V](neuronsLeft, neuronsRight, Array.fill(product)(seed(idx)()))
          Some(w)

        case x =>
          warn(s"Unknown layer type in weight breeding stage: $x.")
          None

      }

      layers.toArray.zipWithIndex.flatMap(i => build(i._1, i._2))

    }

    /**
      * Gives a seed function to generate weights in range `i`.
      */
    def randomSeed(i: (Double, Double))(implicit cp: Double CanProduce V): () => V = () => cp(ThreadLocalRandom.current.nextDouble(i._1, i._2))
    def normalSeed(μ: Double, σ: Double)(implicit cp: Double CanProduce V): () => V = () => cp(breeze.stats.distributions.Gaussian(μ, σ).draw())

  }



  trait FFN[V] extends BuildWeightsFor[V, neuroflow.core.FFN[V]]

  trait CNN[V] extends BuildWeightsFor[V, neuroflow.core.CNN[V]]

  trait RNN[V] extends BuildWeightsFor[V, neuroflow.core.RNN[V]] {

    /**
      * Enriches the given `layers` and their `weights` with recurrent LSTM connections.
      */
    def recurrentEnrichment(layers: Seq[Layer], weights: Weights, seed: Map[Int, () => V])(implicit ct: ClassTag[V], z: Zero[V]): Seq[DenseMatrix[V]] =
      weights.dropRight(1).zipWithIndex.flatMap {
        case (ws, index) =>
          val ns = layers(index + 1).neurons
          val in = (1 to 3) map { _ => DenseMatrix.create[V](ws.rows, ws.cols, (1 to ws.rows * ws.cols).map(_ => seed(index)()).toArray) }
          val cells = (1 to 4) map { _ => DenseMatrix.create[V](ns, ns, (1 to ns * ns).map(_ => seed(index)()).toArray) }
          in ++ cells
      }

    override def random(r: (Double, Double))(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(layers: Seq[Layer]): Weights = {
        val fc = traverseAndBuild(layers, layers.indices.map((_, randomSeed(r))).toMap)
        fc ++ recurrentEnrichment(layers, fc, layers.indices.map((_, randomSeed(r))).toMap)
      }
    }

  }


}

