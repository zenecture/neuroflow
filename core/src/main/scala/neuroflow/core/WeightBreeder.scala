package neuroflow.core

import java.util.concurrent.ThreadLocalRandom

import breeze.linalg.DenseMatrix
import breeze.storage.Zero
import neuroflow.common.Logs
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
trait WeightBreeder[V] extends (Seq[Layer[V]] => Weights[V])

object WeightBreeder {

  /**
    * Gives `init` in scope to construct a `WeightBreeder`.
    */
  def apply[V](implicit init: Initializer[V]): Initializer[V] = init


  trait Initializer[V] extends BaseOps[V] {

    /**
      * Weights drawn from normal distribution with parameters `μ` and `σ`.
      */
    def normal[N <: Network[V, _, _]](μ: V, σ: V)(implicit ct: ClassTag[V], zero: Zero[V], cp_dv: Double CanProduce V, cp_vd: V CanProduce Double): WeightBreeder[V] =
      new WeightBreeder[V] {
        def apply(layers: Seq[Layer[V]]): Weights = traverseAndBuild(layers, normalSeed(μ, σ))
      }

    /**
      * Weights drawn from normal distribution with parameters
      * specified individually for each layer using `config`, which maps
      * from index to `μ` and `σ`. The index is zero-based, e. g:
      *   Layout = 0 :: 1 :: 2 :: Loss
      */
    def normal[N <: Network[V, _, _]](config: Map[Int, (V, V)])(implicit ct: ClassTag[V], zero: Zero[V], cp_dv: Double CanProduce V, cp_vd: V CanProduce Double): WeightBreeder[V] =
      new WeightBreeder[V] {
        def apply(layers: Seq[Layer[V]]): Weights = traverseAndBuild(layers, config.mapValues { case (μ, σ) => normalSeed(μ, σ) })
      }

    /**
      * Weights drawn from random distribution in range `r`.
      */
    def random[N <: Network[V, _, _]](r: (V, V))(implicit ct: ClassTag[V], zero: Zero[V], cp_dv: Double CanProduce V, cp_vd: V CanProduce Double): WeightBreeder[V] =
      new WeightBreeder[V] {
        def apply(layers: Seq[Layer[V]]): Weights = traverseAndBuild(layers, randomSeed(r))
      }

    /**
      * Weights drawn from random distribution with range
      * specified individually for each layer using `config`, which maps
      * from index to range. The index is zero-based, e. g:
      *   Layout = 0 :: 1 :: 2 :: Loss
      */
    def random[N <: Network[V, _, _]](config: Map[Int, (V, V)])(implicit ct: ClassTag[V], zero: Zero[V], cp_dv: Double CanProduce V, cp_vd: V CanProduce Double): WeightBreeder[V] =
      new WeightBreeder[V] {
        def apply(layers: Seq[Layer[V]]): Weights = traverseAndBuild(layers, config.mapValues { r => randomSeed(r) } )
      }

    /**
      * Weights are all equal to `seed`.
      */
    def static[N <: Network[V, _, _]](seed: V)(implicit ct: ClassTag[V], zero: Zero[V], cp_dv: Double CanProduce V, cp_vd: V CanProduce Double): WeightBreeder[V] =
      new WeightBreeder[V] {
        def apply(layers: Seq[Layer[V]]): Weights = traverseAndBuild(layers, () => seed)
      }

    /**
      * Weights are the same for each layer, specified individually using `config`,
      * which maps from index to seed. The index is zero-based, e. g:
      *   Layout = 0 :: 1 :: 2 :: Loss
      */
    def static[N <: Network[V, _, _]](config: Map[Int, V])(implicit ct: ClassTag[V], zero: Zero[V], cp_dv: Double CanProduce V, cp_vd: V CanProduce Double): WeightBreeder[V] =
      new WeightBreeder[V] {
        def apply(layers: Seq[Layer[V]]): Weights = traverseAndBuild(layers, config.mapValues(seed => () => seed))
      }

  }



  trait BaseOps[V] extends Logs {

    type Weights = Network.Weights[V]

    def traverseAndBuild(layers: Seq[Layer[V]], seed: () => V)(implicit ct: ClassTag[V], z: Zero[V]): Weights = {
      val m = layers.indices.map((_, seed)).toMap
      traverseAndBuild(layers, m)
    }

    def traverseAndBuild(layers: Seq[Layer[V]], seed: Map[Int, () => V])(implicit ct: ClassTag[V], z: Zero[V]): Weights = {

      def build(l: Layer[V], idx: Int): Option[DenseMatrix[V]] = l match {

        case v: Vector[_]      =>
          if (seed.isDefinedAt(idx))
            debug(s"A plain Vector layer does not have weights. Ignoring config index: $idx.")
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
    def randomSeed(i: (V, V))(implicit cp_dv: Double CanProduce V, cp_vd: V CanProduce Double): () => V = () => cp_dv(ThreadLocalRandom.current.nextDouble(cp_vd(i._1), cp_vd(i._2)))
    def normalSeed(μ: V, σ: V)(implicit cp_dv: Double CanProduce V, cp_vd: V CanProduce Double): () => V = () => cp_dv(breeze.stats.distributions.Gaussian(cp_vd(μ), cp_vd(σ)).draw())

  }



  trait RNN_Initializer[V] extends Initializer[V] {


    /**
      * Weights drawn from random distribution in range `r`.
      */
    override def random[N <: Network[V, _, _]](r: (V, V))(implicit ct: ClassTag[V], zero: Zero[V], cp_dv: CanProduce[Double, V], cp_vd: CanProduce[V, Double]): WeightBreeder[V] =
      new WeightBreeder[V] {
        def apply(layers: Seq[Layer[V]]): Weights = {
          val fc = traverseAndBuild(layers, layers.indices.map((_, randomSeed(r))).toMap)
          fc ++ recurrentEnrichment(layers, fc, layers.indices.map((_, randomSeed(r))).toMap)
        }
      }

    /**
      * Enriches the given `layers` and their `weights` with recurrent LSTM connections.
      */
    private def recurrentEnrichment(layers: Seq[Layer[V]], weights: Weights, seed: Map[Int, () => V])(implicit ct: ClassTag[V], z: Zero[V]): Seq[DenseMatrix[V]] =
      weights.dropRight(1).zipWithIndex.flatMap {
        case (ws, index) =>
          val ns = layers(index + 1).neurons
          val in = (1 to 3) map { _ => DenseMatrix.create[V](ws.rows, ws.cols, (1 to ws.rows * ws.cols).map(_ => seed(index)()).toArray) }
          val cells = (1 to 4) map { _ => DenseMatrix.create[V](ns, ns, (1 to ns * ns).map(_ => seed(index)()).toArray) }
          in ++ cells
      }

  }


}

