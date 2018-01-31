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
  * A [[WeightProvider]] connects the neurons of a [[Layer]] through the weights, or synapses.
  */
@implicitNotFound(
  "No `WeightProvider` in scope for type ${V}. Add your own or use a predefined provider: " +
  "neuroflow.core.WeightProvider.X[${V}] (where X = { CNN | FFN | RNN })")
trait WeightProvider[V] extends (Seq[Layer] => Weights[V])


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

}

object WeightProvider {

  /**
    * Gives a seed function to generate weights in range `i`.
    */
  def randomSeed[V](i: (Double, Double))(implicit cp: Double CanProduce V): () => V = () => cp(ThreadLocalRandom.current.nextDouble(i._1, i._2))
  def normalSeed[V](μ: Double, σ: Double)(implicit cp: Double CanProduce V): () => V = () => cp(breeze.stats.distributions.Gaussian(μ, σ).draw())

  trait FFN[V] extends BaseOps[V] {

    /**
      * Gives a weight provider with weights drawn from normal distribution.
      */
    def normal(μ: Double, σ: Double)(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightProvider[V] = new WeightProvider[V] {
      def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, normalSeed(μ, σ))
    }

    /**
      * Gives a weight provider with weights drawn from normal distribution.
      * The parameters are specified for each layer individually using `config` (0-index based).
      */
    def normal(config: Map[Int, (Double, Double)])(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightProvider[V] = new WeightProvider[V] {
      def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, config.mapValues { case (μ, σ) => normalSeed(μ, σ) } )
    }

    /**
      * Gives a weight provider with random weights in range `r`.
      */
    def random(r: (Double, Double))(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightProvider[V] = new WeightProvider[V] {
      def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, randomSeed(r))
    }

    /**
      * Gives a weight provider with random weights.
      * The ranges are specified individually for each layer in `config` (0-index based).
      */
    def random(config: Map[Int, (Double, Double)])(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightProvider[V] = new WeightProvider[V] {
      def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, config.mapValues { r => randomSeed(r) } )
    }

    def static(seed: V)(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightProvider[V] = new WeightProvider[V] {
      def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, () => seed)
    }

  }

  trait CNN[V] extends BaseOps[V] {

    /**
      * Gives a weight provider with weights drawn from normal distribution.
      */
    def normal(μ: Double, σ: Double)(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightProvider[V] = new WeightProvider[V] {
      def apply(layers: Seq[Layer]): Weights = convoluted(layers, normalSeed(μ, σ))
    }

    /**
      * Gives a weight provider with weights drawn from normal distribution.
      * The parameters μ, σ are specified for each layer individually using `config` (0-index based).
      */
    def normal(config: Map[Int, (Double, Double)])(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightProvider[V] = new WeightProvider[V] {
      def apply(layers: Seq[Layer]): Weights = convoluted(layers, config.mapValues { case (μ, σ) => normalSeed(μ, σ) } )
    }

    /**
      * Gives a weight provider with random weights in range `r`.
      */
    def random(r: (Double, Double))(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightProvider[V] = new WeightProvider[V] {
      def apply(layers: Seq[Layer]): Weights = convoluted(layers, randomSeed(r))
    }

    /**
      * Gives a weight provider with random weights.
      * The ranges are specified individually for each layer in `config` (0-index based).
      */
    def random(config: Map[Int, (Double, Double)])(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightProvider[V] = new WeightProvider[V] {
      def apply(layers: Seq[Layer]): Weights = convoluted(layers, config.mapValues { r => randomSeed(r) } )
    }

    def static(seed: V)(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightProvider[V] = new WeightProvider[V] {
      def apply(layers: Seq[Layer]): Weights = convoluted(layers, () => seed)
    }

  }

  trait RNN[V] extends BaseOps[V] {

    /**
      * Gives a weight provider with random weights in range `r`.
      */
    def random(r: (Double, Double))(implicit ct: ClassTag[V], zero: Zero[V], cp: Double CanProduce V): WeightProvider[V] = new WeightProvider[V] {
      def apply(layers: Seq[Layer]): Weights = {
        val fc = fullyConnected(layers, layers.indices.map((_, randomSeed(r))).toMap)
        fc ++ recurrentEnrichment(layers, fc, layers.indices.map((_, randomSeed(r))).toMap)
      }
    }

  }

  implicit object ffn_double extends FFN[Double]
  implicit object ffn_float extends FFN[Float]

  implicit object cnn_double extends CNN[Double]
  implicit object cnn_float extends CNN[Float]

  implicit object rnn_double extends RNN[Double]
  implicit object rnn_float extends RNN[Float]

  def FFN[V](implicit impl: FFN[V]) = impl
  def CNN[V](implicit impl: CNN[V]) = impl
  def RNN[V](implicit impl: RNN[V]) = impl

}
