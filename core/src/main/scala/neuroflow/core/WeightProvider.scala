package neuroflow.core

import java.util.concurrent.ThreadLocalRandom

import breeze.linalg.DenseMatrix
import breeze.storage.Zero
import neuroflow.core.Network.Weights

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
  "No weight provider in scope. Add your own or import a predefined provider: " +
  "import neuroflow.core.WeightProvider.Y.X._ (where X = { FFN | RNN | CNN }, Y = { Double | Float }).")
trait WeightProvider[V] extends (Seq[Layer] => Weights[V])


trait BaseOps[V] {

  type Weights = Network.Weights[V]

  /**
    * All `layers` are fully connected such that their weight matrices can
    * flow from left to right by regular matrix multiplication.
    * The `seed` determines the initial weight values.
    */
  def fullyConnected(layers: Seq[Layer], seed: () => V)(implicit ct: ClassTag[V], z: Zero[V]): Weights =
    layers.dropRight(1).zipWithIndex.toArray.map {
      case (layer, index) =>
        val (neuronsLeft, neuronsRight) = (layer.neurons, layers(index + 1).neurons)
        val product = neuronsLeft * neuronsRight
        DenseMatrix.create[V](neuronsLeft, neuronsRight, Array.fill(product)(seed()))
    }

  /**
    * Convolutional layers produce im2col-ready weight matrices.
    */
  def convoluted(layers: Seq[Layer], seed: () => V)(implicit ct: ClassTag[V], z: Zero[V]): Weights =
    layers.zipWithIndex.toArray.map {
      case (Convolution(dimIn, filter, filters, _, _), idx) =>
        val depth = dimIn._3
        val field = filter._1 * filter._2 * depth
        DenseMatrix.create[V](filters, field, Array.fill(field * filters)(seed()))
      case (Focus(inner), idx) =>
        inner match {
          case Convolution(dimIn, filter, filters, _, _) =>
            val depth = dimIn._3
            val field = filter._1 * filter._2 * depth
            DenseMatrix.create[V](filters, field, Array.fill(field * filters)(seed()))
          case layer =>
            val (neuronsLeft, neuronsRight) = (layers(idx - 1).neurons, layer.neurons)
            val product = neuronsLeft * neuronsRight
            DenseMatrix.create[V](neuronsLeft, neuronsRight, Array.fill(product)(seed()))
        }
      case (layer, idx)  =>
        val (neuronsLeft, neuronsRight) = (layers(idx - 1).neurons, layer.neurons)
        val product = neuronsLeft * neuronsRight
        DenseMatrix.create[V](neuronsLeft, neuronsRight, Array.fill(product)(seed()))
    }

  /**
    * Enriches the given `layers` and their `weights` with recurrent LSTM connections.
    */
  def recurrentEnrichment(layers: Seq[Layer], weights: Weights, seed: () => V)(implicit ct: ClassTag[V], z: Zero[V]): Weights =
    weights.dropRight(1).zipWithIndex.flatMap {
      case (ws, index) =>
        val ns = layers(index + 1).neurons
        val in = (1 to 3) map { _ => DenseMatrix.create[V](ws.rows, ws.cols, (1 to ws.rows * ws.cols).map(_ => seed.apply).toArray) }
        val cells = (1 to 4) map { _ => DenseMatrix.create[V](ns, ns, (1 to ns * ns).map(_ => seed.apply).toArray) }
        in ++ cells
    }

  /**
    * Gives a seed function to generate weights in range `i`.
    */
  def randomD(i: (Double, Double)): () => Double = () => ThreadLocalRandom.current.nextDouble(i._1, i._2)
  def randomF(i: (Double, Double)): () => Float  = () => ThreadLocalRandom.current.nextDouble(i._1, i._2).toFloat

}

object WeightProvider {

  object Double {

    object FFN extends BaseOps[Double] {

      implicit val zeroWeights = new WeightProvider[Double] {
        def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, () => 0.0)
      }

      implicit val oneWeights = new WeightProvider[Double] {
        def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, () => 1.0)
      }

      implicit val minusOneWeights = new WeightProvider[Double] {
        def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, () => -1.0)
      }

      implicit val randomWeights: WeightProvider[Double] = apply(-1, 1)

      /**
        * Gives a weight provider with random weights in range `r`.
        */
      def apply(r: (Double, Double)): WeightProvider[Double] = new WeightProvider[Double] {
        def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, randomD(r))
      }

      def static(seed: Double): WeightProvider[Double] = new WeightProvider[Double] {
        def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, () => seed)
      }

    }

    object CNN extends BaseOps[Double] {

      implicit val zeroWeights = new WeightProvider[Double] {
        def apply(layers: Seq[Layer]): Weights = convoluted(layers, () => 0.0)
      }

      implicit val oneWeights = new WeightProvider[Double] {
        def apply(layers: Seq[Layer]): Weights = convoluted(layers, () => 1.0)
      }

      implicit val minusOneWeights = new WeightProvider[Double] {
        def apply(layers: Seq[Layer]): Weights = convoluted(layers, () => -1.0)
      }

      implicit val randomWeights: WeightProvider[Double] = apply(-1, 1)

      /**
        * Gives a weight provider with random weights in range `r`.
        */
      def apply(r: (Double, Double)): WeightProvider[Double] = new WeightProvider[Double] {
        def apply(layers: Seq[Layer]): Weights = convoluted(layers, randomD(r))
      }

      def static(seed: Double): WeightProvider[Double] = new WeightProvider[Double] {
        def apply(layers: Seq[Layer]): Weights = convoluted(layers, () => seed)
      }

    }

    object RNN extends BaseOps[Double] {

      /**
        * Gives a weight provider with random weights in range `r`.
        */
      def apply(r: (Double, Double)): WeightProvider[Double] = new WeightProvider[Double] {
        def apply(layers: Seq[Layer]): Weights = {
          val fc = fullyConnected(layers, randomD(r))
          fc ++ recurrentEnrichment(layers, fc, randomD(r))
        }
      }

      implicit val randomWeights: WeightProvider[Double] = apply(-1, 1)

    }

  }

  object Float {

    object FFN extends BaseOps[Float] {

      implicit val zeroWeights = new WeightProvider[Float] {
        def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, () => 0.0f)
      }

      implicit val oneWeights = new WeightProvider[Float] {
        def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, () => 1.0f)
      }

      implicit val minusOneWeights = new WeightProvider[Float] {
        def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, () => -1.0f)
      }

      implicit val randomWeights: WeightProvider[Float] = apply(-1, 1)

      /**
        * Gives a weight provider with random weights in range `r`.
        */
      def apply(r: (Double, Double)): WeightProvider[Float] = new WeightProvider[Float] {
        def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, randomF(r))
      }

      def static(seed: Float): WeightProvider[Float] = new WeightProvider[Float] {
        def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, () => seed)
      }

    }

    object CNN extends BaseOps[Float] {

      implicit val zeroWeights = new WeightProvider[Float] {
        def apply(layers: Seq[Layer]): Weights = convoluted(layers, () => 0.0f)
      }

      implicit val oneWeights = new WeightProvider[Float] {
        def apply(layers: Seq[Layer]): Weights = convoluted(layers, () => 1.0f)
      }

      implicit val minusOneWeights = new WeightProvider[Float] {
        def apply(layers: Seq[Layer]): Weights = convoluted(layers, () => -1.0f)
      }

      implicit val randomWeights: WeightProvider[Float] = apply(-1, 1)

      /**
        * Gives a weight provider with random weights in range `r`.
        */
      def apply(r: (Double, Double)): WeightProvider[Float] = new WeightProvider[Float] {
        def apply(layers: Seq[Layer]): Weights = convoluted(layers, randomF(r))
      }

      def static(seed: Float): WeightProvider[Float] = new WeightProvider[Float] {
        def apply(layers: Seq[Layer]): Weights = convoluted(layers, () => seed)
      }

    }

    object RNN extends BaseOps[Float] {

      /**
        * Gives a weight provider with random weights in range `r`.
        */
      def apply(r: (Double, Double)): WeightProvider[Float] = new WeightProvider[Float] {
        def apply(layers: Seq[Layer]): Weights = {
          val fc = fullyConnected(layers, randomF(r))
          fc ++ recurrentEnrichment(layers, fc, randomF(r))
        }
      }

      implicit val randomWeights: WeightProvider[Float] = apply(-1, 1)

    }

  }

}
