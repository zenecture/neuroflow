package neuroflow.core

import java.util.concurrent.ThreadLocalRandom

import breeze.linalg.DenseMatrix
import neuroflow.core.Network.Weights

import scala.annotation.implicitNotFound

/**
  * @author bogdanski
  * @since 18.01.16
  */

/**
  * A [[WeightProvider]] connects the neurons of a [[Layer]] through the weights, or synapses.
  */
@implicitNotFound(
  "No weight provider in scope. Import your desired provider or try: " +
  "import neuroflow.core.XXX.WeightProvider._ (where XXX = { FFN | RNN | CNN }).")
trait WeightProvider extends (Seq[Layer] => Weights)


trait BaseOps {

  /**
    * Fully connected means all `layers` are connected such that their weight matrices can
    * flow from left to right by regular matrix multiplication. The `seed` determines the initial weight values.
    */
  def fullyConnected(layers: Seq[Layer], seed: () => Double): Weights =
    layers.dropRight(1).zipWithIndex.toArray.map {
      case (layer, index) =>
        val (neuronsLeft, neuronsRight) = (layer.neurons, layers(index + 1).neurons)
        val product = neuronsLeft * neuronsRight
        val initialWeights = (1 to product).map(_ => seed.apply).toArray
        DenseMatrix.create[Double](neuronsLeft, neuronsRight, initialWeights)
    }

  /**
    * Enriches the given `layers` and their `weights` with recurrent LSTM connections.
    */
  def recurrentEnrichment(layers: Seq[Layer], weights: Weights, seed: () => Double): Weights =
    weights.dropRight(1).zipWithIndex.flatMap {
      case (ws, index) =>
        val ns = layers(index + 1).neurons
        val in = (1 to 3) map { _ => DenseMatrix.create[Double](ws.rows, ws.cols, (1 to ws.rows * ws.cols).map(_ => seed.apply).toArray) }
        val cells = (1 to 4) map { _ => DenseMatrix.create[Double](ns, ns, (1 to ns * ns).map(_ => seed.apply).toArray) }
        in ++ cells
    }

  /**
    * Non-convolutional layers will be fully connected, whereas [[Convolutable]]
    * layers will be connected locally.
    */
  def fullyAndLocallyConnected(layers: Seq[Layer], seed: () => Double): Weights = {


    def generateCon(c: Convolutable): Seq[DenseMatrix[Double]] = {
      Seq((1 to c.filters).map(_ => DenseMatrix.create[Double](1, c.fieldSize, (1 to c.fieldSize)
        .map(_ => seed.apply).toArray)).reduce((l, r) => DenseMatrix.vertcat(l, r)))
    }

    def generate(l: Layer, r: Layer): Seq[DenseMatrix[Double]] = {
      val initialWeights = (1 to l.neurons * r.neurons).map(_ => seed.apply).toArray
      Seq(DenseMatrix.create[Double](l.neurons, r.neurons, initialWeights))
    }


    layers.dropRight(1).zipWithIndex.toArray.flatMap {

      case (l: Convolutable, index) =>
        layers(index + 1) match {
          case r: Convolutable =>
            generateCon(r)
          case r: Layer =>
            generate(l, r)
        }

      case (l: Layer, index) =>
        layers(index + 1) match {
          case r: Convolutable =>
            generateCon(r)
          case r: Layer =>
            generate(l, r)
        }

    }

  }

  /**
    * Gives a seed function to generate weights in range `i`.
    */
  def random(i: (Double, Double)): () => Double = () => ThreadLocalRandom.current.nextDouble(i._1, i._2)

}


object FFN extends BaseOps {

  trait LowPrioWeightProviders {

    implicit val zeroWeights = new WeightProvider {
      def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, () => 0.0)
    }

    implicit val oneWeights = new WeightProvider {
      def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, () => 1.0)
    }

    implicit val minusOneWeights = new WeightProvider {
      def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, () => -1.0)
    }

  }


  object WeightProvider extends LowPrioWeightProviders {

    /**
      * Gives a weight provider with random weights in range `r`.
      */
    def apply(r: (Double, Double)): WeightProvider = new WeightProvider {
      def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, random(r))
    }

    implicit val randomWeights: WeightProvider = apply(-1, 1)

  }

}


object RNN extends BaseOps {

  object WeightProvider {

    /**
      * Gives a weight provider with random weights in range `r`.
      */
    def apply(r: (Double, Double)): WeightProvider = new WeightProvider {
      def apply(layers: Seq[Layer]): Weights = {
        val fc = fullyConnected(layers, random(r))
        fc ++ recurrentEnrichment(layers, fc, random(r))
      }
    }

    implicit val randomWeights: WeightProvider = apply(-1, 1)

  }

}


object CNN extends BaseOps {

  object WeightProvider {

    /**
      * Gives a weight provider with random weights in range `r`.
      */
    def apply(r: (Double, Double)): WeightProvider = new WeightProvider {
      def apply(layers: Seq[Layer]): Weights = {
        fullyAndLocallyConnected(layers, random(r))
      }
    }

    implicit val randomWeights: WeightProvider = apply(-1, 1)

  }

}
