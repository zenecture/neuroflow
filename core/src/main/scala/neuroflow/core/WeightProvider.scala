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
  "import neuroflow.core.XXX.WeightProvider._ (where XXX = { FFN | RNN }).")
trait WeightProvider extends (Seq[Layer] => Weights)


trait BaseOps {

  /**
    * Fully connected means all `layers` are connected such that their weight matrices can
    * flow from left to right by regular matrix multiplication. The `seed` determines the initial weight value.
    */
  def fullyConnected(layers: Seq[Layer], seed: () => Double): Weights = layers.zipWithIndex.flatMap {
    case (layer, index) =>
      if (index < (layers.size - 1)) {
        val (neuronsLeft, neuronsRight) = (layer.neurons, layers(index + 1).neurons)
        val product = neuronsLeft * neuronsRight
        val initialWeights = (1 to product).map(k => seed.apply).toArray
        Some(DenseMatrix.create[Double](neuronsLeft, neuronsRight, initialWeights))
      } else None
  }

  /**
    * Gives a seed function to generate weights in range `i`.
    */
  def random(i: (Double, Double)) = () => ThreadLocalRandom.current.nextDouble(i._1, i._2)

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
      * Gives a weight provider with random weights in range `i`.
      */
    def apply(i: (Double, Double)): WeightProvider = new WeightProvider {
      def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, random(i))
    }

    implicit val randomWeights: WeightProvider = apply(-1, 1)

  }

}


object RNN extends BaseOps {

  object WeightProvider {

    /**
      * Gives a weight provider with random weights in range `i`.
      */
    def apply(i: (Double, Double)): WeightProvider = ???

    implicit val randomWeights: WeightProvider = apply(-1, 1)

  }

}
