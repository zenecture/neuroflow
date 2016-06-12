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
  * A `WeightProvider` connects the neurons within a `Layer`
  * through the `Weights` (Synapses)
  */
@implicitNotFound("No weight provider in scope. Import your desired provider or try: import neuroflow.core.WeightProvider.randomWeights")
trait WeightProvider extends (Seq[Layer] => Weights) {

  /**
    * Fully connected means all `layers` are connected such that their weight matrices can
    * flow from left to right by regular matrix multiplication. The `seed` determines the initial weight value.
    */
  def fullyConnected(layers: Seq[Layer], seed: () => Double): Weights = layers.zipWithIndex.flatMap { li =>
    val (layer, index) = li
    if (index < (layers.size - 1)) {
      val (neuronsLeft, neuronsRight) = (layer.neurons, layers(index + 1).neurons)
      val product = neuronsLeft * neuronsRight
      val initialWeights = (1 to product).map(k => seed.apply).toArray
      Some(DenseMatrix.create[Double](neuronsLeft, neuronsRight, initialWeights))
    } else None
  }

}


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

  implicit val randomWeights = new WeightProvider {
    def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, () => ThreadLocalRandom.current.nextDouble(-1, 1))
  }

}
