package neuroflow.core

import java.util.concurrent.ThreadLocalRandom

import breeze.linalg.DenseMatrix
import neuroflow.core.Network.Weights

/**
  * @author bogdanski
  * @since 18.01.16
  */

trait WeightProvider extends (Seq[Layer] => Weights)

object WeightProvider {
  implicit val randomWeights = new WeightProvider {
    def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, () => ThreadLocalRandom.current.nextDouble(-1, 1))
  }

  implicit val zeroWeights = new WeightProvider {
    def apply(layers: Seq[Layer]): Weights = fullyConnected(layers, () => 0.0)
  }

  private def fullyConnected(layers: Seq[Layer], seed: () => Double): Weights = layers.zipWithIndex.flatMap { li =>
    val (layer, index) = li
    if (index < (layers.size - 1)) {
      val (neuronsLeft, neuronsRight) = (layer.neurons, layers(index + 1).neurons)
      val product = neuronsLeft * neuronsRight
      val initialWeights = (1 to product).map(k => seed.apply).toArray
      Some(DenseMatrix.create[Double](neuronsLeft, neuronsRight, initialWeights))
    } else None
  }.toList
}