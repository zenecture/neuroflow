package neuroflow.nets

import breeze.linalg._
import neuroflow.core.Network._
import neuroflow.core._

import scala.util.Random

/**
  * Essentially, this is the same as a [[DynamicNetwork]], but it considers the respective
  * [[Cluster]] layer as the desired model output, if it is specified. If not,
  * the [[Output]] layer is used.
  *
  * @author bogdanski
  * @since 22.04.17
  */
object DynamicCluster {
  implicit val constructor: Constructor[DynamicCluster] = new Constructor[DynamicCluster] {
    def apply(ls: Seq[Layer], settings: Settings)(implicit weightProvider: WeightProvider): DynamicCluster = {
      new DynamicCluster(ls, settings, weightProvider(ls))
    }
  }
}

private[nets] class DynamicCluster(override val layers: Seq[Layer],
                                   override val settings: Settings,
                                   override val weights: Weights,
                                   override val identifier: String = Random.alphanumeric.take(3).mkString)
  extends DynamicNetwork(layers, settings, weights, identifier) {

  import neuroflow.core.Network._

  /**
    * Takes the input vector `x` to compute the output vector.
    */
  override def evaluate(x: Vector): Vector = {
    val input = DenseMatrix.create[Double](1, x.size, x.toArray)
    layers.collect {
      case c @ Cluster(_, _) => c
    }.headOption.map { cl =>
      flow(input, 0, layers.indexOf(cl) - 1).map(cl.activator).toArray.toVector
    }.getOrElse {
      info("Couldn't find Cluster Layer. Using Output Layer.")
      flow(input, 0, layers.size - 1).toArray.toVector
    }
  }

}
