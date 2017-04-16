package neuroflow.nets

import breeze.linalg._
import neuroflow.core.Network._
import neuroflow.core._

import scala.collection.Seq
import scala.util.Random

/**
  * Essentially, this is the same as a [[LBFGSNetwork]], but it considers the respective
  * [[Cluster]] layer as the desired model output, if it is specified. If not,
  * the [[Output]] layer is used.
  *
  * @author bogdanski
  * @since 15.04.17
  */

object LBFGSCluster {
  implicit val constructor: Constructor[LBFGSCluster] = new Constructor[LBFGSCluster] {
    def apply(ls: Seq[Layer], settings: Settings)(implicit weightProvider: WeightProvider): LBFGSCluster = {
      new LBFGSCluster(ls, settings, weightProvider(ls))
    }
  }
}

private[nets] class LBFGSCluster(override val layers: Seq[Layer],
                                 override val settings: Settings,
                                 override val weights: Weights,
                                 override val identifier: String = Random.alphanumeric.take(3).mkString)
  extends LBFGSNetwork(layers, settings, weights, identifier) {

  import neuroflow.core.Network._

  /**
    * Takes the input vector `x` to compute the output vector.
    */
  override def evaluate(x: Vector): Vector = {
    val input = DenseMatrix.create[Double](1, x.size, x.toArray)
    layers.collect {
      case c @ Cluster(_, _) => c
    }.headOption.map { cl =>
      flow(weights, input, 0, layers.indexOf(cl) - 1).map(cl.activator).toArray.toVector
    }.getOrElse {
      info("Couldn't find Cluster Layer. Using Output Layer.")
      flow(weights, input, 0, layers.size - 1).toArray.toVector
    }

  }
}
