package neuroflow.nets

import neuroflow.core.Network.Weights
import neuroflow.core._

import scala.collection.Seq
import scala.util.Random

/**
  * Unsupervised Auto Encoder using a [[LBFGSNetwork]].
  *
  * @author bogdanski
  * @since 16.04.17
  */

object AutoEncoder {
  implicit val constructor: Constructor[AutoEncoder] = new Constructor[AutoEncoder] {
    def apply(ls: Seq[Layer], settings: Settings)(implicit weightProvider: WeightProvider): AutoEncoder = {
      AutoEncoder(ls, settings, weightProvider(ls))
    }
  }
}

private[nets] case class AutoEncoder(layers: Seq[Layer],
                                     settings: Settings,
                                     weights: Weights,
                                     identifier: String = Random.alphanumeric.take(3).mkString)
  extends FeedForwardNetwork with UnsupervisedTraining {

  import neuroflow.core.Network._

  private val net = new DefaultNetwork(layers, settings, weights, identifier) {
    override def sayHi(): Unit = ()
  }

  /**
    * Takes the input vector `x` to compute the output vector.
    */
  def evaluate(x: Vector): Vector = net.evaluate(x)

  /**
    * Takes a sequence of input vectors `xs` and trains this
    * network using the unsupervised learning strategy.
    */
  def train(xs: Seq[Vector]): Unit = net.train(xs, xs)

}
