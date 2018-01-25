package neuroflow.core

import neuroflow.core.Network._

import scala.concurrent.Future


/**
  * @author bogdanski
  * @since 20.01.16
  */


/**
  * Marker trait for regulators
  */
trait Regularization extends Serializable


/**
  * Keep weights, which had the smallest loss during training.
  * In particular, this is useful for RNNs, as they can oscillate during training.
  */
case object KeepBest extends Regularization

trait KeepBestLogic[V] { self: Network[V, _, _] =>

  import scala.concurrent.ExecutionContext.Implicits.global

  private var bestLoss = Double.PositiveInfinity
  private var bestWs: Weights[V] = self.weights

  def keepBest(loss: Double): Unit = {
    if (loss < bestLoss) Future {
      bestLoss = loss
      bestWs = self.weights.map(_.copy)
    }
  }

  def takeBest(): Unit = self.settings.regularization match {
    case Some(KeepBest) =>
      info(f"Applying KeepBest strategy. Best loss so far: $bestLoss%.6g.")
      weights.zip(bestWs).foreach {
        case (a, b) => a := b
      }
    case _ =>
  }

}
