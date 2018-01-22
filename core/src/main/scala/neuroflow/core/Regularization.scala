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
  * The KeepBest regularization strategy takes weights, which led to the least error during training.
  * In particular, this is useful for nets whose gradients oscillate heavily during training.
  */
case object KeepBest extends Regularization

trait KeepBestLogic[V] { self: Network[V, _, _] =>

  import scala.concurrent.ExecutionContext.Implicits.global

  private var bestErr = Double.PositiveInfinity
  private var bestWs: Weights[V] = self.weights

  def keepBest(error: Double): Unit = {
    if (error < bestErr) Future {
      bestErr = error
      bestWs = self.weights.map(_.copy)
    }
  }

  def takeBest(): Unit = self.settings.regularization match {
    case Some(KeepBest) =>
      info(f"Applying KeepBest strategy. Best test error so far: $bestErr%.6g.")
      weights.zip(bestWs).foreach {
        case (a, b) => a := b
      }
    case _ =>
  }

}
