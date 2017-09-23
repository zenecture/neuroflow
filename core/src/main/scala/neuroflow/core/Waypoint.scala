package neuroflow.core

import neuroflow.core.Network.Weights

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future

/**
  * @author bogdanski
  * @since 14.09.17
  */

/**
  * Performs `action` on every `nth` iteration during training.
  */
case class Waypoint[V](nth: Int, action: Weights[V] => Unit)

trait WaypointLogic[V] { self: Network[V, _, _] =>

  def waypoint(iteration: Int): Unit = Future {
    self.settings.waypoint match {
      case Some(Waypoint(nth, action)) =>
        if (iteration % nth == 0) {
          info("Waypoint ...")
          action(self.weights)
        }
      case _ =>
    }
  }

}
