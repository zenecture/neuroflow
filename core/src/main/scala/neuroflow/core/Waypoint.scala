package neuroflow.core

import neuroflow.core.Network.Weights

/**
  * @author bogdanski
  * @since 14.09.17
  */

/**
  * Performs function `action` every `nth` step.
  * The function is passed iteration count and a snapshot of the weights.
  */
case class Waypoint[V](nth: Int, action: (Int, Weights[V]) => Unit)

trait WaypointLogic[V] { self: Network[V, _, _] =>

  def waypoint(sync: () => Unit)(iteration: Int): Unit = {
    self.settings.waypoint match {
      case Some(Waypoint(nth, action)) =>
        if (iteration % nth == 0) {
          info("Waypoint ...")
          sync()
          action(iteration, self.weights)
        }
      case _ =>
    }
  }

}

object WaypointLogic {
  object NoOp extends (() => Unit) {
    def apply(): Unit = ()
  }
}

