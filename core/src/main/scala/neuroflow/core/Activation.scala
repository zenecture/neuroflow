package neuroflow.core

import breeze.numerics._

/**
  * @author bogdanski
  * @since 03.01.16
  */

/**
  * Label for neurons in the network performing a function on their synapses
  * (thus all, except input neurons)
  */
trait HasActivator[N] {
  val activator: Activator[N]
}

/**
  * The activator (or transport) function with its derivative, generic in `N`.
  */
trait Activator[N] extends (N => N) {
  def apply(x: N): N
  def derivative: N => N
}

/**
  * Various, common pre-build functions
  */
object Activator {

  object Sigmoid {
    def apply = new Activator[Double] {
      def apply(x: Double): Double = 1 / (1 + exp(-x))
      def derivative: Double => Double = x => exp(x) / pow(exp(x) + 1, 2)
    }
  }

  object Tanh {
    def apply = new Activator[Double] {
      def apply(x: Double): Double = (exp(x) - exp(-x))/(exp(x) + exp(-x))
      def derivative: Double => Double = x => 4 * exp(2*x) / pow(exp(2*x) + 1, 2)
    }
  }

  object Linear {
    def apply = new Activator[Double] {
      def apply(x: Double): Double = x
      def derivative: Double => Double = x => 1
    }
  }

  object Square {
    def apply = new Activator[Double] {
      def apply(x: Double): Double = x * x
      def derivative: Double => Double = x => 2 * x
    }
  }

}