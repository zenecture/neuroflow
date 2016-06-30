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
trait Activator[N] extends (N => N) with Serializable {
  def apply(x: N): N
  def derivative(x: N): N
}

/**
  * Common pre-build activator/squashing functions.
  */
object Activator {

  object Sigmoid extends Activator[Double] {
    def apply(x: Double): Double = 1 / (1 + exp(-x))
    def derivative(x: Double): Double = exp(x) / pow(exp(x) + 1, 2)
  }

  object CenteredSigmoid {
    def apply(f: Int) = new Activator[Double] {
      def apply(x: Double): Double = (f / (1 + exp(-x))) - 2
      def derivative(x: Double): Double = f * exp(x) / pow(exp(x) + 1, 2)
    }
  }

  object Tanh extends Activator[Double] {
    def apply(x: Double): Double = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    def derivative(x: Double): Double = 4 * exp(2 * x) / pow(exp(2 * x) + 1, 2)
  }

  object Linear extends Activator[Double] {
    def apply(x: Double): Double = x
    def derivative(x: Double): Double = 1
  }

  object Square extends Activator[Double] {
    def apply(x: Double): Double = x * x
    def derivative(x: Double): Double = 2 * x
  }

}
