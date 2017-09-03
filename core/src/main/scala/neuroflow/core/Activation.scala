package neuroflow.core

import java.lang.Math.{exp, log, pow, tanh}

import breeze.generic._
import breeze.linalg.max


/**
  * @author bogdanski
  * @since 03.01.16
  */

/**
  * Label for neurons in the network performing a function on their synapses.
  */
trait HasActivator[N] {
  val activator: Activator[N]
}

/**
  * The activator function with its derivative.
  */
trait Activator[N] extends (N => N) with UFunc with MappingUFunc with Serializable { self =>
  implicit object impl extends Impl[N, N] { def apply(v: N): N = self.apply(x = v) }
  val symbol: String
  def apply(x: N): N
  def derivative(x: N): N
}

/**
  * Activator functions.
  */
object Activator {

  object ReLU extends Activator[Double] {
    val symbol = "R"
    def apply(x: Double): Double = max(0.0, x)
    def derivative(x: Double): Double = if (x > 0.0) 1.0 else 0.0
  }

  object LeakyReLU {
    def apply(f: Double) = new Activator[Double] {
      val symbol = s"R<$f>"
      def apply(x: Double): Double = max(0.0, x)
      def derivative(x: Double): Double = if (x > 0.0) 1.0 else f * x
    }
  }

  object SoftPlus extends Activator[Double] {
    val symbol = "Σ"
    def apply(x: Double): Double = log(1 + exp(x))
    def derivative(x: Double): Double = exp(x) / (exp(x) + 1)
  }

  object Sigmoid extends Activator[Double] {
    val symbol = "σ"
    def apply(x: Double): Double = 1 / (1 + exp(-x))
    def derivative(x: Double): Double = exp(x) / pow(exp(x) + 1, 2)
  }

  object CustomSigmoid {
    def apply(f: Double, g: Double, b: Double) = new Activator[Double] {
      val symbol = s"σ<$f, $g, $b>"
      def apply(x: Double): Double = (f / (1 + exp(-x * g))) - b
      def derivative(x: Double): Double = f * exp(x) / pow(exp(x) + 1, 2)
    }
  }

  object Tanh extends Activator[Double] {
    val symbol = "φ"
    def apply(x: Double): Double = tanh(x)
    def derivative(x: Double): Double = 1 - pow(tanh(x), 2)
  }

  object Linear extends Activator[Double] {
    val symbol = "x"
    def apply(x: Double): Double = x
    def derivative(x: Double): Double = 1
  }

  object Square extends Activator[Double] {
    val symbol = "x²"
    def apply(x: Double): Double = x * x
    def derivative(x: Double): Double = 2 * x
  }

}
