package neuroflow.core

import breeze.generic._
import breeze.linalg.max
import breeze.math.Semiring
import breeze.numerics.{exp, log, pow, tanh}


/**
  * @author bogdanski
  * @since 03.01.16
  */

/**
  * A label for a [[neuroflow.dsl.Layer]] with an `activator`
  * which gets applied on its output cells.
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
  def map[B](f: B => N, g: N => B): Activator[B] = new Activator[B] {
    val symbol: String = self.symbol
    def apply(x: B): B = g(self(f(x)))
    def derivative(x: B): B = g(self.derivative(f(x)))
  }
}

/**
  * Collection of pre-defined activators expressed as [[UFunc]].
  * The CPU implementations are found here, the GPU implicits
  * are found in the [[neuroflow.cuda.CuMatrix]] area.
  */
object Activator {

  import Ordering.Implicits._
  import Fractional.Implicits._

  sealed trait ReLU[V] extends Activator[V]
  object ReLU {
    def apply[V : Fractional](implicit _max: max.Impl2[V, V, V], ring: Semiring[V]): ReLU[V] = new ReLU[V] {
      val `0` = ring.zero
      val `1` = ring.one
      def apply(x: V): V = _max(`0`, x)
      def derivative(x: V): V = if (x > `0`) `1` else `0`
      val symbol: String = "ReLU"
    }
  }


  sealed trait LeakyReLU[V] extends Activator[V]
  object LeakyReLU {
    def apply[V : Fractional](f: V)(implicit _max: max.Impl2[V, V, V], ring: Semiring[V]): LeakyReLU[V] = new LeakyReLU[V] {
      val `0` = ring.zero
      val `1` = ring.one
      def apply(x: V): V = _max(`0`, x)
      def derivative(x: V): V = if (x > `0`) `1` else f * x
      val symbol: String = s"ReLU<$f>"
    }
  }

  sealed trait SoftPlus[V] extends Activator[V]
  object SoftPlus {
    def apply[V : Fractional](implicit _log: log.Impl[V, V], _exp: exp.Impl[V, V], ring: Semiring[V]): SoftPlus[V] = new SoftPlus[V] {
      val `1` = ring.one
      def apply(x: V): V = _log(`1` + _exp(x))
      def derivative(x: V): V = _exp(x) / (_exp(x) + `1`)
      val symbol: String = "SoftPlus"
    }
  }

  sealed trait Sigmoid[V] extends Activator[V]
  object Sigmoid {
    def apply[V : Fractional](implicit _exp: exp.Impl[V, V], ring: Semiring[V], _pow: pow.Impl2[V, V, V]): Sigmoid[V] = new Sigmoid[V] {
      val `1` = ring.one
      val `2` = ring.one + ring.one
      def apply(x: V): V = `1` / (`1` + _exp(-x))
      def derivative(x: V): V = _exp(x) / pow(_exp(x) + `1`, `2`)
      val symbol = "σ"
    }
  }

  sealed trait Tanh[V] extends Activator[V]
  object Tanh {
    def apply[V : Fractional](implicit _tanh: tanh.Impl[V, V], ring: Semiring[V], p: pow.Impl2[V, V, V]): Tanh[V] = new Tanh[V] {
      val `1` = ring.one
      val `2` = ring.one + ring.one
      def apply(x: V): V = _tanh(x)
      def derivative(x: V): V = `1` - pow(_tanh(x), `2`)
      val symbol = "φ"
    }
  }

  sealed trait Linear[V] extends Activator[V]
  object Linear {
    def apply[V : Fractional](implicit ring: Semiring[V]): Linear[V] = new Linear[V] {
      val `1` = ring.one
      def apply(x: V): V = x
      def derivative(x: V): V = `1`
      val symbol = "x"
    }
  }

  sealed trait Square[V] extends Activator[V]
  object Square {
    def apply[V : Fractional](implicit ring: Semiring[V]): Square[V] = new Square[V] {
      val `2` = ring.one + ring.one
      def apply(x: V): V = x * x
      def derivative(x: V): V = `2` * x
      val symbol = "x²"
    }
  }

}



object Activators {

  /* Short Cuts for Syntax */

  object Double {
    val ReLU = Activator.ReLU[Double]
    def LeakyReLU(f: Double) = Activator.LeakyReLU[Double](f)
    val SoftPlus = Activator.SoftPlus[Double]
    val Sigmoid = Activator.Sigmoid[Double]
    val Tanh = Activator.Tanh[Double]
    val Linear = Activator.Linear[Double]
    val Square = Activator.Square[Double]
  }

  object Float {
    val ReLU = Activator.ReLU[Float]
    def LeakyReLU(f: Float) = Activator.LeakyReLU[Float](f)
    val SoftPlus = Activator.SoftPlus[Float]
    val Sigmoid = Activator.Sigmoid[Float]
    val Tanh = Activator.Tanh[Float]
    val Linear = Activator.Linear[Float]
    val Square = Activator.Square[Float]
  }

}


