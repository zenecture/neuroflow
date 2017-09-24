package neuroflow.application.plugin

import java.util.concurrent.ThreadLocalRandom

import breeze.linalg.DenseVector
import breeze.storage.Zero
import neuroflow.application.plugin.Notation.Implicits.CanProduceDouble
import neuroflow.core.Network

import scala.reflect.ClassTag

/**
  * @author bogdanski
  * @since 20.01.16
  */
object Notation {

  import Network.Vector

  def ->[V: ClassTag](elems: V*): Vector[V] = DenseVector[V](elems.toArray)

  def infinity[V: ClassTag](dimension: Int)(implicit c: CanProduceDouble[V]): Vector[V] = DenseVector((0 until dimension).map(_ => c(Double.PositiveInfinity)).toArray)
  def ∞[V: ClassTag](dimension: Int)(implicit c: CanProduceDouble[V]): Vector[V] = infinity[V](dimension)

  def zero[V: ClassTag](dimension: Int)(implicit z: Zero[V]): Vector[V] = DenseVector((0 until dimension).map(_ => z.zero).toArray)
  def ζ[V: ClassTag](dimension: Int)(implicit z: Zero[V]): Vector[V] = zero(dimension)

  def random[V: ClassTag](dimension: Int)(implicit c: CanProduceDouble[V]): Vector[V] = random(dimension, 0.0, 1.0)
  def random[V: ClassTag](dimension: Int, a: Double, b: Double)(implicit c: CanProduceDouble[V]): Vector[V] =
    DenseVector((0 until dimension).map(_ => c(ThreadLocalRandom.current.nextDouble(a, b))).toArray)

  def ρ[V: ClassTag](dimension: Int)(implicit c: CanProduceDouble[V]): Vector[V] = random(dimension)
  def ρ[V: ClassTag](dimension: Int, a: Double, b: Double)(implicit c: CanProduceDouble[V]): Vector[V] = random(dimension, a, b)

  def partition(step: Int, n: Int): Set[Int] = Range.Int.inclusive(step - 1, step * n, step).toSet
  def Π(step: Int, n: Int): Set[Int] = partition(step, n)

  object Implicits {

    implicit def toVector[V: ClassTag](seq: Seq[V]): Vector[V] = DenseVector(seq.toArray)

    trait CanProduceDouble[V] {
      def apply(d: Double): V
    }

    implicit object CanProduceDoubleFromFloat extends CanProduceDouble[Float] {
      def apply(d: Double): Float = d.toFloat
    }

    implicit object CanProduceDoubleFromDouble extends CanProduceDouble[Double] {
      def apply(d: Double): Double = d
    }

  }

}
