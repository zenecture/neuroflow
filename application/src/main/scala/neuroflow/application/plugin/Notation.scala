package neuroflow.application.plugin

import java.util.concurrent.ThreadLocalRandom

import breeze.linalg.DenseVector
import breeze.storage.Zero
import neuroflow.core.{CanProduce, Network}

import scala.language.implicitConversions
import scala.reflect.ClassTag

/**
  * @author bogdanski
  * @since 20.01.16
  */
object Notation {

  import Network.Vector

  def ->[V: ClassTag](elems: V*): Vector[V] = DenseVector[V](elems.toArray)

  def infinity[V: ClassTag](dimension: Int)(implicit c: Double CanProduce V): Vector[V] = DenseVector((0 until dimension).map(_ => c(Double.PositiveInfinity)).toArray)
  def ∞[V: ClassTag](dimension: Int)(implicit c: Double CanProduce V): Vector[V] = infinity[V](dimension)

  def zero[V: ClassTag](dimension: Int)(implicit z: Zero[V]): Vector[V] = DenseVector((0 until dimension).map(_ => z.zero).toArray)
  def ζ[V: ClassTag](dimension: Int)(implicit z: Zero[V]): Vector[V] = zero(dimension)

  def random[V: ClassTag](dimension: Int)(implicit c: Double CanProduce V): Vector[V] = random(dimension, 0.0, 1.0)
  def random[V: ClassTag](dimension: Int, a: Double, b: Double)(implicit c: Double CanProduce V): Vector[V] =
    DenseVector((0 until dimension).map(_ => c(ThreadLocalRandom.current.nextDouble(a, b))).toArray)

  def ρ[V: ClassTag](dimension: Int)(implicit c: Double CanProduce V): Vector[V] = random(dimension)
  def ρ[V: ClassTag](dimension: Int, a: Double, b: Double)(implicit c: Double CanProduce V): Vector[V] = random(dimension, a, b)

  def partition(step: Int, n: Int): Set[Int] = Range.Int.inclusive(step - 1, step * n, step).toSet
  def Π(step: Int, n: Int): Set[Int] = partition(step, n)

  object Implicits {

    implicit def seqToVector[V: ClassTag](seq: Seq[V]): Vector[V] = DenseVector(seq.toArray)

  }

}
