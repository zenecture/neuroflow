package neuroflow.application.plugin

import java.util.concurrent.ThreadLocalRandom

import breeze.linalg.DenseVector

/**
  * @author bogdanski
  * @since 20.01.16
  */
object Notation {

  import neuroflow.core.Network._

  def ->(elems: Double*): Vector = DenseVector(elems.toArray)

  def infinity(dimension: Int): Vector = DenseVector((0 until dimension).map(_ => Double.PositiveInfinity).toArray)
  def ∞(dimension: Int): Vector = infinity(dimension)

  def zero(dimension: Int): Vector = DenseVector((0 until dimension).map(_ => 0.0).toArray)
  def ζ(dimension: Int): Vector = zero(dimension)

  def random(dimension: Int): Vector = random(dimension, 0.0, 1.0)
  def random(dimension: Int, a: Double, b: Double): Vector = DenseVector((0 until dimension).map(_ => ThreadLocalRandom.current.nextDouble(a, b)).toArray)
  def ρ(dimension: Int): Vector = random(dimension)
  def ρ(dimension: Int, a: Double, b: Double): Vector = random(dimension, a, b)

  def partition(step: Int, n: Int): Set[Int] = Range.Int.inclusive(step - 1, step * n, step).toSet
  def Π(step: Int, n: Int): Set[Int] = partition(step, n)

  object Implicits {

    implicit def toVector(seq: Seq[Double]): Vector = DenseVector(seq.toArray)

  }

}
