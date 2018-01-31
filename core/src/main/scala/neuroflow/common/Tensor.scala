package neuroflow.common

import breeze.linalg.DenseMatrix
import breeze.storage.Zero

import scala.reflect.ClassTag

/**
  * @author bogdanski
  * @since 31.01.18
  */


/**
  * A tensorish exposes access to underlying
  * `matrix` under a `projection` for coordinates `K`.
  */
trait Tensorish[K, V] extends (K => V) {

  val matrix: DenseMatrix[V]

  val projection: K => (Int, Int)

  def apply(x: K): V = {
    val (row, col) = projection(x)
    matrix(row, col)
  }

  def mapAt(x: K)(f: V => V): Tensorish[K, V]

  def mapAll[T: ClassTag : Zero](f: V => T): Tensorish[K, T]

}

/**
  * A standard tensor exposes access to
  * underlying `matrix` using (x, y, z)
  * coordinates.
  */
trait Tensor[V] extends Tensorish[(Int, Int, Int), V]


