package neuroflow.common

import breeze.linalg.DenseMatrix
import breeze.storage.Zero

import scala.reflect.ClassTag

/**
  * @author bogdanski
  * @since 31.01.18
  */


/**
  * A tensor exposes access to underlying `matrix` under a `projection` for coordinates `K`.
  */
trait Tensor[K, V] extends (K => V) {

  val matrix: DenseMatrix[V]

  val projection: K => (Int, Int)

  def apply(x: K): V = {
    val (row, col) = projection(x)
    matrix(row, col)
  }

  def mapAt(x: K)(f: V => V): Tensor[K, V]

  def mapAll[T: ClassTag : Zero](f: V => T): Tensor[K, T]

}

/**
  * A 3d-tensor exposes access to underlying `matrix` using (x, y, z) coordinates.
  */
trait Tensor3D[V] extends Tensor[(Int, Int, Int), V] {

  def mapAll[T: ClassTag : Zero](f: V => T): Tensor3D[T]

  def mapAt(x: (Int, Int, Int))(f: V => V): Tensor3D[V]

}

