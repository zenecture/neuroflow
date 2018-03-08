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
  * A tensor3d is a cubic volume, accessed using (x, y, z) coordinates.
  * Internally, it is linearized in `matrix` under `projection` with `stride`.
  */
trait Tensor3D[V] extends Tensor[(Int, Int, Int), V] {

  val matrix: DenseMatrix[V]

  val stride: Int

  val projection: ((Int, Int, Int)) => (Int, Int) = { case (x, y, z) => (z, x * stride + y) }

  def mapAll[T: ClassTag : Zero](f: V => T): Tensor3D[T]

  def mapAt(x: (Int, Int, Int))(f: V => V): Tensor3D[V]

}

