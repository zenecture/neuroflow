package neuroflow.core

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.storage.Zero

import scala.reflect.ClassTag

/**
  * @author bogdanski
  * @since 31.01.18
  */


/**
  * A tensor exposes access to underlying `matrix` under
  * a row-col `projection` for coordinates `K`.
  */
trait Tensor[K, V] extends (K => V) {

  val matrix: DenseMatrix[V]

  val projection: K => (Int, Int)

  def apply(x: K): V = {
    val (row, col) = projection(x)
    matrix(row, col)
  }

  def mapAll[T: ClassTag : Zero](f: V => T): Tensor[K, T]

  def mapAt(x: K)(f: V => V): Tensor[K, V]

  def updateAt(x: K)(v: V): Unit

}


/**
  * A tensor3d is a cubic volume, accessed using (x, y, z) coordinates.
  * Internally, it is stored as linearized `matrix`, accessed by `projection`
  * function with column stride `Y`.
  */
trait Tensor3D[V] extends Tensor[(Int, Int, Int), V] {

  val X, Y, Z: Int

  val matrix: DenseMatrix[V]

  val projection: ((Int, Int, Int)) => (Int, Int) = { case (x, y, z) => (z, x * Y + y) }

  def mapAll[T: ClassTag : Zero](f: V => T): Tensor3D[T]

  def updateAt(x: (Int, Int, Int))(v: V): Unit

  def mapAt(x: (Int, Int, Int))(f: V => V): Tensor3D[V]

}


object Tensor3D {

  /**
    * Creates vertically shaped [[Tensor3D]] from vector `v` with dim (X, Y, Z) = (1, VectorLength, 1).
    */
  def fromVector[V: ClassTag : Zero](v: DenseVector[V]): Tensor3D[V] = {
    val t = new Tensor3DImpl[V](DenseMatrix.create[V](1, v.length, v.data), X = 1, Y = v.length, Z = 1)
    t
  }

  /**
    * Creates [[Tensor3D]] from matrix `m` with dim (X, Y, Z) = (Cols, Rows, 1).
    */
  def fromMatrix[V: ClassTag : Zero](m: DenseMatrix[V]): Tensor3D[V] = {
    val t = new Tensor3DImpl[V](DenseMatrix.zeros[V](1, m.rows * m.cols), X = m.cols, Y = m.rows, Z = 1)
    (0 until m.rows).foreach { row =>
      (0 until m.cols).foreach { col =>
        t.updateAt(col, row, 0)(m(row, col))
      }
    }
    t
  }

  /**
    * Merges tensors `t` by z-dimension (depth).
    */
  def zCat[V](t: Seq[Tensor3D[V]]): Tensor3D[V] = {
    // TODO
    ???
  }

}


class Tensor3DImpl[V](val matrix: DenseMatrix[V], val X: Int, val Y: Int, val Z: Int) extends Tensor3D[V] {

  def mapAt(x: (Int, Int, Int))(f: V => V): Tensor3D[V] = {
    val newMat = matrix.copy
    val (row, col) = projection(x._1, x._2, x._3)
    newMat.update(row, col, f(apply(x)))
    new Tensor3DImpl(newMat, X, Y, Z)
  }

  def mapAll[T: ClassTag : Zero](f: V => T): Tensor3D[T] = {
    new Tensor3DImpl(matrix.map(f), X, Y, Z)
  }

  def updateAt(x: (Int, Int, Int))(v: V): Unit = {
    val (row, col) = projection(x._1, x._2, x._3)
    matrix.update(row, col, v)
  }

}

