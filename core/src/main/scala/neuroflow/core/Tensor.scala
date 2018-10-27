package neuroflow.core

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.math.Semiring
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

  /** Immutable **/
  def mapAll[T: ClassTag : Zero](f: V => T): Tensor[K, T]

  /** Immutable **/
  def mapAt(x: K)(f: V => V): Tensor[K, V]

  /** Mutable */
  def updateAt(x: K)(v: V): Unit

}


/**
  * A tensor3d is a cubic volume, accessed using (x, y, z) coordinates.
  * Internally, it is stored as linearized `matrix`, accessed by `projection` with stride `Y`.
  */
trait Tensor3D[V] extends Tensor[(Int, Int, Int), V] {

  val X, Y, Z: Int

  val matrix: DenseMatrix[V]

  val projection: ((Int, Int, Int)) => (Int, Int) = { case (x, y, z) => (z, x * Y + y) }

  def mapAll[T: ClassTag : Zero](f: V => T): Tensor3D[T]

  def mapAt(x: (Int, Int, Int))(f: V => V): Tensor3D[V]

  def updateAt(x: (Int, Int, Int))(v: V): Unit

}


object Tensor3D {

  /**
    * Creates all zero [[Tensor3D]] of dim (X, Y, Z) = (`x`, `y`, `z`).
    */
  def zeros[V: ClassTag : Zero](x: Int, y: Int, z: Int): Tensor3D[V] = {
    new Tensor3DImpl[V](DenseMatrix.zeros[V](rows = z, cols = x * y), X = x, Y = y, Z = z)
  }

  /**
    * Creates all one [[Tensor3D]] of dim (X, Y, Z) = (`x`, `y`, `z`).
    */
  def ones[V: ClassTag : Zero : Semiring](x: Int, y: Int, z: Int): Tensor3D[V] = {
    val zs = zeros(x, y, z)
    zs.mapAll(_ => implicitly[Semiring[V]].one)
  }

  /**
    * Creates vertically shaped [[Tensor3D]] from vector `v` with dim (X, Y, Z) = (1, |Vector|, 1).
    */
  def fromVector[V: ClassTag : Zero](v: DenseVector[V]): Tensor3D[V] = {
    new Tensor3DImpl[V](DenseMatrix.create[V](1, v.length, v.data), X = 1, Y = v.length, Z = 1)
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
    * Concatenates tensors `t` by z-dimension (depth).
    */
  def deepCat[V: ClassTag : Zero](ts: Seq[Tensor3D[V]]): Tensor3D[V] = {
    val x = ts.head.X
    val y = ts.head.Y
    val z = ts.map(_.Z).sum
    require(ts.forall(t => t.X == x && t.Y == y), "All tensors must share same dimension X, Y!")
    val mergedMat = ts.map(_.matrix).reduce((a, b) => DenseMatrix.vertcat(a, b))
    new Tensor3DImpl[V](mergedMat, X = x, Y = y, Z = z)
  }

}


class Tensor3DImpl[V](val matrix: DenseMatrix[V], val X: Int, val Y: Int, val Z: Int) extends Tensor3D[V] {

  def mapAll[T: ClassTag : Zero](f: V => T): Tensor3D[T] = {
    new Tensor3DImpl(matrix.map(f), X, Y, Z)
  }

  def mapAt(x: (Int, Int, Int))(f: V => V): Tensor3D[V] = {
    val newMat = matrix.copy
    val (row, col) = projection(x._1, x._2, x._3)
    newMat.update(row, col, f(apply(x)))
    new Tensor3DImpl(newMat, X, Y, Z)
  }

  def updateAt(x: (Int, Int, Int))(v: V): Unit = {
    val (row, col) = projection(x._1, x._2, x._3)
    matrix.update(row, col, v)
  }

}

