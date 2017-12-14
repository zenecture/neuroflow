package neuroflow.common

import breeze.linalg.DenseVector

import scala.reflect.ClassTag

/**
  * @author bogdanski
  * @since 02.05.17
  */
object VectorTranslation {

  implicit class AsDenseVector[V: ClassTag](v: scala.Vector[V]) {
    def dv: DenseVector[V] = DenseVector(v.toArray)
  }

  implicit class AsVector[V: ClassTag](w: DenseVector[V]) {
    def vv: scala.Vector[V] = w.toArray.toVector
  }

}
