package neuroflow.common

import breeze.linalg.DenseVector

/**
  * @author bogdanski
  * @since 02.05.17
  */
object VectorTranslation {

  implicit class AsDenseVector(v: Vector[Double]) {
    def dv: DenseVector[Double] = DenseVector(v.toArray)
  }

  implicit class AsVector(w: DenseVector[Double]) {
    def vv: Vector[Double] = w.toArray.toVector
  }

}
