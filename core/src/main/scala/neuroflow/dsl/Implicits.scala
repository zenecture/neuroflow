package neuroflow.dsl

import breeze.linalg.{DenseMatrix, DenseVector}
import neuroflow.common.CanProduce
import neuroflow.core.{Tensor3D, Tensor3DImpl}

/**
  * @author bogdanski
  * @since 11.03.18
  */
object Implicits {

  // These CanProduces are for focusing
  
  implicit object doubleVector extends ((DenseMatrix[Double], Vector[Double]) CanProduce DenseVector[Double]) {
    def apply(a: (DenseMatrix[Double], Vector[Double])): DenseVector[Double] = a._1.toDenseVector
  }

  implicit object doubleFloat extends ((DenseMatrix[Float], Vector[Float]) CanProduce DenseVector[Float]) {
    def apply(a: (DenseMatrix[Float], Vector[Float])): DenseVector[Float] = a._1.toDenseVector
  }

  implicit object doubleDense extends ((DenseMatrix[Double], Dense[Double]) CanProduce DenseVector[Double]) {
    def apply(a: (DenseMatrix[Double], Dense[Double])): DenseVector[Double] = a._1.toDenseVector
  }

  implicit object floatDense extends ((DenseMatrix[Float], Dense[Float]) CanProduce DenseVector[Float]) {
    def apply(a: (DenseMatrix[Float], Dense[Float])): DenseVector[Float] = a._1.toDenseVector
  }
  
  implicit object doubleConvolution extends ((DenseMatrix[Double], Convolution[Double]) CanProduce Tensor3D[Double]) {
    def apply(a: (DenseMatrix[Double], Convolution[Double])): Tensor3D[Double] = new Tensor3DImpl[Double](a._1, X = a._2.dimOut._1, Y = a._2.dimOut._2, Z = a._2.dimOut._3)
  }

  implicit object floatConvolution extends ((DenseMatrix[Float], Convolution[Float]) CanProduce Tensor3D[Float]) {
    def apply(a: (DenseMatrix[Float], Convolution[Float])): Tensor3D[Float] = new Tensor3DImpl[Float](a._1, X = a._2.dimOut._1, Y = a._2.dimOut._2, Z = a._2.dimOut._3)
  }

}

