package neuroflow.dsl

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.storage.Zero
import neuroflow.common.CanProduce
import neuroflow.core.Tensor3D

import scala.reflect.ClassTag

/**
  * @author bogdanski
  * @since 11.03.18
  */
object Implicits {
  
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
    def apply(a: (DenseMatrix[Double], Convolution[Double])): Tensor3D[Double] = new Tensor3D[Double] {
      def mapAt(x: (Int, Int, Int))(f: Double => Double): Tensor3D[Double] = ???
      def mapAll[T: ClassTag : Zero](f: Double => T): Tensor3D[T] = ???
      val matrix: DenseMatrix[Double] = a._1
      val stride: Int = a._2.dimOut._2
    }
  }

  implicit object floatConvolution extends ((DenseMatrix[Float], Convolution[Float]) CanProduce Tensor3D[Float]) {
    def apply(a: (DenseMatrix[Float], Convolution[Float])): Tensor3D[Float] = new Tensor3D[Float] {
      def mapAt(x: (Int, Int, Int))(f: Float => Float): Tensor3D[Float] = ???
      def mapAll[T: ClassTag : Zero](f: Float => T): Tensor3D[T] = ???
      val matrix: DenseMatrix[Float] = a._1
      val stride: Int = a._2.dimOut._2
    }
  }

}

