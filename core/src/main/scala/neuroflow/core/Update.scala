package neuroflow.core

import breeze.linalg.DenseMatrix
import jcuda.jcublas._
import neuroflow.nets.gpu.cuda.CuMatrix

/**
  * @author bogdanski
  * @since 09.09.17
  */


/**
  * Updates weights `ws` using derivatives `dws` and `learningRate` for layer `position`.
  */
trait Update {
  def apply(ws: DenseMatrix[Double], dws: DenseMatrix[Double], learningRate: Double, position: Int): Unit
  def apply(ws: DenseMatrix[Float], dws: DenseMatrix[Float], learningRate: Float, position: Int): Unit
  def apply(ws: CuMatrix[Double], dws: CuMatrix[Double], learningRate: Double, position: Int)(implicit handle: cublasHandle): Unit
  def apply(ws: CuMatrix[Float], dws: CuMatrix[Float], learningRate: Float, position: Int)(implicit handle: cublasHandle): Unit
}



case object Vanilla extends Update {
  def apply(ws: DenseMatrix[Double], dws: DenseMatrix[Double], learningRate: Double, position: Int): Unit = ws -= (dws *= learningRate)
  def apply(ws: DenseMatrix[Float], dws: DenseMatrix[Float], learningRate: Float, position: Int): Unit = ws -= (dws *= learningRate)
  def apply(ws: CuMatrix[Double], dws: CuMatrix[Double], learningRate: Double, position: Int)(implicit handle: cublasHandle): Unit = ws -= (dws *= learningRate)
  def apply(ws: CuMatrix[Float], dws: CuMatrix[Float], learningRate: Float, position: Int)(implicit handle: cublasHandle): Unit = ws -= (dws *= learningRate)
}



case class Momentum(μ: Double = 0.9) extends Update {

  val μf = μ.toFloat

  private val vs1 = collection.mutable.HashMap.empty[Int, DenseMatrix[Double]]
  private val vs2 = collection.mutable.HashMap.empty[Int, DenseMatrix[Float]]
  private val vs3 = collection.mutable.HashMap.empty[Int, CuMatrix[Double]]
  private val vs4 = collection.mutable.HashMap.empty[Int, CuMatrix[Float]]

  def apply(ws: DenseMatrix[Double], dws: DenseMatrix[Double], learningRate: Double, position: Int): Unit = {
    if (vs1.isDefinedAt(position)) vs1(position) := (μ * vs1(position)) - (dws *= learningRate)
    else vs1 += position -> -(dws * learningRate)
    ws += vs1(position)
  }

  def apply(ws: DenseMatrix[Float], dws: DenseMatrix[Float], learningRate: Float, position: Int): Unit = {
    if (vs2.isDefinedAt(position)) vs2(position) := (μf * vs2(position)) - (dws *= learningRate)
    else vs2 += position -> -(dws * learningRate)
    ws += vs2(position)
  }

  def apply(ws: CuMatrix[Double], dws: CuMatrix[Double], learningRate: Double, position: Int)(implicit handle: cublasHandle): Unit = {
    if (vs3.isDefinedAt(position)) {
      val r1 = μ * vs3(position)
      val r2 = r1 - (dws *= learningRate)
      vs3(position) := r2
      r1.release()
      r2.release()
    } else vs3 += position -> -(dws * learningRate)
    ws += vs3(position)
  }

  def apply(ws: CuMatrix[Float], dws: CuMatrix[Float], learningRate: Float, position: Int)(implicit handle: cublasHandle): Unit = {
    if (vs4.isDefinedAt(position)) {
      val r1 = μf * vs4(position)
      val r2 = r1 - (dws *= learningRate)
      vs4(position) := r2
      r1.release()
      r2.release()
    } else vs4 += position -> -(dws * learningRate)
    ws += vs4(position)
  }

  def release(): Unit = {
    vs3.values.foreach(_.release())
    vs4.values.foreach(_.release())
  }

}



case class Debuggable() extends Update {

  var lastGradients = collection.mutable.Map.empty[Int, DenseMatrix[Double]]

  def apply(ws: DenseMatrix[Double], dws: DenseMatrix[Double], learningRate: Double, position: Int): Unit = {
    lastGradients += position -> dws
    ws -= (dws *= learningRate)
  }

  def apply(ws: DenseMatrix[Float], dws: DenseMatrix[Float], learningRate: Float, position: Int): Unit = ???
  def apply(ws: CuMatrix[Double], dws: CuMatrix[Double], learningRate: Double, position: Int)(implicit handle: cublasHandle): Unit = ???
  def apply(ws: CuMatrix[Float], dws: CuMatrix[Float], learningRate: Float, position: Int)(implicit handle: cublasHandle): Unit = ???

}
