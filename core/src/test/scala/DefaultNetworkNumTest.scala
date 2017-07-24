package neuroflow.nets

import breeze.linalg.DenseMatrix
import neuroflow.core.Activator._
import neuroflow.core.FFN.WeightProvider._
import neuroflow.core._
import org.specs2.Specification
import org.specs2.specification.core.SpecStructure
import shapeless._

/**
  * @author bogdanski
  * @since 19.01.16
  */
class DefaultNetworkNumTest extends Specification {

  def is: SpecStructure =
    s2"""

    This spec will test the gradients from DefaultNetwork by comparison of the derived values with
    the approximated ones.

      - Check the linear gradients                                $linGrad
      - Check the linear gradients multiple output                $linGradMultiple
      - Check the nonlinear gradients                             $nonlinearGrad

  """

  val ru = scala.reflect.runtime.universe
  val m = ru.runtimeMirror(getClass.getClassLoader)

  private def toMatrix(xs: Seq[Double]) = DenseMatrix.create[Double](1, xs.size, xs.toArray)

  def linGrad = {
    import neuroflow.nets.DefaultNetwork._

    val fn = Linear
    val sets = Settings(learningRate = { case _ => 0.01 }, iterations = 1000, approximation = Some(Approximation(1E-4)))
    val net = Network(Input(1) :: Output(1, fn) :: HNil, sets)

    val xs = (Seq(1.0) :: Seq(2.0) :: Seq(3.0) :: Nil) map toMatrix
    val ys = (Seq(1.0) :: Seq(2.0) :: Seq(3.0) :: Nil) map toMatrix

    val layer = 0
    val weight = (0, 0)

    val instance = m.reflect(net)
    val deriveGrad = instance.reflectMethod(ru.typeOf[DefaultNetwork].decl(ru.TermName("errorFuncDerivative")).asMethod)
    val numericGrad = instance.reflectMethod(ru.typeOf[DefaultNetwork].decl(ru.TermName("approximateErrorFuncDerivative")).asMethod)

    val a = deriveGrad(xs, ys, layer, weight).asInstanceOf[DenseMatrix[Double]]
    val b = numericGrad(xs, ys, layer, weight).asInstanceOf[DenseMatrix[Double]]

    if ((a - b) forall(_.abs < 0.0001)) success else failure
  }

  def linGradMultiple = {
    import neuroflow.nets.DefaultNetwork._

    val fn = Linear
    val sets = Settings(learningRate = { case _ => 0.01 }, iterations = 1000, approximation = Some(Approximation(1E-4)))
    val net = Network(Input(1) :: Output(2, fn) :: HNil, sets)

    val xs = (Seq(1.0) :: Seq(2.0) :: Seq(3.0) :: Nil) map toMatrix
    val ys = (Seq(1.0, 1.0) :: Seq(2.0, 2.0) :: Seq(3.0, 3.0) :: Nil) map toMatrix

    val layers = 0 :: 0 :: Nil
    val weights = (0, 0) :: (0, 1) :: Nil

    val results = layers zip weights map { lw =>
      val (layer, weight) = lw
      val instance = m.reflect(net)
      val deriveGrad = instance.reflectMethod(ru.typeOf[DefaultNetwork].decl(ru.TermName("errorFuncDerivative")).asMethod)
      val numericGrad = instance.reflectMethod(ru.typeOf[DefaultNetwork].decl(ru.TermName("approximateErrorFuncDerivative")).asMethod)
      val a = deriveGrad(xs, ys, layer, weight).asInstanceOf[DenseMatrix[Double]]
      val b = numericGrad(xs, ys, layer, weight).asInstanceOf[DenseMatrix[Double]]
      if ((a - b).forall { (w, v) => v.abs < 0.0001 }) success else failure
    }

    if (results.forall(_.isSuccess)) success else failure
  }

  def nonlinearGrad = {
    import neuroflow.nets.DefaultNetwork._

    val fn = Sigmoid
    val gn = Tanh
    val sets = Settings(learningRate = { case _ => 0.01 }, iterations = 1000, approximation = Some(Approximation(1E-4)))
    val net = Network(Input(2) :: Hidden(30, fn) :: Hidden(10, gn) :: Output(2, fn) :: HNil, sets)

    val xs = (Seq(1.0, 2.0) :: Seq(2.0, 4.0) :: Seq(3.0, 6.0) :: Nil) map toMatrix
    val ys = (Seq(1.0, 1.0) :: Seq(2.0, 2.0) :: Seq(3.0, 3.0) :: Nil) map toMatrix

    val weights = net.layers.indices.dropRight(1) flatMap { i =>
      net.weights(i).mapPairs((p, v) => (i, p)).activeValuesIterator.toList
    }

    val results = weights map { lw =>
      val (layer, weight) = lw
      val instance = m.reflect(net)
      val deriveGrad = instance.reflectMethod(ru.typeOf[DefaultNetwork].decl(ru.TermName("errorFuncDerivative")).asMethod)
      val numericGrad = instance.reflectMethod(ru.typeOf[DefaultNetwork].decl(ru.TermName("approximateErrorFuncDerivative")).asMethod)
      val a = deriveGrad(xs, ys, layer, weight).asInstanceOf[DenseMatrix[Double]]
      val b = numericGrad(xs, ys, layer, weight).asInstanceOf[DenseMatrix[Double]]
      if ((a - b).forall { (w, v) => v.abs < 0.0001 }) success else failure
    }

    if (results.forall(_.isSuccess)) success else failure
  }

}
