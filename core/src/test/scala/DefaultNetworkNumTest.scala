import breeze.linalg.DenseMatrix
import neuroflow.core.Activator.{Sigmoid, Linear}
import neuroflow.core._
import neuroflow.nets.DefaultNetwork
import org.specs2.Specification
import org.specs2.specification.core.SpecStructure

/**
  * @author bogdanski
  * @since 19.01.16
  */
class DefaultNetworkNumTest extends Specification {

  def is: SpecStructure =
    s2"""

    This spec will test the gradients from DefaultNetwork

    It should:
      - Check the linear gradients                                $linGrad
      - Check the linear gradients multiple output                $linGradMultiple
      - Check the nonlinear gradients                             $nonlinearGrad

  """

  val ru = scala.reflect.runtime.universe
  val m = ru.runtimeMirror(getClass.getClassLoader)

  private def toMatrix(xs: Seq[Double]) = DenseMatrix.create[Double](1, xs.size, xs.toArray)

  def linGrad = {
    import neuroflow.core.WeightProvider.randomWeights
    import neuroflow.nets.DefaultNetwork.constructor

    val fn = Linear.apply
    val sets = NetSettings(true, 0.00001, true)
    val net = Network(Input(1) :: Output(1, fn) :: Nil, sets)

    val xs = (Seq(1.0) :: Seq(2.0) :: Seq(3.0) :: Nil) map toMatrix
    val ys = (Seq(1.0) :: Seq(2.0) :: Seq(3.0) :: Nil) map toMatrix

    val layer = 0
    val weight = (0, 0)

    val instance = m.reflect(net)
    val deriveGrad = instance.reflectMethod(ru.typeOf[DefaultNetwork].decl(ru.TermName("deriveErrorFunc")).asMethod)
    val numericGrad = instance.reflectMethod(ru.typeOf[DefaultNetwork].decl(ru.TermName("deriveErrorFuncNumerically")).asMethod)

    val a = deriveGrad(xs, ys, layer, weight).asInstanceOf[DenseMatrix[Double]]
    val b = numericGrad(xs, ys, layer, weight).asInstanceOf[DenseMatrix[Double]]

    if ((a - b) forall(_.abs < 0.0001)) success else failure
  }

  def linGradMultiple = {
    import neuroflow.core.WeightProvider.randomWeights
    import neuroflow.nets.DefaultNetwork.constructor

    val fn = Linear.apply
    val sets = NetSettings(true, 0.00001, true)
    val net = Network(Input(1) :: Output(2, fn) :: Nil, sets)

    val xs = (Seq(1.0) :: Seq(2.0) :: Seq(3.0) :: Nil) map toMatrix
    val ys = (Seq(1.0, 1.0) :: Seq(2.0, 2.0) :: Seq(3.0, 3.0) :: Nil) map toMatrix

    val layers = 0 :: 0 :: Nil
    val weights = (0, 0) :: (0, 1) :: Nil

    val results = layers zip weights map { lw =>
      val (layer, weight) = lw
      val instance = m.reflect(net)
      val deriveGrad = instance.reflectMethod(ru.typeOf[DefaultNetwork].decl(ru.TermName("deriveErrorFunc")).asMethod)
      val numericGrad = instance.reflectMethod(ru.typeOf[DefaultNetwork].decl(ru.TermName("deriveErrorFuncNumerically")).asMethod)
      val a = deriveGrad(xs, ys, layer, weight).asInstanceOf[DenseMatrix[Double]]
      val b = numericGrad(xs, ys, layer, weight).asInstanceOf[DenseMatrix[Double]]
      if ((a - b).forall { (w, v) => v.abs < 0.0001 }) success else failure
    }

    if (results.forall(_.isSuccess)) success else failure
  }

  def nonlinearGrad = {
    import neuroflow.core.WeightProvider.randomWeights
    import neuroflow.nets.DefaultNetwork.constructor

    val fn = Sigmoid.apply
    val sets = NetSettings(true, 0.00001, true)
    val net = Network(Input(2) :: Hidden(3, fn) :: Output(2, fn) :: Nil, sets)

    val xs = (Seq(1.0, 2.0) :: Seq(2.0, 4.0) :: Seq(3.0, 6.0) :: Nil) map toMatrix
    val ys = (Seq(1.0, 1.0) :: Seq(2.0, 2.0) :: Seq(3.0, 3.0) :: Nil) map toMatrix

    val instance = m.reflect(net)
    val deriveGrad = instance.reflectMethod(ru.typeOf[DefaultNetwork].decl(ru.TermName("deriveErrorFunc")).asMethod)
    val numericGrad = instance.reflectMethod(ru.typeOf[DefaultNetwork].decl(ru.TermName("deriveErrorFuncNumerically")).asMethod)

    val layers = 0 :: 0 :: 0 :: 0 :: 0 :: 0 :: 1 :: 1 :: 1 :: 1 :: 1 :: 1 :: Nil
    val weights =
      (0, 0) :: (0, 1) :: (0, 2) ::
      (1, 0) :: (1, 1) :: (1, 2) ::
      (0, 0) :: (0, 1) ::
      (1, 0) :: (1, 1) ::
      (2, 0) :: (2, 1) :: Nil


    val results = layers zip weights map { lw =>
      val (layer, weight) = lw
      val instance = m.reflect(net)
      val deriveGrad = instance.reflectMethod(ru.typeOf[DefaultNetwork].decl(ru.TermName("deriveErrorFunc")).asMethod)
      val numericGrad = instance.reflectMethod(ru.typeOf[DefaultNetwork].decl(ru.TermName("deriveErrorFuncNumerically")).asMethod)
      val a = deriveGrad(xs, ys, layer, weight).asInstanceOf[DenseMatrix[Double]]
      val b = numericGrad(xs, ys, layer, weight).asInstanceOf[DenseMatrix[Double]]
      if ((a - b).forall { (w, v) => v.abs < 0.0001 }) success else failure
    }

    if (results.forall(_.isSuccess)) success else failure
  }

}
