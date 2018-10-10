import org.scalatest.FunSuite
import breeze.linalg.DenseVector
import neuroflow.core.Activators.Double._
import neuroflow.core._
import neuroflow.dsl._

/**
  * @author bogdanski
  * @since 10.10.18
  */
class BatchModeTest extends FunSuite {

  test("Batch Mode for Dense Net CPU") {

    import neuroflow.nets.cpu.DenseNetwork._
    implicit val weights = WeightBreeder[Double].random(-1, 1)
    val f = Sigmoid
    val net = Network(layout = Vector(2) :: Dense(3, f) :: Dense(10, f) :: SquaredError())
    val batch = (1 to 100).map { _ => DenseVector.rand[Double](size = 2) }
    val res = net.batchApply(batch)

    assert(res.size == batch.size)

  }

  test("Batch Mode for Conv Net CPU") {

    import neuroflow.nets.cpu.ConvNetwork._
    implicit val weights = WeightBreeder[Double].random(-1, 1)
    val f = Sigmoid
    val net = Network(layout =
      Convolution((1, 2, 1), (0, 0), (1, 2), (1, 1), 3, f) :: Dense(10, f) :: SquaredError()
    )
    val batch = (1 to 100).map { _ => Tensor3D.fromVector(DenseVector.rand[Double](size = 2)) }
    val res = net.batchApply(batch)

    assert(res.size == batch.size)

  }

  test("Batch Mode for Dense Net GPU") {

    import neuroflow.nets.gpu.DenseNetwork._
    implicit val weights = WeightBreeder[Double].random(-1, 1)
    val f = Sigmoid
    val net = Network(layout = Vector(2) :: Dense(3, f) :: Dense(10, f) :: SquaredError())
    val batch = (1 to 100).map { _ => DenseVector.rand[Double](size = 2) }
    val res = net.batchApply(batch)

    assert(res.size == batch.size)

  }

  test("Batch Mode for Conv Net GPU") {

    import neuroflow.nets.gpu.ConvNetwork._
    implicit val weights = WeightBreeder[Double].random(-1, 1)
    val f = Sigmoid
    val net = Network(layout =
      Convolution((1, 2, 1), (0, 0), (1, 2), (1, 1), 3, f) :: Dense(10, f) :: SquaredError()
    )
    val batch = (1 to 100).map { _ => Tensor3D.fromVector(DenseVector.rand[Double](size = 2)) }
    val res = net.batchApply(batch)

    assert(res.size == batch.size)

  }

}
