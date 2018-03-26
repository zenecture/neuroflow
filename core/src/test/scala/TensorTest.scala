import breeze.linalg.{DenseMatrix, DenseVector}
import neuroflow.core.Tensor3D
import org.scalatest.FunSuite

/**
  * @author bogdanski
  * @since 26.03.18
  */
class TensorTest extends FunSuite {

  test("Tensor3D from DenseVector") {

    val vec = DenseVector(0.1, 0.2, 0.3)
    val tensor = Tensor3D.fromVector(vec)

    assert(tensor.X == 1)
    assert(tensor.Y == vec.length)
    assert(tensor.Z == 1)

  }

  test("Tensor3D from DenseMatrix") {

    val mat = DenseMatrix.rand[Double](3, 3)
    val tensor = Tensor3D.fromMatrix(mat)

    assert(tensor.X == mat.cols)
    assert(tensor.Y == mat.rows)
    assert(tensor.Z == 1)

  }

}

