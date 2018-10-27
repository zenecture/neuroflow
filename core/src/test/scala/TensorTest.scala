import breeze.linalg.{DenseMatrix, DenseVector}
import neuroflow.core.Tensor3D
import org.scalatest.FunSuite

/**
  * @author bogdanski
  * @since 26.03.18
  */
class TensorTest extends FunSuite {

  test("Tensor3D all zerzoes") {

    val X = 5
    val Y = 5
    val Z = 5

    val tensor = Tensor3D.zeros[Double](x = X, y = Y, z = Z)

    for {
      x <- 0 until X
      y <- 0 until Y
      z <- 0 until Z
    } {
      assert(tensor(x, y, z) == 0.0)
    }

  }

  test("Tensor3D all ones") {

    val X = 5
    val Y = 5
    val Z = 5

    val tensor = Tensor3D.ones[Double](x = X, y = Y, z = Z)

    for {
      x <- 0 until X
      y <- 0 until Y
      z <- 0 until Z
    } {
      assert(tensor(x, y, z) == 1.0)
    }

  }

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

  test("DeepCat Tensor3Ds") {

    val X = 5
    val Y = 5
    val Z = 5
    val no = 10

    val ts = (1 to no).map(_ => Tensor3D.ones[Double](x = X, y = Y, z = Z))
    val cat = Tensor3D.deepCat(ts)

    assert(cat.X == X)
    assert(cat.Y == Y)
    assert(cat.Z == Z * no)

  }

}

