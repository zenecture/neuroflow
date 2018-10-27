package neuroflow.core

import breeze.linalg.{sum, _}
import breeze.linalg.operators._
import breeze.math.{Field, Semiring}
import breeze.{linalg, numerics}
import breeze.numerics._
import breeze.storage.Zero
import jcuda.jcublas.cublasHandle
import neuroflow.cuda._
import neuroflow.dsl.Layout

import scala.reflect.ClassTag

/**
  * @author bogdanski
  * @since 02.10.17
  */


/**
  * A loss function gets target `y`, prediction `x`, returns loss and gradient,
  * which will be backpropped into the raw output layer of a net.
  */
trait LossFunction[V] extends Layout {

  /** CPU */
  def apply(y: DenseMatrix[V], x: DenseMatrix[V])
           (implicit
            field: Field[V], classTag: ClassTag[V],
            _srm: subRowMax.Impl[DenseMatrix[V], DenseMatrix[V]],
            _exp: exp.Impl[DenseMatrix[V], DenseMatrix[V]],
            _sum: sum.Impl[DenseMatrix[V], V],
            _log: log.Impl[DenseMatrix[V], DenseMatrix[V]],
            _abs: abs.Impl[DenseMatrix[V], DenseMatrix[V]],
            _mat: OpMulMatrix.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            _mat2: OpMulMatrix.Impl2[V, DenseMatrix[V], DenseMatrix[V]],
            _mat3: OpMulMatrix.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            _add: OpAdd.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            _sub: OpSub.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            _neg: OpNeg.Impl[DenseMatrix[V], DenseMatrix[V]],
            _pow: OpPow.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            _mulInPl: OpMulScalar.InPlaceImpl2[DenseMatrix[V], V],
            _setInPl: OpSet.InPlaceImpl2[DenseMatrix[V], V],
            _powInPl: OpPow.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            _subInPl: OpSub.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            _addInPl: OpAdd.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]]): (DenseMatrix[V], DenseMatrix[V])

  /** GPU */
  def apply(y: CuMatrix[V], x: CuMatrix[V])
           (implicit
            handle: cublasHandle, field: Field[V], classTag: ClassTag[V],
            _srm: subRowMax.Impl[CuMatrix[V], CuMatrix[V]],
            _exp: exp.Impl[CuMatrix[V], CuMatrix[V]],
            _sum: sum.Impl[CuMatrix[V], V],
            _log: log.Impl[CuMatrix[V], CuMatrix[V]],
            _abs: abs.Impl[CuMatrix[V], CuMatrix[V]],
            _mat: OpMulMatrix.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _mat2: OpMulMatrix.Impl2[V, CuMatrix[V], CuMatrix[V]],
            _mat3: OpMulMatrix.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _add: OpAdd.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _sub: OpSub.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _sub2: OpSub.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _mul1: OpMulScalar.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _mul2: OpMulScalar.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _neg: OpNeg.Impl[CuMatrix[V], CuMatrix[V]],
            _pow: OpPow.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _div: OpDiv.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _mulInPl: OpMulScalar.InPlaceImpl2[CuMatrix[V], V],
            _setInPl: OpSet.InPlaceImpl2[CuMatrix[V], V],
            _powInPl: OpPow.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            _subInPl: OpSub.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            _addInPl: OpAdd.InPlaceImpl2[CuMatrix[V], CuMatrix[V]]): (CuMatrix[V], CuMatrix[V])

  /**
    * Optional post processing of last layer when evaluating
    * the net with this loss function attached. This can be seen
    * as an implicit sink layer.
    *  */
  def sink(x: DenseMatrix[V])
           (implicit
            field: Field[V], classTag: ClassTag[V],
            _srm: subRowMax.Impl[DenseMatrix[V], DenseMatrix[V]],
            _exp: exp.Impl[DenseMatrix[V], DenseMatrix[V]],
            _sum: sum.Impl[DenseMatrix[V], V],
            _log: log.Impl[DenseMatrix[V], DenseMatrix[V]],
            _abs: abs.Impl[DenseMatrix[V], DenseMatrix[V]],
            _mat: OpMulMatrix.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            _mat2: OpMulMatrix.Impl2[V, DenseMatrix[V], DenseMatrix[V]],
            _mat3: OpMulMatrix.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            _add: OpAdd.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            _sub: OpSub.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            _neg: OpNeg.Impl[DenseMatrix[V], DenseMatrix[V]],
            _pow: OpPow.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            _mulInPl: OpMulScalar.InPlaceImpl2[DenseMatrix[V], V],
            _setInPl: OpSet.InPlaceImpl2[DenseMatrix[V], V],
            _powInPl: OpPow.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            _subInPl: OpSub.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            _addInPl: OpAdd.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]]): DenseMatrix[V] = x

}


/**
  *
  *   L = Σ|1/3(y - x)³|
  *
  * Where `y` is the target and `x` the prediction.
  * The sum Σ is taken over the full batch and
  * the absolute cubic gives a convex functional form.
  *
  */
case class AbsCubicError[V]() extends LossFunction[V] {

  def apply(y: DenseMatrix[V], x: DenseMatrix[V])
           (implicit
            field: Field[V], classTag: ClassTag[V],
            _srm: subRowMax.Impl[DenseMatrix[V], DenseMatrix[V]],
            _exp: exp.Impl[DenseMatrix[V], DenseMatrix[V]],
            _sum: sum.Impl[DenseMatrix[V], V],
            _log: log.Impl[DenseMatrix[V], DenseMatrix[V]],
            _abs: abs.Impl[DenseMatrix[V], DenseMatrix[V]],
            _mat: OpMulMatrix.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            _mat2: OpMulMatrix.Impl2[V, DenseMatrix[V], DenseMatrix[V]],
            _mat3: OpMulMatrix.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            _add: OpAdd.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            _sub: OpSub.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            _neg: OpNeg.Impl[DenseMatrix[V], DenseMatrix[V]],
            _pow: OpPow.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            _mulInPl: OpMulScalar.InPlaceImpl2[DenseMatrix[V], V],
            _setInPl: OpSet.InPlaceImpl2[DenseMatrix[V], V],
            _powInPl: OpPow.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            _subInPl: OpSub.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            _addInPl: OpAdd.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]]): (DenseMatrix[V], DenseMatrix[V]) = {


    val `1` = field.one
    val `2` = field + (`1`, `1`)
    val `3` = field + (`2`, `1`)
    val threes = DenseMatrix.zeros[V](y.rows, y.cols)
    threes := `3`

    val cubic = (y - x) ^:^ threes
    val err = _abs(cubic / threes)
    val grad = (x - y) *:* _abs(x - y)

    (err, grad)

  }

  def apply(y: CuMatrix[V], x: CuMatrix[V])
           (implicit
            handle: cublasHandle, field: Field[V], classTag: ClassTag[V],
            _srm: subRowMax.Impl[CuMatrix[V], CuMatrix[V]],
            _exp: exp.Impl[CuMatrix[V], CuMatrix[V]],
            _sum: sum.Impl[CuMatrix[V], V],
            _log: log.Impl[CuMatrix[V], CuMatrix[V]],
            _abs: abs.Impl[CuMatrix[V], CuMatrix[V]],
            _mat: OpMulMatrix.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _mat2: OpMulMatrix.Impl2[V, CuMatrix[V], CuMatrix[V]],
            _mat3: OpMulMatrix.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _add: OpAdd.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _sub: OpSub.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _sub2: OpSub.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _mul1: OpMulScalar.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _mul2: OpMulScalar.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _neg: OpNeg.Impl[CuMatrix[V], CuMatrix[V]],
            _pow: OpPow.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _div: OpDiv.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _mulInPl: OpMulScalar.InPlaceImpl2[CuMatrix[V], V],
            _setInPl: OpSet.InPlaceImpl2[CuMatrix[V], V],
            _powInPl: OpPow.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            _subInPl: OpSub.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            _addInPl: OpAdd.InPlaceImpl2[CuMatrix[V], CuMatrix[V]]): (CuMatrix[V], CuMatrix[V]) = {

    val `1` = field.one
    val `2` = field + (`1`, `1`)
    val `3` = field + (`2`, `1`)
    val threes = CuMatrix.zeros[V](y.rows, y.cols)
    threes := `3`

    val cubic = (y - x) ^:^ threes
    val err = _abs(cubic / threes)
    val grad = (x - y) *:* _abs(x - y)

    (err, grad)

  }

}



/**
  *
  *   L = Σ1/2(y - x)²
  *
  * Where `y` is the target and `x` the prediction.
  * The sum Σ is taken over the full batch and
  * the square ² gives a convex functional form.
  *
  */
case class SquaredError[V]() extends LossFunction[V] {

  def apply(y: DenseMatrix[V], x: DenseMatrix[V])
           (implicit
            field: Field[V], classTag: ClassTag[V],
            _srm: subRowMax.Impl[DenseMatrix[V], DenseMatrix[V]],
            _exp: exp.Impl[DenseMatrix[V], DenseMatrix[V]],
            _sum: sum.Impl[DenseMatrix[V], V],
            _log: log.Impl[DenseMatrix[V], DenseMatrix[V]],
            _abs: abs.Impl[DenseMatrix[V], DenseMatrix[V]],
            _mat: OpMulMatrix.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            _mat2: OpMulMatrix.Impl2[V, DenseMatrix[V], DenseMatrix[V]],
            _mat3: OpMulMatrix.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            _add: OpAdd.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            _sub: OpSub.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            _neg: OpNeg.Impl[DenseMatrix[V], DenseMatrix[V]],
            _pow: OpPow.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            _mulInPl: OpMulScalar.InPlaceImpl2[DenseMatrix[V], V],
            _setInPl: OpSet.InPlaceImpl2[DenseMatrix[V], V],
            _powInPl: OpPow.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            _subInPl: OpSub.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            _addInPl: OpAdd.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]]): (DenseMatrix[V], DenseMatrix[V]) = {

    val `2` = field + (field.one, field.one)
    val `0.5` = field / (field.one, `2`)
    val pow = DenseMatrix.zeros[V](y.rows, y.cols)
    pow := `2`
    val grad = y - x
    val err = grad ^:^ pow
    err *= `0.5`

    (err, -grad)

  }

  def apply(y: CuMatrix[V], x: CuMatrix[V])
           (implicit
            handle: cublasHandle, field: Field[V], classTag: ClassTag[V],
            _srm: subRowMax.Impl[CuMatrix[V], CuMatrix[V]],
            _exp: exp.Impl[CuMatrix[V], CuMatrix[V]],
            _sum: sum.Impl[CuMatrix[V], V],
            _log: log.Impl[CuMatrix[V], CuMatrix[V]],
            _abs: abs.Impl[CuMatrix[V], CuMatrix[V]],
            _mat: OpMulMatrix.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _mat2: OpMulMatrix.Impl2[V, CuMatrix[V], CuMatrix[V]],
            _mat3: OpMulMatrix.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _add: OpAdd.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _sub: OpSub.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _sub2: OpSub.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _mul1: OpMulScalar.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _mul2: OpMulScalar.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _neg: OpNeg.Impl[CuMatrix[V], CuMatrix[V]],
            _pow: OpPow.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _div: OpDiv.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _mulInPl: OpMulScalar.InPlaceImpl2[CuMatrix[V], V],
            _setInPl: OpSet.InPlaceImpl2[CuMatrix[V], V],
            _powInPl: OpPow.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            _subInPl: OpSub.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            _addInPl: OpAdd.InPlaceImpl2[CuMatrix[V], CuMatrix[V]]): (CuMatrix[V], CuMatrix[V]) = {

    val `2` = field + (field.one, field.one)
    val `0.5` = field / (field.one, `2`)
    val pow = CuMatrix.zeros[V](y.rows, y.cols)
    pow := `2`
    val r1 = y - x
    val err = r1 ^:^ pow
    err *= `0.5`
    val grad = -r1

    pow.release()
    r1.release()

    (err, grad)

  }

}



/**
  *
  *   L = -Σ(y * log(e^x / Σe^X))
  *
  * Works for 1-of-K classification, where `y` is the target and `x` the prediction. The target is
  * expressed using hot-vector encoding, e. g. (0, 1, 0, 0) where 1 is the true class. The loss is
  * formulated under a cross-entropy regime. K is set implicitly by the neurons of the last layer.
  * The softmaxed class scores sum up to 1, such that they are interpretable as percent, e. g.
  *
  *   N, K:       1, 4
  *   Target:     (0, 1, 0, 0)
  *   Prediction: (0.2, 0.4, 0.3, 0.1)
  *   Loss:       -log(0.4) ≈ 0,916
  *
  */
case class SoftmaxLogEntropy[V]() extends LossFunction[V] {

  def apply(y: DenseMatrix[V], x: DenseMatrix[V])
           (implicit
            field: Field[V], classTag: ClassTag[V],
            _srm: subRowMax.Impl[DenseMatrix[V], DenseMatrix[V]],
            _exp: exp.Impl[DenseMatrix[V], DenseMatrix[V]],
            _sum: sum.Impl[DenseMatrix[V], V],
            _log: log.Impl[DenseMatrix[V], DenseMatrix[V]],
            _abs: abs.Impl[DenseMatrix[V], DenseMatrix[V]],
            _mat: OpMulMatrix.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            _mat2: OpMulMatrix.Impl2[V, DenseMatrix[V], DenseMatrix[V]],
            _mat3: OpMulMatrix.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            _add: OpAdd.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            _sub: OpSub.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            _neg: OpNeg.Impl[DenseMatrix[V], DenseMatrix[V]],
            _pow: OpPow.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            _mulInPl: OpMulScalar.InPlaceImpl2[DenseMatrix[V], V],
            _setInPl: OpSet.InPlaceImpl2[DenseMatrix[V], V],
            _powInPl: OpPow.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            _subInPl: OpSub.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            _addInPl: OpAdd.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]]): (DenseMatrix[V], DenseMatrix[V]) = {

    val probs = SoftmaxImpl(x)
    val err = -(y *:* _log(probs))
    val grad = probs - y

    (err, grad)

  }

  def apply(y: CuMatrix[V], x: CuMatrix[V])
           (implicit
            handle: cublasHandle, field: Field[V], classTag: ClassTag[V],
            _srm: subRowMax.Impl[CuMatrix[V], CuMatrix[V]],
            _exp: exp.Impl[CuMatrix[V], CuMatrix[V]],
            _sum: sum.Impl[CuMatrix[V], V],
            _log: log.Impl[CuMatrix[V], CuMatrix[V]],
            _abs: abs.Impl[CuMatrix[V], CuMatrix[V]],
            _mat: OpMulMatrix.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _mat2: OpMulMatrix.Impl2[V, CuMatrix[V], CuMatrix[V]],
            _mat3: OpMulMatrix.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _add: OpAdd.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _sub: OpSub.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _sub2: OpSub.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _mul1: OpMulScalar.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _mul2: OpMulScalar.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _neg: OpNeg.Impl[CuMatrix[V], CuMatrix[V]],
            _pow: OpPow.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _div: OpDiv.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _mulInPl: OpMulScalar.InPlaceImpl2[CuMatrix[V], V],
            _setInPl: OpSet.InPlaceImpl2[CuMatrix[V], V],
            _powInPl: OpPow.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            _subInPl: OpSub.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            _addInPl: OpAdd.InPlaceImpl2[CuMatrix[V], CuMatrix[V]]): (CuMatrix[V], CuMatrix[V]) = {

    val probs = SoftmaxImpl(x)
    val err = -(y *:* _log(probs))
    val grad = probs - y

    probs.release()

    (err, grad)

  }

  override def sink(x: DenseMatrix[V])(implicit field: Field[V], classTag: ClassTag[V], _srm: subRowMax.Impl[DenseMatrix[V], DenseMatrix[V]], _exp: numerics.exp.Impl[DenseMatrix[V], DenseMatrix[V]], _sum: sum.Impl[DenseMatrix[V], V], _log: numerics.log.Impl[DenseMatrix[V], DenseMatrix[V]], _abs: numerics.abs.Impl[DenseMatrix[V], DenseMatrix[V]], _mat: linalg.operators.OpMulMatrix.Impl2[DenseMatrix[V], V, DenseMatrix[V]], _mat2: linalg.operators.OpMulMatrix.Impl2[V, DenseMatrix[V], DenseMatrix[V]], _mat3: linalg.operators.OpMulMatrix.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]], _add: linalg.operators.OpAdd.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
                                       _sub: linalg.operators.OpSub.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]], _neg: linalg.operators.OpNeg.Impl[DenseMatrix[V], DenseMatrix[V]], _pow: linalg.operators.OpPow.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]], _mulInPl: linalg.operators.OpMulScalar.InPlaceImpl2[DenseMatrix[V], V], _setInPl: linalg.operators.OpSet.InPlaceImpl2[DenseMatrix[V], V], _powInPl: linalg.operators.OpPow.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]], _subInPl: linalg.operators.OpSub.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]], _addInPl: linalg.operators.OpAdd.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]]): DenseMatrix[V] =
    SoftmaxImpl(x)

}



/**
  *
  *   L = -Σ(y * log(e^x / (Σe^X / N)))
  *
  * Works for `N`-of-K classification, where `y` is the target and `x` the prediction. The target is
  * expressed using hot-vector encoding, i. e. (0, 1, 0, 1) where ones are the true classes. The loss is
  * formulated under a cross-entropy regime. The softmaxed class scores sum up to `N`, K is set implicitly by
  * the neurons of the last layer. Example:
  *
  *   N, K:       2, 4
  *   Target:     (0, 1, 0, 1)
  *   Prediction: (0.4, 0.8, 0.5, 0.3)
  *   Loss:       -log(0.8) + -log(0.3) ≈ 1,4271
  *
  */
case class SoftmaxLogMultEntropy[V](N: Int)(implicit cp: Int CanProduce V) extends LossFunction[V] {

  private val _N = cp.apply(N)

  def apply(y: DenseMatrix[V], x: DenseMatrix[V])
           (implicit
            field: Field[V], classTag: ClassTag[V],
            _srm: subRowMax.Impl[DenseMatrix[V], DenseMatrix[V]],
            _exp: exp.Impl[DenseMatrix[V], DenseMatrix[V]],
            _sum: sum.Impl[DenseMatrix[V], V],
            _log: log.Impl[DenseMatrix[V], DenseMatrix[V]],
            _abs: abs.Impl[DenseMatrix[V], DenseMatrix[V]],
            _mat: OpMulMatrix.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            _mat2: OpMulMatrix.Impl2[V, DenseMatrix[V], DenseMatrix[V]],
            _mat3: OpMulMatrix.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            _add: OpAdd.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            _sub: OpSub.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            _neg: OpNeg.Impl[DenseMatrix[V], DenseMatrix[V]],
            _pow: OpPow.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            _mulInPl: OpMulScalar.InPlaceImpl2[DenseMatrix[V], V],
            _setInPl: OpSet.InPlaceImpl2[DenseMatrix[V], V],
            _powInPl: OpPow.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            _subInPl: OpSub.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            _addInPl: OpAdd.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]]): (DenseMatrix[V], DenseMatrix[V]) = {

    val probs = SoftmaxImpl(x)
    probs *= _N
    val err = -(y *:* _log(probs))
    val grad = probs - y

    (err, grad)

  }

  def apply(y: CuMatrix[V], x: CuMatrix[V])
           (implicit
            handle: cublasHandle, field: Field[V], classTag: ClassTag[V],
            _srm: subRowMax.Impl[CuMatrix[V], CuMatrix[V]],
            _exp: exp.Impl[CuMatrix[V], CuMatrix[V]],
            _sum: sum.Impl[CuMatrix[V], V],
            _log: log.Impl[CuMatrix[V], CuMatrix[V]],
            _abs: abs.Impl[CuMatrix[V], CuMatrix[V]],
            _mat: OpMulMatrix.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _mat2: OpMulMatrix.Impl2[V, CuMatrix[V], CuMatrix[V]],
            _mat3: OpMulMatrix.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _add: OpAdd.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _sub: OpSub.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _sub2: OpSub.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _mul1: OpMulScalar.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _mul2: OpMulScalar.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _neg: OpNeg.Impl[CuMatrix[V], CuMatrix[V]],
            _pow: OpPow.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _div: OpDiv.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            _mulInPl: OpMulScalar.InPlaceImpl2[CuMatrix[V], V],
            _setInPl: OpSet.InPlaceImpl2[CuMatrix[V], V],
            _powInPl: OpPow.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            _subInPl: OpSub.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            _addInPl: OpAdd.InPlaceImpl2[CuMatrix[V], CuMatrix[V]]): (CuMatrix[V], CuMatrix[V]) = {

    val probs = SoftmaxImpl(x)
    probs *= _N
    val err = -(y *:* _log(probs))
    val grad = probs - y

    probs.release()

    (err, grad)

  }

  override def sink(x: DenseMatrix[V])(implicit field: Field[V], classTag: ClassTag[V], _srm: subRowMax.Impl[DenseMatrix[V], DenseMatrix[V]], _exp: numerics.exp.Impl[DenseMatrix[V], DenseMatrix[V]], _sum: sum.Impl[DenseMatrix[V], V], _log: numerics.log.Impl[DenseMatrix[V], DenseMatrix[V]], _abs: numerics.abs.Impl[DenseMatrix[V], DenseMatrix[V]], _mat: linalg.operators.OpMulMatrix.Impl2[DenseMatrix[V], V, DenseMatrix[V]], _mat2: linalg.operators.OpMulMatrix.Impl2[V, DenseMatrix[V], DenseMatrix[V]], _mat3: linalg.operators.OpMulMatrix.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]], _add: linalg.operators.OpAdd.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
                                       _sub: linalg.operators.OpSub.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]], _neg: linalg.operators.OpNeg.Impl[DenseMatrix[V], DenseMatrix[V]], _pow: linalg.operators.OpPow.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]], _mulInPl: linalg.operators.OpMulScalar.InPlaceImpl2[DenseMatrix[V], V], _setInPl: linalg.operators.OpSet.InPlaceImpl2[DenseMatrix[V], V], _powInPl: linalg.operators.OpPow.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]], _subInPl: linalg.operators.OpSub.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]], _addInPl: linalg.operators.OpAdd.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]]): DenseMatrix[V] = {
      val probs = SoftmaxImpl(x)
      probs *= _N
      probs
    }

}



/**
  * Computes e^x / Σe^X for given matrix `x` by row.
  * To not numerically overflow, rows are subtracted by row max.
  */
object SoftmaxImpl {

  def apply[V: ClassTag : Zero : Semiring](x: DenseMatrix[V])
              (implicit
               _srm: subRowMax.Impl[DenseMatrix[V], DenseMatrix[V]],
               _exp: exp.Impl[DenseMatrix[V], DenseMatrix[V]],
               _sum: sum.Impl[DenseMatrix[V], V],
               _sub: OpSub.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
               _div: OpDiv.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]]): DenseMatrix[V] = {
    val r1 = _srm(x)
    val r2 = _exp(r1)
    val id = DenseMatrix.ones[V](x.cols, x.cols)
    val sm = r2 * id
    val probs = r2 / sm
    probs
  }

  def apply[V: ClassTag : Zero : Semiring](x: CuMatrix[V])
              (implicit
               _cs: OpSet.InPlaceImpl2[CuMatrix[V],V],
               _srm: subRowMax.Impl[CuMatrix[V], CuMatrix[V]],
               _exp: exp.Impl[CuMatrix[V], CuMatrix[V]],
               _mat: OpMulMatrix.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
               _sum: sum.Impl[CuMatrix[V], V],
               _sub: OpSub.Impl2[CuMatrix[V], V, CuMatrix[V]],
               _div: OpDiv.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]]): CuMatrix[V] = {
    val r1 = _srm(x)
    val r2 = _exp(r1)
    val id = CuMatrix.ones[V](x.cols, x.cols)
    val sm = r2 * id
    val probs = r2 / sm
    r1.release()
    r2.release()
    sm.release()
    id.release()
    probs
  }

}

