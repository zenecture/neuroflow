package neuroflow.core

import breeze.linalg._
import breeze.linalg.operators._
import breeze.math.Field
import breeze.numerics._
import jcuda.jcublas.cublasHandle
import neuroflow.cuda._
import scala.collection.immutable.{ :: => all }

import scala.reflect.ClassTag

/**
  * @author bogdanski
  * @since 02.10.17
  */

/**
  * A loss function gets target `y`, prediction `x`, computes loss and gradient,
  * which will be backpropped into the raw output layer of a net.
  */
trait LossFunction[V] extends Layout {

  def apply(y: DenseMatrix[V], x: DenseMatrix[V])
           (implicit
            field: Field[V], classTag: ClassTag[V],
            _max: max.Impl[DenseMatrix[V], V],
            _exp: exp.Impl[DenseMatrix[V], DenseMatrix[V]],
            _sum: sum.Impl[DenseMatrix[V], V],
            _log: log.Impl[V, V],
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

  def apply(y: CuMatrix[V], x: CuMatrix[V])
           (implicit
            handle: cublasHandle, field: Field[V], classTag: ClassTag[V],
            _max: max.Impl[CuMatrix[V], V],
            _exp: exp.Impl[CuMatrix[V], CuMatrix[V]],
            _sum: sum.Impl[CuMatrix[V], V],
            _log: log.Impl[V, V],
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
            _div: OpDiv.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _mulInPl: OpMulScalar.InPlaceImpl2[CuMatrix[V], V],
            _setInPl: OpSet.InPlaceImpl2[CuMatrix[V], V],
            _powInPl: OpPow.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            _subInPl: OpSub.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            _addInPl: OpAdd.InPlaceImpl2[CuMatrix[V], CuMatrix[V]]): (CuMatrix[V], CuMatrix[V])

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
case class SquaredMeanError[V]() extends LossFunction[V] {

  def apply(y: DenseMatrix[V], x: DenseMatrix[V])
           (implicit
            field: Field[V], classTag: ClassTag[V],
            _max: max.Impl[DenseMatrix[V], V],
            _exp: exp.Impl[DenseMatrix[V], DenseMatrix[V]],
            _sum: sum.Impl[DenseMatrix[V], V],
            _log: log.Impl[V, V],
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

    val `2`   = field + (field.one, field.one)
    val `0.5` = field / (field.one, `2`)
    val pow   = DenseMatrix.zeros[V](y.rows, y.cols)
    pow := `2`
    val r1 = y - x
    val r2 = r1 ^:^ pow
    r2 *= `0.5`

    (r2, -r1)

  }

  def apply(y: CuMatrix[V], x: CuMatrix[V])
           (implicit
            handle: cublasHandle, field: Field[V], classTag: ClassTag[V],
            _max: max.Impl[CuMatrix[V], V],
            _exp: exp.Impl[CuMatrix[V], CuMatrix[V]],
            _sum: sum.Impl[CuMatrix[V], V],
            _log: log.Impl[V, V],
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
            _div: OpDiv.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _mulInPl: OpMulScalar.InPlaceImpl2[CuMatrix[V], V],
            _setInPl: OpSet.InPlaceImpl2[CuMatrix[V], V],
            _powInPl: OpPow.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            _subInPl: OpSub.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            _addInPl: OpAdd.InPlaceImpl2[CuMatrix[V], CuMatrix[V]]): (CuMatrix[V], CuMatrix[V]) = {

    val `2`   = field + (field.one, field.one)
    val `0.5` = field / (field.one, `2`)
    val pow   = CuMatrix.zeros[V](y.rows, y.cols)
    pow := `2`
    val r1 = y - x
    val r2 = r1 ^:^ pow
    r2 *= `0.5`
    val r3 = -r1

    pow.release()
    r1.release()

    (r2, r3)

  }

}

/**
  *
  *   L = -Σ(y * log(e^x / Σe^X))
  *
  * Works for 1-of-K classification, under a cross-entropy regime,
  * where `y` is the target and `x` the prediction. The target is expressed
  * using hot-vector encoding, e. g. (0, 1, 0, 0) where 1 is the true class.
  * The first sum Σ is taken over the full batch and both exponentials
  * give a convex functional form. The second sum Σ produces scores in
  * range [0.0, 1.0] such that they sum up to 1 and are interpretable
  * as percent.
  *
  */
case class Softmax[V]() extends LossFunction[V] {

  def apply(y: DenseMatrix[V], x: DenseMatrix[V])
           (implicit
            field: Field[V], classTag: ClassTag[V],
            _max: max.Impl[DenseMatrix[V], V],
            _exp: exp.Impl[DenseMatrix[V], DenseMatrix[V]],
            _sum: sum.Impl[DenseMatrix[V], V],
            _log: log.Impl[V, V],
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

    val batchSize = y.rows

    (0 until batchSize).map { i =>
      val (_x, _y) = (x.t(all, i).asDenseMatrix, y.t(all, i).asDenseMatrix)
      val probs = SoftmaxImpl(_x)
      val mask = _sum(_y *:* probs)
      val err = _y *:* -_log(mask)
      val grad = probs - _y
      (err, grad)
    }.reduce { (a, b) =>
      DenseMatrix.vertcat(a._1, b._1) -> DenseMatrix.vertcat(a._2, b._2)
    }

  }

  def apply(y: CuMatrix[V], x: CuMatrix[V])
           (implicit
            handle: cublasHandle, field: Field[V], classTag: ClassTag[V],
            _max: max.Impl[CuMatrix[V], V],
            _exp: exp.Impl[CuMatrix[V], CuMatrix[V]],
            _sum: sum.Impl[CuMatrix[V], V],
            _log: log.Impl[V, V],
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
            _div: OpDiv.Impl2[CuMatrix[V], V, CuMatrix[V]],
            _mulInPl: OpMulScalar.InPlaceImpl2[CuMatrix[V], V],
            _setInPl: OpSet.InPlaceImpl2[CuMatrix[V], V],
            _powInPl: OpPow.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            _subInPl: OpSub.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            _addInPl: OpAdd.InPlaceImpl2[CuMatrix[V], CuMatrix[V]]): (CuMatrix[V], CuMatrix[V]) = {

    val batchSize = y.rows

    val (err, grad) = (0 until batchSize).map { i =>
      val (_x, _y) = (x.t(all, i).t, y.t(all, i).t)
      val probs = SoftmaxImpl(_x)
      val r1 = _y *:* probs
      val mask = _sum(r1)
      val err = _y *:* -_log(mask)
      val grad = probs - _y
      probs.release()
      r1.release()
      (err.toDense, grad.toDense)
    }.reduce { (a, b) =>
      DenseMatrix.vertcat(a._1, b._1) -> DenseMatrix.vertcat(a._2, b._2)
    }

    CuMatrix.fromDense(err) -> CuMatrix.fromDense(grad)

  }

}

object SoftmaxImpl {

  def apply[V](x: DenseMatrix[V])
              (implicit
               _max: max.Impl[DenseMatrix[V], V],
               _exp: exp.Impl[DenseMatrix[V], DenseMatrix[V]],
               _sum: sum.Impl[DenseMatrix[V], V],
               _sub: OpSub.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
               _div: OpDiv.Impl2[DenseMatrix[V], V, DenseMatrix[V]]): DenseMatrix[V] = {
    val r1 = x - _max(x)
    val r2 = _exp(r1)
    val probs = r2 / _sum(r2)
    probs
  }

  def apply[V](x: CuMatrix[V])
              (implicit
               _max: max.Impl[CuMatrix[V], V],
               _exp: exp.Impl[CuMatrix[V], CuMatrix[V]],
               _sum: sum.Impl[CuMatrix[V], V],
               _sub: OpSub.Impl2[CuMatrix[V], V, CuMatrix[V]],
               _div: OpDiv.Impl2[CuMatrix[V], V, CuMatrix[V]]): CuMatrix[V] = {
    val r1 = x - _max(x)
    val r2 = _exp(r1)
    val r3 = _sum(r2)
    val r4 = r2 / r3
    r1.release()
    r2.release()
    r4
  }

}
