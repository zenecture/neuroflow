package neuroflow.core

import breeze.linalg._
import breeze.linalg.operators._
import breeze.math.Field
import breeze.numerics._
import jcuda.jcublas.cublasHandle
import neuroflow.nets.gpu.cuda.CuMatrix

import scala.reflect.ClassTag

/**
  * @author bogdanski
  * @since 02.10.17
  */

/**
  * A loss function gets target `y`, prediction `x` and computes loss and gradient,
  * which will be backpropped into the raw output layer of a net.
  */
trait Loss[V] {

  def apply(y: DenseMatrix[V], x: DenseMatrix[V])
           (implicit
            field: Field[V], classTag: ClassTag[V],
            _max: max.Impl[DenseMatrix[V], V], _exp: exp.Impl[DenseMatrix[V], DenseMatrix[V]],
            _sum: sum.Impl[DenseMatrix[V], V], _log: log.Impl[V, V],
            mat: OpMulMatrix.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            mat2: OpMulMatrix.Impl2[V, DenseMatrix[V], DenseMatrix[V]],
            mat3: OpMulMatrix.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            add: OpAdd.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            sub: OpSub.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            neg: OpNeg.Impl[DenseMatrix[V], DenseMatrix[V]],
            pow: OpPow.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            mulInPl: OpMulScalar.InPlaceImpl2[DenseMatrix[V], V],
            setInPl: OpSet.InPlaceImpl2[DenseMatrix[V], V],
            powInPl: OpPow.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            subInPl: OpSub.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            addInPl: OpAdd.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]]): (DenseMatrix[V], DenseMatrix[V])

  def apply(y: CuMatrix[V], x: CuMatrix[V])
           (implicit
            handle: cublasHandle,
            field: Field[V], classTag: ClassTag[V],
            _max: max.Impl[CuMatrix[V], V], _exp: exp.Impl[CuMatrix[V], CuMatrix[V]],
            _sum: sum.Impl[CuMatrix[V], V], _log: log.Impl[V, V],
            mat: OpMulMatrix.Impl2[CuMatrix[V], V, CuMatrix[V]],
            mat2: OpMulMatrix.Impl2[V, CuMatrix[V], CuMatrix[V]],
            mat3: OpMulMatrix.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            add: OpAdd.Impl2[CuMatrix[V], V, CuMatrix[V]],
            sub: OpSub.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            sub2: OpSub.Impl2[CuMatrix[V], V, CuMatrix[V]],
            mul1: OpMulScalar.Impl2[CuMatrix[V], V, CuMatrix[V]],
            mul2: OpMulScalar.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            neg: OpNeg.Impl[CuMatrix[V], CuMatrix[V]],
            pow: OpPow.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            div: OpDiv.Impl2[CuMatrix[V], V, CuMatrix[V]],
            mulInPl: OpMulScalar.InPlaceImpl2[CuMatrix[V], V],
            setInPl: OpSet.InPlaceImpl2[CuMatrix[V], V],
            powInPl: OpPow.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            subInPl: OpSub.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            addInPl: OpAdd.InPlaceImpl2[CuMatrix[V], CuMatrix[V]]): (CuMatrix[V], CuMatrix[V])

}


case class SquaredMeanError[V]() extends Loss[V] {

  def apply(y: DenseMatrix[V], x: DenseMatrix[V])
           (implicit
            field: Field[V], classTag: ClassTag[V],
            _max: max.Impl[DenseMatrix[V], V], _exp: exp.Impl[DenseMatrix[V], DenseMatrix[V]],
            _sum: sum.Impl[DenseMatrix[V], V], _log: log.Impl[V, V],
            mat: OpMulMatrix.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            mat2: OpMulMatrix.Impl2[V, DenseMatrix[V], DenseMatrix[V]],
            mat3: OpMulMatrix.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            add: OpAdd.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            sub: OpSub.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            neg: OpNeg.Impl[DenseMatrix[V], DenseMatrix[V]],
            pow: OpPow.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            mulInPl: OpMulScalar.InPlaceImpl2[DenseMatrix[V], V],
            setInPl: OpSet.InPlaceImpl2[DenseMatrix[V], V],
            powInPl: OpPow.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            subInPl: OpSub.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            addInPl: OpAdd.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]]): (DenseMatrix[V], DenseMatrix[V]) = {

    val `2`   = field + (field.one, field.one)
    val `0.5` = field / (field.one, `2`)
    val exp   = DenseMatrix.zeros[V](1, y.cols)
    exp := `2`
    val r1 = y - x
    val r2 = r1 ^:^ exp
    r2 *= `0.5`

    (r2, -r1)

  }

  def apply(y: CuMatrix[V], x: CuMatrix[V])
           (implicit
            handle: cublasHandle,
            field: Field[V], classTag: ClassTag[V],
            _max: max.Impl[CuMatrix[V], V], _exp: exp.Impl[CuMatrix[V], CuMatrix[V]],
            _sum: sum.Impl[CuMatrix[V], V], _log: log.Impl[V, V],
            mat: OpMulMatrix.Impl2[CuMatrix[V], V, CuMatrix[V]],
            mat2: OpMulMatrix.Impl2[V, CuMatrix[V], CuMatrix[V]],
            mat3: OpMulMatrix.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            add: OpAdd.Impl2[CuMatrix[V], V, CuMatrix[V]],
            sub: OpSub.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            sub2: OpSub.Impl2[CuMatrix[V], V, CuMatrix[V]],
            mul1: OpMulScalar.Impl2[CuMatrix[V], V, CuMatrix[V]],
            mul2: OpMulScalar.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            neg: OpNeg.Impl[CuMatrix[V], CuMatrix[V]],
            pow: OpPow.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            div: OpDiv.Impl2[CuMatrix[V], V, CuMatrix[V]],
            mulInPl: OpMulScalar.InPlaceImpl2[CuMatrix[V], V],
            setInPl: OpSet.InPlaceImpl2[CuMatrix[V], V],
            powInPl: OpPow.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            subInPl: OpSub.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            addInPl: OpAdd.InPlaceImpl2[CuMatrix[V], CuMatrix[V]]): (CuMatrix[V], CuMatrix[V]) = {

    val `2`   = field + (field.one, field.one)
    val `0.5` = field / (field.one, `2`)
    val exp   = CuMatrix.zeros[V](1, y.cols)
    exp := `2`
    val r1 = y - x
    val r2 = r1 ^:^ exp
    r2 *= `0.5`
    val r3 = -r1

    exp.release()
    r1.release()

    (r2, r3)

  }

}

/**
  * Works for 1-of-K softmax classification, under a cross-entropy regime.
  */
case class Softmax[V]() extends Loss[V] {

  def apply(y: DenseMatrix[V], x: DenseMatrix[V])
           (implicit
            field: Field[V], classTag: ClassTag[V],
            _max: max.Impl[DenseMatrix[V], V], _exp: exp.Impl[DenseMatrix[V], DenseMatrix[V]],
            _sum: sum.Impl[DenseMatrix[V], V], _log: log.Impl[V, V],
            mat: OpMulMatrix.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            mat2: OpMulMatrix.Impl2[V, DenseMatrix[V], DenseMatrix[V]],
            mat3: OpMulMatrix.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            add: OpAdd.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            sub: OpSub.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            neg: OpNeg.Impl[DenseMatrix[V], DenseMatrix[V]],
            pow: OpPow.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            mulInPl: OpMulScalar.InPlaceImpl2[DenseMatrix[V], V],
            setInPl: OpSet.InPlaceImpl2[DenseMatrix[V], V],
            powInPl: OpPow.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            subInPl: OpSub.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            addInPl: OpAdd.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]]): (DenseMatrix[V], DenseMatrix[V]) = {

    val probs = SoftmaxImpl(x)
    val mask = _sum(y *:* probs)
    val err = y *:* -_log(mask)
    val grad = probs - y

    (err, grad)

  }

  def apply(y: CuMatrix[V], x: CuMatrix[V])
           (implicit
            handle: cublasHandle,
            field: Field[V], classTag: ClassTag[V],
            _max: max.Impl[CuMatrix[V], V], _exp: exp.Impl[CuMatrix[V], CuMatrix[V]],
            _sum: sum.Impl[CuMatrix[V], V], _log: log.Impl[V, V],
            mat: OpMulMatrix.Impl2[CuMatrix[V], V, CuMatrix[V]],
            mat2: OpMulMatrix.Impl2[V, CuMatrix[V], CuMatrix[V]],
            mat3: OpMulMatrix.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            add: OpAdd.Impl2[CuMatrix[V], V, CuMatrix[V]],
            sub: OpSub.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            sub2: OpSub.Impl2[CuMatrix[V], V, CuMatrix[V]],
            mul1: OpMulScalar.Impl2[CuMatrix[V], V, CuMatrix[V]],
            mul2: OpMulScalar.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            neg: OpNeg.Impl[CuMatrix[V], CuMatrix[V]],
            pow: OpPow.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            div: OpDiv.Impl2[CuMatrix[V], V, CuMatrix[V]],
            mulInPl: OpMulScalar.InPlaceImpl2[CuMatrix[V], V],
            setInPl: OpSet.InPlaceImpl2[CuMatrix[V], V],
            powInPl: OpPow.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            subInPl: OpSub.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            addInPl: OpAdd.InPlaceImpl2[CuMatrix[V], CuMatrix[V]]): (CuMatrix[V], CuMatrix[V]) = {

    val probs = SoftmaxImpl(x)
    val r1 = y *:* probs
    val mask = _sum(r1)
    val err = y *:* -_log(mask)
    val grad = probs - y

    probs.release()
    r1.release()

    (err, grad)

  }

}

object SoftmaxImpl {

  def apply[V](x: DenseMatrix[V])
              (implicit
               _max: max.Impl[DenseMatrix[V], V],
               _exp: exp.Impl[DenseMatrix[V], DenseMatrix[V]],
               _sum: sum.Impl[DenseMatrix[V], V],
               sub: OpSub.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
               div: OpDiv.Impl2[DenseMatrix[V], V, DenseMatrix[V]]): DenseMatrix[V] = {
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
               sub: OpSub.Impl2[CuMatrix[V], V, CuMatrix[V]],
               div: OpDiv.Impl2[CuMatrix[V], V, CuMatrix[V]]): CuMatrix[V] = {
    val r1 = x - _max(x)
    val r2 = _exp(r1)
    val r3 = _sum(r2)
    val r4 = r2 / r3
    r1.release()
    r2.release()
    r4
  }

}
