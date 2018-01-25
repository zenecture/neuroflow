package neuroflow.core

import breeze.linalg._
import breeze.linalg.operators._
import jcuda.jcublas._
import neuroflow.cuda._

/**
  * @author bogdanski
  * @since 09.09.17
  */


/**
  * Updates weights `ws` using derivatives `dws` and `learningRate` for layer `position`.
  */
trait Update[V] {

  def apply(ws: DenseMatrix[V], dws: DenseMatrix[V], learningRate: V, position: Int)
           (implicit
            mul: OpMulScalar.InPlaceImpl2[DenseMatrix[V], V],
            mat: OpMulMatrix.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            mat2: OpMulMatrix.Impl2[V, DenseMatrix[V], DenseMatrix[V]],
            mat3: OpMulMatrix.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            add: OpAdd.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            sub: OpSub.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            neg: OpNeg.Impl[DenseMatrix[V], DenseMatrix[V]],
            subInPl: OpSub.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            addInPl: OpAdd.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]]): Unit

  def apply(ws: CuMatrix[V], dws: CuMatrix[V], learningRate: V, position: Int)
           (implicit
            handle: cublasHandle,
            mul: OpMulScalar.InPlaceImpl2[CuMatrix[V], V],
            mat: OpMulMatrix.Impl2[CuMatrix[V], V, CuMatrix[V]],
            mat2: OpMulMatrix.Impl2[V, CuMatrix[V], CuMatrix[V]],
            mat3: OpMulMatrix.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            add: OpAdd.Impl2[CuMatrix[V], V, CuMatrix[V]],
            sub: OpSub.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            neg: OpNeg.Impl[CuMatrix[V], CuMatrix[V]],
            subInPl: OpSub.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            addInPl: OpAdd.InPlaceImpl2[CuMatrix[V], CuMatrix[V]]): Unit

}


/**
  * Gingerly stepping vanilla.
  *   Weights_{n} = Weights_{n-1} - (Grads_{n-1} * learningRate)
  */
case class Vanilla[V]() extends Update[V] {

  def apply(ws: DenseMatrix[V], dws: DenseMatrix[V], learningRate: V, position: Int)
           (implicit
            mul: OpMulScalar.InPlaceImpl2[DenseMatrix[V], V],
            mat: OpMulMatrix.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            mat2: OpMulMatrix.Impl2[V, DenseMatrix[V], DenseMatrix[V]],
            mat3: OpMulMatrix.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            add: OpAdd.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            sub: OpSub.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            neg: OpNeg.Impl[DenseMatrix[V], DenseMatrix[V]],
            subInPl: OpSub.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            addInPl: OpAdd.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]]): Unit = {
    ws -= (dws *= learningRate)
  }

  def apply(ws: CuMatrix[V], dws: CuMatrix[V], learningRate: V, position: Int)
           (implicit
            handle: cublasHandle,
            mul: OpMulScalar.InPlaceImpl2[CuMatrix[V], V],
            mat: OpMulMatrix.Impl2[CuMatrix[V], V, CuMatrix[V]],
            mat2: OpMulMatrix.Impl2[V, CuMatrix[V], CuMatrix[V]],
            mat3: OpMulMatrix.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            add: OpAdd.Impl2[CuMatrix[V], V, CuMatrix[V]],
            sub: OpSub.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            neg: OpNeg.Impl[CuMatrix[V], CuMatrix[V]],
            subInPl: OpSub.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            addInPl: OpAdd.InPlaceImpl2[CuMatrix[V], CuMatrix[V]]): Unit = {
    ws -= (dws *= learningRate)
  }

}


/**
  * Momentum update is jumping downhill into the loss' minimum, iteratively re-gaining
  * momentum into all directions by varying gradients, which is decelerated by factor `μ`.
  */
case class Momentum[V](μ: V) extends Update[V] {

  private val vsd = collection.mutable.HashMap.empty[Int, DenseMatrix[V]]
  private val vsc = collection.mutable.HashMap.empty[Int, CuMatrix[V]]

  def apply(ws: DenseMatrix[V], dws: DenseMatrix[V], learningRate: V, position: Int)
           (implicit
            mul: OpMulScalar.InPlaceImpl2[DenseMatrix[V], V],
            mat: OpMulMatrix.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            mat2: OpMulMatrix.Impl2[V, DenseMatrix[V], DenseMatrix[V]],
            mat3: OpMulMatrix.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            add: OpAdd.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            sub: OpSub.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            neg: OpNeg.Impl[DenseMatrix[V], DenseMatrix[V]],
            subInPl: OpSub.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            addInPl: OpAdd.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]]): Unit = {
    if (vsd.isDefinedAt(position)) vsd(position) := (μ * vsd(position)) - (dws *= learningRate)
    else vsd += position -> -(dws * learningRate)
    ws += vsd(position)
  }

  def apply(ws: CuMatrix[V], dws: CuMatrix[V], learningRate: V, position: Int)
           (implicit
            handle: cublasHandle,
            mul: OpMulScalar.InPlaceImpl2[CuMatrix[V], V],
            mat: OpMulMatrix.Impl2[CuMatrix[V], V, CuMatrix[V]],
            mat2: OpMulMatrix.Impl2[V, CuMatrix[V], CuMatrix[V]],
            mat3: OpMulMatrix.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            add: OpAdd.Impl2[CuMatrix[V], V, CuMatrix[V]],
            sub: OpSub.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            neg: OpNeg.Impl[CuMatrix[V], CuMatrix[V]],
            subInPl: OpSub.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            addInPl: OpAdd.InPlaceImpl2[CuMatrix[V], CuMatrix[V]]): Unit = {
    if (vsc.isDefinedAt(position)) {
      val r1 = μ * vsc(position)
      val r2 = r1 - (dws *= learningRate)
      vsc(position) := r2
      r1.release()
      r2.release()
    } else vsc += position -> {
      val r1 = dws * learningRate
      val r2 = -r1
      r1.release()
      r2
    }
    ws += vsc(position)
  }

  def release(): Unit = {
    vsc.values.foreach(_.release())
    vsc.values.foreach(_.release())
  }

}


/**
  * Exposes the `lastGradients` for debugging.
  */
case class Debuggable[V]() extends Update[V] {

  var lastGradients = collection.mutable.Map.empty[Int, DenseMatrix[V]]

  def apply(ws: DenseMatrix[V], dws: DenseMatrix[V], learningRate: V, position: Int)
           (implicit
            mul: OpMulScalar.InPlaceImpl2[DenseMatrix[V], V],
            mat: OpMulMatrix.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            mat2: OpMulMatrix.Impl2[V, DenseMatrix[V], DenseMatrix[V]],
            mat3: OpMulMatrix.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            add: OpAdd.Impl2[DenseMatrix[V], V, DenseMatrix[V]],
            sub: OpSub.Impl2[DenseMatrix[V], DenseMatrix[V], DenseMatrix[V]],
            neg: OpNeg.Impl[DenseMatrix[V], DenseMatrix[V]],
            subInPl: OpSub.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]],
            addInPl: OpAdd.InPlaceImpl2[DenseMatrix[V], DenseMatrix[V]]): Unit = {
    lastGradients += position -> dws.copy
    ws -= (dws *= learningRate)
  }

  def apply(ws: CuMatrix[V], dws: CuMatrix[V], learningRate: V, position: Int)
           (implicit
            handle: cublasHandle,
            mul: OpMulScalar.InPlaceImpl2[CuMatrix[V], V],
            mat: OpMulMatrix.Impl2[CuMatrix[V], V, CuMatrix[V]],
            mat2: OpMulMatrix.Impl2[V, CuMatrix[V], CuMatrix[V]],
            mat3: OpMulMatrix.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            add: OpAdd.Impl2[CuMatrix[V], V, CuMatrix[V]],
            sub: OpSub.Impl2[CuMatrix[V], CuMatrix[V], CuMatrix[V]],
            neg: OpNeg.Impl[CuMatrix[V], CuMatrix[V]],
            subInPl: OpSub.InPlaceImpl2[CuMatrix[V], CuMatrix[V]],
            addInPl: OpAdd.InPlaceImpl2[CuMatrix[V], CuMatrix[V]]): Unit = {
    lastGradients += position -> dws.toDense
    ws -= (dws *= learningRate)
  }

}
