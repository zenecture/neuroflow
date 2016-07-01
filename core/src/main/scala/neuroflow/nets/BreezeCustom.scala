package neuroflow.nets

import breeze.linalg._
import breeze.math._
import breeze.optimize.FirstOrderMinimizer._
import breeze.optimize._
import neuroflow.nets.NFLBFGS.ErrorFunctionMin

/**
  * Breeze related custom classes.
  *
  * @author bogdanski
  * @since 01.07.16
  */


private[nets] class NFLBFGS(cc: ConvergenceCheck[DenseVector[Double]], m: Int, maxZoomIter: Int, maxLineSearchIter: Int)
                           (implicit space: MutableInnerProductModule[DenseVector[Double], Double]) extends LBFGS[DenseVector[Double]](cc, m)(space) {

  def this(maxIter: Int = -1, m: Int = 7, tolerance: Double = 1E-5, maxZoomIter: Int, maxLineSearchIter: Int)(implicit space: MutableInnerProductModule[DenseVector[Double], Double]) =
    this(NFLBFGS.defaultConvergenceCheck(maxIter, tolerance), m, maxZoomIter, maxLineSearchIter)

  override protected def determineStepSize(state: State, f: DiffFunction[DenseVector[Double]], dir: DenseVector[Double]): Double = {
    val x = state.x
    val ff = LineSearch.functionFromSearchDirection(f, x, dir)
    val search = new StrongWolfeLineSearch(maxZoomIter, maxLineSearchIter)
    search.minimize(ff, if(state.iter == 0.0) 1.0 / norm(dir) else 1.0)
  }

}


private[nets] object NFLBFGS {

  import FirstOrderMinimizer._

  def defaultConvergenceCheck[T](maxIter: Int, tolerance: Double, relative: Boolean = false, fvalMemory: Int = 20)
                                (implicit space: NormedModule[T, Double]): ConvergenceCheck[T] =
    maxIterationsReached[T](maxIter) ||
      ErrorFunctionValue(lessThan = tolerance, historyLength = 10) ||
      searchFailed

  case object ErrorFunctionMin extends ConvergenceReason {
    def reason = "error function is sufficiently minimal."
  }

}


private[nets] case class ErrorFunctionValue[T](lessThan: Double, historyLength: Int) extends ConvergenceCheck[T] {

  type Info = IndexedSeq[Double]

  def update(newX: T, newGrad: T, newVal: Double, oldState: State[T, _, _], oldInfo: Info): Info =
    (oldInfo :+ newVal).takeRight(historyLength)

  def apply(state: State[T, _, _], info: IndexedSeq[Double]): Option[ConvergenceReason] =
    if (info.length >= 2 && (state.adjustedValue <= lessThan)) Some(ErrorFunctionMin) else None

  def initialInfo: Info = IndexedSeq(Double.PositiveInfinity)

}
