package neuroflow.nets.cpu

import breeze.linalg._
import breeze.math._
import breeze.optimize.FirstOrderMinimizer._
import breeze.optimize._
import neuroflow.common.Logs
import neuroflow.nets.cpu.NFLBFGS.ErrorFunctionMin

/**
  * Breeze related custom classes.
  *
  * @author bogdanski
  * @since 01.07.16
  */


private[nets] class NFLBFGS(verbose: Boolean, cc: ConvergenceCheck[DenseVector[Double]], m: Int, maxZoomIter: Int, maxLineSearchIter: Int, maybeGraph: (Double) => Unit)
                           (implicit space: MutableInnerProductModule[DenseVector[Double], Double]) extends LBFGS[DenseVector[Double]](cc, m)(space) with Logs {

  def this(verbose: Boolean, maxIter: Int = -1, m: Int = 7, tolerance: Double = 1E-5, maxZoomIter: Int, maxLineSearchIter: Int, maybeGraph: (Double) => Unit)
          (implicit space: MutableInnerProductModule[DenseVector[Double], Double]) =
    this(verbose, NFLBFGS.defaultConvergenceCheck(maxIter, tolerance), m, maxZoomIter, maxLineSearchIter, maybeGraph)

  override protected def determineStepSize(state: State, f: DiffFunction[DenseVector[Double]], dir: DenseVector[Double]): Double = {
    val x = state.x
    val ff = LineSearch.functionFromSearchDirection(f, x, dir)
    val search = new StrongWolfeLineSearch(maxZoomIter, maxLineSearchIter)
    search.minimize(ff, if(state.iter == 0.0) 1.0 / norm(dir) else 1.0)
  }

  override def infiniteIterations(f: DiffFunction[DenseVector[Double]], state: State): Iterator[State] = {
      var failedOnce = false
      val adjustedFun = adjustFunction(f)
      Iterator.iterate(state) { state =>
        try {
          val dir = chooseDescentDirection(state, adjustedFun)
          val stepSize = determineStepSize(state, adjustedFun, dir)
          val x = takeStep(state, dir, stepSize)
          val (value, grad) = calculateObjective(adjustedFun, x, state.history)
          val (adjValue, adjGrad) = adjust(x, grad, value)
          val oneOffImprovement = (state.adjustedValue - adjValue) / (state.adjustedValue.abs max adjValue.abs max 1E-6 * state.initialAdjVal.abs)
          if (verbose) info(f"Taking step ${state.iter}. Step Size: $stepSize%.4g. Val and Grad Norm: $adjValue%.6g (rel: $oneOffImprovement%.3g) ${norm(adjGrad)}%.6g")
          maybeGraph(adjValue)
          val history = updateHistory(x, grad, value, adjustedFun, state)
          val newCInfo = convergenceCheck.update(x, grad, value, state, state.convergenceInfo)
          failedOnce = false
          FirstOrderMinimizer.State(x, value, grad, adjValue, adjGrad, state.iter + 1, state.initialAdjVal, history, newCInfo)
        } catch {
          case x: FirstOrderException if !failedOnce =>
            failedOnce = true
            info("Failure! Resetting history: " + x)
            state.copy(history = initialHistory(adjustedFun, state.x))
          case x: FirstOrderException =>
            info("Failure again! Giving up and returning. Maybe the objective is just poorly behaved?")
            state.copy(searchFailed = true)
        }
      }
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
    def reason = "Error function is sufficiently minimal."
  }

}


private[nets] case class ErrorFunctionValue[T](lessThan: Double, historyLength: Int) extends ConvergenceCheck[T] with Logs {

  type Info = IndexedSeq[Double]

  def update(newX: T, newGrad: T, newVal: Double, oldState: State[T, _, _], oldInfo: Info): Info =
    (oldInfo :+ newVal).takeRight(historyLength)

  def apply(state: State[T, _, _], info: IndexedSeq[Double]): Option[ConvergenceReason] =
    if (info.length >= 2 && (state.adjustedValue <= lessThan)) Some(ErrorFunctionMin) else None

  def initialInfo: Info = IndexedSeq(Double.PositiveInfinity)

}
