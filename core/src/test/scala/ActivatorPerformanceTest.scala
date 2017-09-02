package neuroflow.nets

import org.scalatest.FunSuite

import neuroflow.core.Activator._

/**
  * @author bogdanski
  * @since 22.04.17
  */
class ActivatorPerformanceTest extends FunSuite {

  def withTimer[B](f: => B): Long = {
    val n1 = System.nanoTime()
    f
    val n2 = System.nanoTime()
    n2 - n1
  }

  test("Benchmark Activator Funcs") {

    val funcs = Seq(Linear, Square, Sigmoid, Tanh, CustomSigmoid(1, 1, 1), ReLU, LeakyReLU(0.01), SoftPlus)
    val bench = funcs.map { f =>
      (f, withTimer(f(1.0)), withTimer(f.derivative(1.0)))
    }

    bench.sortBy(_._2).foreach {
      case (f, value, derivative) => println(s"Function: ${f.symbol}, Evaluation: $value, Derivative: $derivative")
    }

  }

}
