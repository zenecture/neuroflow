package neuroflow.nets.distributed

import akka.actor.{Actor, ActorRef, ActorSystem, Props}
import breeze.linalg.DenseMatrix
import neuroflow.common.Logs
import neuroflow.core.Network.{Matrix, _}
import neuroflow.core._

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool
import scala.util.{Failure, Success}

/**
  * @author bogdanski
  * @since 28.08.17
  */

object DefaultExecutor extends Logs {

  def apply(node: Node, xs: Vectors, ys: Vectors, settings: Settings = Settings()): Unit = {

    info(s"Booting DefaultExecutor ${node.host}:${node.port} ...")

    val _akka = ActorSystem("NeuroFlow", Configuration(node, settings))
    _akka.actorOf(Props(new DefaultExecutor(xs, ys, settings)), "executor")

    info("Up and running.")
    info("Type 'exit' to exit.")

    while (scala.io.StdIn.readLine() != "exit") { }

    import _akka.dispatcher

    _akka.terminate().onComplete {
      case Success(_) => System.exit(0)
      case Failure(_) => System.exit(1)
    }

  }

}


class DefaultExecutor(xs: Vectors, ys: Vectors, settings: Settings) extends ExecutorActor[Vectors, Vectors](xs, ys, settings) {

  protected def compute(xs: Vectors, ys: Vectors, layers: Seq[Layer], weights: ArrayBuffer[Matrix],
                      learningRate: Double, parallelism: Int): (Weights, Matrix) = {

    val _xsys = xs.par.zip(ys).map { case (a, b) => (a.asDenseMatrix, b.asDenseMatrix)}
    _xsys.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(parallelism))

    val _layersNI       = layers.tail.map { case h: HasActivator[Double] => h }
    val _outputDim      = layers.last.neurons
    val _lastWlayerIdx  = weights.size - 1

    val _ds = (0 to _lastWlayerIdx).map { i =>
      i -> DenseMatrix.zeros[Double](weights(i).rows, weights(i).cols)
    }.toMap

    val _errSum = DenseMatrix.zeros[Double](1, _outputDim)
    val _square = DenseMatrix.zeros[Double](1, _outputDim)
    _square := 2.0

    _xsys.map { xy =>
      val (x, y) = xy
      val fa  = collection.mutable.Map.empty[Int, Matrix]
      val fb  = collection.mutable.Map.empty[Int, Matrix]
      val dws = collection.mutable.Map.empty[Int, Matrix]
      val ds  = collection.mutable.Map.empty[Int, Matrix]
      val e   = DenseMatrix.zeros[Double](1, _outputDim)

      @tailrec def forward(in: Matrix, i: Int): Unit = {
        val p = in * weights(i)
        val a = p.map(_layersNI(i).activator)
        val b = p.map(_layersNI(i).activator.derivative)
        fa += i -> a
        fb += i -> b
        if (i < _lastWlayerIdx) forward(a, i + 1)
      }

      @tailrec def derive(i: Int): Unit = {
        if (i == 0 && _lastWlayerIdx == 0) {
          val yf = y - fa(0)
          val d = -yf *:* fb(0)
          val dw = x.t * d
          dws += 0 -> dw
          e += yf
        } else if (i == _lastWlayerIdx) {
          val yf = y - fa(i)
          val d = -yf *:* fb(i)
          val dw = fa(i - 1).t * d
          dws += i -> dw
          ds += i -> d
          e += yf
          derive(i - 1)
        } else if (i < _lastWlayerIdx && i > 0) {
          val d = (ds(i + 1) * weights(i + 1).t) *:* fb(i)
          val dw = fa(i - 1).t * d
          dws += i -> dw
          ds += i -> d
          derive(i - 1)
        } else if (i == 0) {
          val d = (ds(i + 1) * weights(i + 1).t) *:* fb(i)
          val dw = x.t * d
          dws += i -> dw
        }
      }

      forward(x, 0)
      derive(_lastWlayerIdx)
      e :^= _square
      e *= 0.5
      (dws, e)
    }.seq.foreach { ab =>
      _errSum += ab._2
      var i = 0
      while (i <= _lastWlayerIdx) {
        val m = _ds(i)
        val n = ab._1(i)
        m += n
        i += 1
      }
    }

    (_ds.values.toIndexedSeq, _errSum)

  }

}
