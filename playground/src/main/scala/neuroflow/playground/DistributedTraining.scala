package neuroflow.playground

import breeze.linalg.DenseVector
import breeze.stats.distributions.Gaussian
import neuroflow.common.Logs
import neuroflow.core.Activator._
import neuroflow.core._
import neuroflow.nets.distributed.DenseExecutor
import neuroflow.nets.distributed.DenseNetwork._

import scala.util.Random

/**
  * @author bogdanski
  * @since 28.08.17
  */

object DistributedTraining extends Logs {

  val nodesC = 3
  val nodes  = (1 to nodesC).map(i => Node("localhost", 2552 + i)).toSet
  val dim    = 1200
  val out    = 210

  def coordinator = {

    implicit val wp = neuroflow.core.WeightProvider.FFN[Double].random(-1, 1)

    val f   = ReLU

    val net: DistFFN[Double]  =
      Network(
        Vector(dim)               ::
        Dense(out, f)             ::
        Dense(out, f)             ::
        Dense(out, f)             ::
        Dense(dim, f)             :: SquaredMeanError(),
        Settings[Double](
          coordinator  = Node("localhost", 2552),
          transport    = Transport(100000, "128 MiB"),
          learningRate = { case (_, _) => 1E-11 },
          iterations   = 2000,
          prettyPrint  = true
        )
      )

    net.train(nodes)
  }

  val samples = 10

  def executors = {

    val xs = (1 to samples).map { i =>
      DenseVector(Array.fill(dim)(Gaussian.distribution((0.0, i.toDouble / samples.toDouble)).draw().abs))
    }

    val ys = (1 to samples).map { i =>
      val a = Array.fill(dim)(0.0)
      a.update(Random.nextInt(dim), 1.0)
      DenseVector(a)
    }

    (1 to nodesC).par.foreach(i => DenseExecutor(Node("localhost", 2552 + i), xs, ys))

  }

}
