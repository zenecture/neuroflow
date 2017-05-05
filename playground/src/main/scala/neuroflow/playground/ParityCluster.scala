package neuroflow.playground

import java.io.{File, FileOutputStream, PrintWriter}

import neuroflow.application.plugin.Notation.{ζ, _}
import neuroflow.application.processor.Util._
import neuroflow.application.processor.{Extensions, Normalizer, Util}
import neuroflow.common.~>
import neuroflow.core.Activator._
import neuroflow.core._
import neuroflow.nets.LBFGSNetwork._
import shapeless._

import scala.util.Random

/**
  * @author bogdanski
  * @since 17.04.17
  */

object ParityCluster {

  val dimension = 25
  val samples = 1000
  val maxObservations = 10

  val clusterOutput = "/Users/felix/github/unversioned/parityCluster.txt"

  def apply = {

    import Extensions.VectorOps
    import neuroflow.core.FFN.WeightProvider._

    val classes = (0 until dimension) map (i => (i, Random.nextInt(2))) map {
      case (i, k) if i % 2 == 0 && k % 2 == 0 => ("xw", ζ(dimension).updated(i, 1.0) ++ ->(1.0))
      case (i, k) if i % 2 == 0 && k % 2 != 0 => ("xo", ζ(dimension).updated(i, 1.0) ++ ->(0.0))
      case (i, k) if i % 2 != 0 && k % 2 == 0 => ("vw", ζ(dimension).updated(i, 1.0) ++ ->(1.0))
      case (i, k) if i % 2 != 0 && k % 2 != 0 => ("vo", ζ(dimension).updated(i, 1.0) ++ ->(0.0))
    }

    val evens = Range(0, dimension, 2).toVector
    val odds = Range(1, dimension, 2).toVector

    val xsys = {
      (0 until samples) flatMap { _ =>
        Util.shuffle {
          (0 until Random.nextInt(maxObservations)).map { _ =>
            classes(evens(Random.nextInt(evens.size)))._2
          }.toList
        }.map { case (k, v) =>
          k -> (v.map(_.slice(0, dimension)).reduce(_ + _) ++ v.map(_.slice(dimension, dimension + 1)).reduce(_ + _).map(_ / dimension))
        }
      }
    } ++ {
      (0 until samples) flatMap { _ =>
        Util.shuffle {
          (0 until Random.nextInt(maxObservations)).map { _ =>
            classes(odds(Random.nextInt(odds.size)))._2
          }.toList
        }.map { case (k, v) =>
          k -> (v.map(_.slice(0, dimension)).reduce(_ + _) ++ v.map(_.slice(dimension, dimension + 1)).reduce(_ + _).map(_ / dimension))
        }
      }
    }

    val net = Network(
        Input(dimension + 1) ::
        Cluster(Hidden(3, Linear)) ::
        Output(dimension + 1, Sigmoid) :: HNil,
        Settings(iterations = 20)
      )

    net.train(xsys.map(_._1), xsys.map(_._2))

    val res = classes.map(c => Normalizer.UnitVector(net.evaluate(c._2)) -> c._1)

    val outputFile = ~>(new File(clusterOutput)).io(_.delete)
    ~>(new PrintWriter(new FileOutputStream(outputFile, true))).io { writer =>
      res.foreach(v => writer.println(prettyPrint(v._1) + " " + v._2))
    }.io(_.close)

  }

}

/*

    [run-main-0] INFO neuroflow.nets.NFLBFGS - Val and Grad Norm: 922,047 (rel: 0,00312) 16,5574
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Step Size: 1,000
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Val and Grad Norm: 919,803 (rel: 0,00243) 10,6843
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Converged because max iterations reached
    [success] Total time: 75 s, completed 19.04.2017 22:39:45


    Result Plot: resources/ParityCloud.png
    With higher dimension and less samples: resources/ParityCloudAlt.png
    With one additional 'greasy' feature that separates the observations: resources/ParityCloudAlt2.png + ParityCloudAlt3.png

 */
