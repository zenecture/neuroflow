package neuroflow.playground

import java.io.{File, FileOutputStream, PrintWriter}

import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Util._
import neuroflow.application.processor.{Extensions, Normalizer, Util}
import neuroflow.common.~>
import neuroflow.core.Activator._
import neuroflow.core._
import neuroflow.nets.LBFGSCluster._
import shapeless._

import scala.util.Random

/**
  * @author bogdanski
  * @since 17.04.17
  */

object ParityCluster {

  val dimension = 20
  val samples = 1000
  val maxObservations = 10

  val clusterOutput = "/Users/felix/github/unversioned/parityCluster.txt"

  def apply = {

    import Extensions.SeqVectorOps
    import neuroflow.core.FFN.WeightProvider._

    val classes = (0 until dimension) map (i => (if (i % 2 == 0) "x" else "o", ζ(dimension).updated(i, 1.0)))

    val evens = Range(0, dimension, 2).toList
    val odds = Range(1, dimension, 2).toList

    val xsys = {
      (0 until samples) flatMap { _ =>
        Util.shuffle {
          val ex = (0 until Random.nextInt(maxObservations)).map { _ =>
            classes(evens(Random.nextInt(evens.size)))._2
          }.toList
          ex
        }.map { case (k, v) => k -> Normalizer.MaxUnit(v.reduce(_ + _)) }
      }
    } ++ {
      (0 until samples) flatMap { _ =>
        Util.shuffle {
          (0 until Random.nextInt(maxObservations)).map { _ =>
            classes(odds(Random.nextInt(odds.size)))._2
          }.toList
        }.map { case (k, v) => k -> Normalizer.MaxUnit(v.reduce(_ + _)) }
      }
    }

    val net = Network(Input(dimension) :: Cluster(3, Linear) :: Output(dimension, Sigmoid) :: HNil, Settings(iterations = 50))

    net.train(xsys.map(_._1), xsys.map(_._2))

    val res = classes.map(c => Normalizer.UnitVector(net.evaluate(c._2)) -> c._1)

    val outputFile = ~>(new File(clusterOutput)).io(_.delete)
    ~>(new PrintWriter(new FileOutputStream(outputFile, true))).io { writer =>
      res.foreach(v => writer.println(prettyPrint(v._1) + " " + v._2))
    }.io(_.close)

  }

}

/*

    Result Plot: resources/ParityCloud.png
    Wth higher dimension and less samples: resources/ParityCloudAlt.png
 */
