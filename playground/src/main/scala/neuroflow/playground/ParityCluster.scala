package neuroflow.playground

import java.io.{File, FileOutputStream, PrintWriter}

import neuroflow.application.plugin.Extensions._
import neuroflow.application.plugin.Notation.ζ
import neuroflow.application.processor.Util._
import neuroflow.application.processor.Util
import neuroflow.common.~>
import neuroflow.core.Activator._
import neuroflow.core._
import neuroflow.nets.cpu.DenseNetwork._
import shapeless._

import scala.util.Random

/**
  * @author bogdanski
  * @since 17.04.17
  */

object ParityCluster {

  val dimension = 250
  val samples = 1000
  val minObs = 5
  val maxObs = 20

  val clusterOutput = "/Users/felix/github/unversioned/parityCluster.txt"

  def apply = {

    implicit val wp = neuroflow.core.WeightProvider.FFN[Double].random(-1, 1)

    val classes = (0 until dimension) map (i => (i, Random.nextInt(2))) map {
      case (i, k) if i % 2 == 0 && k % 2 == 0 => ("xw", ζ[Double](dimension).toScalaVector.updated(i, 1.0) ++ Vector(1.0))
      case (i, k) if i % 2 == 0 && k % 2 != 0 => ("xo", ζ[Double](dimension).toScalaVector.updated(i, 1.0) ++ Vector(0.0))
      case (i, k) if i % 2 != 0 && k % 2 == 0 => ("vw", ζ[Double](dimension).toScalaVector.updated(i, 1.0) ++ Vector(1.0))
      case (i, k) if i % 2 != 0 && k % 2 != 0 => ("vo", ζ[Double](dimension).toScalaVector.updated(i, 1.0) ++ Vector(0.0))
    }

    val evens = Range(0, dimension, 2).toVector
    val odds = Range(1, dimension, 2).toVector

    val xsys = {
      (0 until samples) flatMap { _ =>
        Util.shuffle {
          (minObs until (minObs + Random.nextInt(maxObs))).map { _ =>
            classes(evens(Random.nextInt(evens.size)))._2
          }.toList
        }.map { case (k, v) =>
          k -> (v.map(_.slice(0, dimension)).reduce(_ + _) ++ v.map(_.slice(dimension, dimension + 1)).reduce(_ + _).map(_ / dimension))
        }
      }
    } ++ {
      (0 until samples) flatMap { _ =>
        Util.shuffle {
          (minObs until (minObs + Random.nextInt(maxObs))).map { _ =>
            classes(odds(Random.nextInt(odds.size)))._2
          }.toList
        }.map { case (k, v) =>
          k -> (v.map(_.slice(0, dimension)).reduce(_ + _) ++ v.map(_.slice(dimension, dimension + 1)).reduce(_ + _).map(_ / dimension))
        }
      }
    }

    val net = Network(
        Input(dimension + 1)            ::
        Focus(Dense(3, Linear))         ::
        Output(dimension + 1, Sigmoid)  :: HNil,
        Settings[Double](iterations = 20, learningRate = { case (_, _) => 1E-4 })
      )

    net.train(xsys.map(_._1.dv), xsys.map(_._2.dv))

    val res = classes.map(c => net(c._2.dv) -> c._1)

    val outputFile = ~>(new File(clusterOutput)).io(_.delete)
    ~>(new PrintWriter(new FileOutputStream(outputFile, true))).io { writer =>
      res.foreach(v => writer.println(prettyPrint(v._1.toScalaVector) + " " + v._2))
    }.io(_.close)

  }

}

/*

    Result Plot: resources/ParityCloud.png





                 _   __                      ________
                / | / /__  __  ___________  / ____/ /___ _      __
               /  |/ / _ \/ / / / ___/ __ \/ /_  / / __ \ | /| / /
              / /|  /  __/ /_/ / /  / /_/ / __/ / / /_/ / |/ |/ /
             /_/ |_/\___/\__,_/_/   \____/_/   /_/\____/|__/|__/


             Version 0.805

             Identifier: N1
             Network: neuroflow.nets.DefaultNetwork
             Layout: [251 In, 3 Cluster(Dense(x)), 251 Out (σ)]
             Number of Weights: 1506




    [main] INFO neuroflow.nets.DefaultNetwork - [06.08.2017 23:25:50:360] Training with 19304 samples ...
    Aug 06, 2017 11:25:50 PM com.github.fommil.jni.JniLoader liberalLoad
    INFORMATION: successfully loaded /var/folders/t_/plj660gn6ps0546vj6xtx92m0000gn/T/jniloader8483919133436466263netlib-native_system-osx-x86_64.jnilib
    [main] INFO neuroflow.nets.DefaultNetwork - [06.08.2017 23:25:51:075] Iteration 1 - Mean Error 2656,61 - Error Vector 3279.2229947232217  3061.3903604897746  2283.3098487748557  ... (251 total)
    [main] INFO neuroflow.nets.DefaultNetwork - [06.08.2017 23:25:51:858] Iteration 2 - Mean Error 2800,51 - Error Vector 2130.65388165615  1775.2468656480069  2514.1046603739205  ... (251 total)
    [main] INFO neuroflow.nets.DefaultNetwork - [06.08.2017 23:25:52:158] Iteration 3 - Mean Error 3000,49 - Error Vector 4987.0586594634005  4531.784129830394  1847.4267806266853  ... (251 total)
    ...

 */
