package neuroflow.playground

import java.io.{File, FileOutputStream, PrintWriter}

import neuroflow.application.plugin.Notation.ζ
import neuroflow.application.plugin.Notation.Force._
import neuroflow.application.processor.Normalizer
import neuroflow.application.processor.Util._
import neuroflow.common.~>
import neuroflow.core.Activator._
import neuroflow.core._
import neuroflow.core.FFN.WeightProvider._
import neuroflow.nets.AutoEncoder._
import shapeless._

import scala.io.Source

/**
  *
  * There is a story behind: http://znctr.com/blog/neural-pokemon
  * Data set from: https://www.kaggle.com/abcsds/pokemon
  *
  * @author bogdanski
  * @since 21.04.17
  */

object PokeMonCluster {

  case class Pokemon(name: String, type1: Int, type2: Int, total: Double,
                     hp: Double, attack: Double, defense: Double, spAtk: Double, spDef: Double,
                     speed: Double, gen: Int, legendary: Double)

  def apply = {

    val clusterOutput = "/Users/felix/github/unversioned/pokeCluster.txt"

    val source = Source.fromFile(getResourceFile("file/pokemon.csv")).getLines.drop(1).toList
    val types = source.flatMap { line => val r = line.split(","); Seq(r(2), r(3)) }.toSet.toIndexedSeq
    val gens = source.map(_.split(",")(11).toDouble).toSet.toIndexedSeq
    val pokemons = source.map { line =>
      val r = line.split(",")
      Pokemon(r(1), types.indexOf(r(2)), types.indexOf(r(3)), r(4).toDouble, r(5).toDouble, r(6).toDouble, r(7).toDouble,
        r(8).toDouble, r(9).toDouble, r(10).toDouble, gens.indexOf(r(11).toDouble), if (r(12).toBoolean) 1 else 0)
    }
    val maximums =
      (pokemons.map(_.total).max, pokemons.map(_.hp).max,
        pokemons.map(_.attack).max, pokemons.map(_.defense).max,
        pokemons.map(_.spAtk).max, pokemons.map(_.spDef).max,
        pokemons.map(_.speed).max)

    def toVector(p: Pokemon): Vector[Double] = p match {
      case Pokemon(_, t1, t2, tot, hp, att, defe, spAtk, spDef, speed, gen, leg) =>
        ζ(types.size).updated(t1, 1.0) ++ /* ζ(types.size).updated(t2, 1.0) ++ */ ->(tot / maximums._1) ++
          ->(hp / maximums._2) ++ ->(att / maximums._3) ++ ->(defe / maximums._4)
          /* ++ ->(spAtk / maximums._5) ++ ->(spDef / maximums._6) ++ ->(speed / maximums._7)
             ++ ζ(gens.size).updated(gen, 1.0) ++ ->(leg) */
    }

    val xs = pokemons.map(p => p -> toVector(p))
    val dim = xs.head._2.size

    val net =
      Network(
        Input(dim) ::
        Cluster(Hidden(3, Linear)) ::
        Hidden(dim / 2, ReLU) ::
        Output(dim, ReLU) :: HNil,
        Settings(iterations = 200, prettyPrint = true)
      )


    net.train(xs.map(_._2))

    val cluster = xs.map(t => Normalizer.UnitVector(net.evaluate(t._2)) -> t._1)

    val outputFile = ~>(new File(clusterOutput)).io(_.delete)
    ~>(new PrintWriter(new FileOutputStream(outputFile, true))).io { writer =>
      cluster.foreach(v => writer.println(prettyPrint(v._1, ";") + ";" + s"${v._2.name} (${v._2.type1}, " +
        s"${v._2.total}, ${v._2.hp}, ${v._2.attack}, ${v._2.defense})"))
    }.io(_.close)

  }

}

/*

    For several results see:
      resources/PokeCluster.png     (Linear Cluster Layer)
      resources/PokeClusterSig.png  (Sigmoid Cluster Layer)
      resources/PokeClusterDeep.png (Linear/ReLU Layers, reduced feature space: type1, total, hp, attack, defense)


    [info] Running neuroflow.playground.App
    Run example (1-13):



                 _   __                      ________
                / | / /__  __  ___________  / ____/ /___ _      __
               /  |/ / _ \/ / / / ___/ __ \/ /_  / / __ \ | /| / /
              / /|  /  __/ /_/ / /  / /_/ / __/ / / /_/ / |/ |/ /
             /_/ |_/\___/\__,_/_/   \____/_/   /_/\____/|__/|__/


             Version 0.601

             Identifier: lHw
             Network: neuroflow.nets.AutoEncoder
             Layout: [23, 3 (x), 11 (R), 23 (R)]
             Number of Weights: 355




             O                             O
             O                             O
             O                             O
             O                             O
             O                             O
             O                             O
             O                   O         O
             O                   O         O
             O                   O         O
             O                   O         O
             O         O         O         O
             O         O         O         O
             O         O         O         O
             O                   O         O
             O                   O         O
             O                   O         O
             O                   O         O
             O                             O
             O                             O
             O                             O
             O                             O
             O                             O
             O                             O



    Apr 22, 2017 5:27:11 PM com.github.fommil.jni.JniLoader liberalLoad
    INFORMATION: successfully loaded /var/folders/t_/plj660gn6ps0546vj6xtx92m0000gn/T/jniloader8500436160996467391netlib-native_system-osx-x86_64.jnilib
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Step Size: 0,004680
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Val and Grad Norm: 54,2450 (rel: 0,627) 58,8828
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Step Size: 1,000
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Val and Grad Norm: 40,1199 (rel: 0,260) 34,3295
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Step Size: 1,000
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Val and Grad Norm: 28,9957 (rel: 0,277) 11,6635
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Step Size: 1,000
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Val and Grad Norm: 27,1558 (rel: 0,0635) 6,80454
    ...
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Step Size: 1,000
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Val and Grad Norm: 7,08702 (rel: 0,000473) 0,553453
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Step Size: 1,000
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Val and Grad Norm: 7,08400 (rel: 0,000425) 0,502604
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Converged because max iterations reached
    [success] Total time: 132 s, completed 22.04.2017 17:29:16


 */
