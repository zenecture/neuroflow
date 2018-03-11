package neuroflow.playground

import java.io.{File, FileOutputStream, PrintWriter}

import neuroflow.application.plugin.Extensions._
import neuroflow.application.plugin.IO.Jvm._
import neuroflow.application.plugin.Notation.ζ
import neuroflow.application.processor.Util._
import neuroflow.common.~>
import neuroflow.core.Activators.Double._
import neuroflow.core._
import neuroflow.dsl._
import neuroflow.dsl.Generic._
import neuroflow.nets.cpu.DenseNetwork._

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

    def toVector(p: Pokemon): scala.Vector[Double] = p match {
      case Pokemon(_, t1, t2, tot, hp, att, defe, spAtk, spDef, speed, gen, leg) =>
        ζ[Double](types.size).data.toVector.updated(t1, 1.0) ++ /* ζ(types.size).updated(t2, 1.0) ++ */ scala.Vector(tot / maximums._1) ++
          scala.Vector(hp / maximums._2) ++ scala.Vector(att / maximums._3) ++ scala.Vector(defe / maximums._4)
          /* ++ ->(spAtk / maximums._5) ++ ->(spDef / maximums._6) ++ ->(speed / maximums._7)
             ++ ζ(gens.size).updated(gen, 1.0) ++ ->(leg) */
    }

    val  xs = pokemons.map(p => p -> toVector(p).denseVec)
    val dim = xs.head._2.size

    implicit val breeder = neuroflow.core.WeightBreeder[Double].random(-1, 1)

    val L =
      Vector(dim)                 ::
      Dense(3, Linear)            ::
      Dense(dim / 2, ReLU)        ::
      Dense(dim, ReLU)            ::  SquaredMeanError()

    val net =
      Network(
        layout = L,
        Settings[Double](iterations = 5000, prettyPrint = true, learningRate = { case (_, _) => 1E-5 })
      )

    val xz = xs.map(_._2)

    net.train(xz, xz)

    val cluster = xs.map(t => net.focus(L.tail.head).apply(t._2) -> t._1)

    val outputFile = ~>(new File(clusterOutput)).io(_.delete)
    ~>(new PrintWriter(new FileOutputStream(outputFile, true))).io { writer =>
      cluster.foreach(v => writer.println(prettyPrint(v._1.toScalaVector, ";") + ";" + s"${v._2.name} (${v._2.type1}, " +
        s"${v._2.total}, ${v._2.hp}, ${v._2.attack}, ${v._2.defense})"))
    }.io(_.close)

  }

}

/*

    For result see:
      resources/PokeCluster.png

 */
