package neuroflow.playground

import java.io.{File, FileOutputStream, PrintWriter}

import neuroflow.application.plugin.IO
import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Util._
import neuroflow.application.processor.{Extensions, Normalizer, Util}
import Extensions.VectorOps
import neuroflow.common.~>
import neuroflow.core.Activator._
import neuroflow.core._
import shapeless._

import scala.io.{Source, StdIn}

/**
  * @author bogdanski
  * @since 15.04.17
  */

object MovieRecommender {

  case class Movie(id: Int, title: String)
  case class Rating(user: Int, movieId: Int, rating: Int)

  val file = "/Users/felix/github/unversioned/movierec.nf"

  val movies: List[Movie] =
    ~>(Source.fromFile(getResourceFile("file/ml-100k/u.item")).getLines.toList).map { ms =>
      ms.map { line =>
        val r = line.replace("|", ";").split(";")
        Movie(r(0).toInt, r(1))
      }
    }

  val ratings: List[Rating] = Source.fromFile(getResourceFile("file/ml-100k/u.data"))
    .getLines.map(_.split("\t")).map(r => Rating(r(0).toInt, r(1).toInt, r(2).toInt)).toList

  val tts = ratings.count(_.rating == 5)
  val fls = ratings.count(_.rating == 1)

  println(s"tts: $tts, fls: $fls")

  val xs = ratings.groupBy(_.user).map {
    case (userId, rs) => rs.map { r =>
      ζ(movies.size).updated(r.movieId - 1, r.rating.toDouble * 0.2)
    }.reduce(_ + _)
  }.toList

  val layout = Input(movies.size) :: Hidden(4, Sigmoid) :: Output(movies.size, Sigmoid) :: HNil

  import neuroflow.nets.DefaultNetwork._

  def apply = {

    import neuroflow.core.FFN.WeightProvider._

    val net = Network(layout,
      Settings(iterations = 25,
        learningRate = {
          case iter if iter <= 10              => 10.0
          case iter if iter  > 10 && iter < 20 =>  5.0
          case iter                            =>  0.2
        },
        approximation = Some(Approximation(1E-5))))

    net.train(xs, xs)

    IO.File.write(net, file)

  }

  def eval = {

    implicit val wp = IO.File.read(file)
    val net = Network(layout)

    xs.foreach { x =>
      val z = x.zipWithIndex.flatMap {
        case (i, k) if i == 0 => Some(k)
        case _ => None
      }

      val o = x.zipWithIndex.flatMap {
        case (i, k) if i > 0.6 => Some(movies.find(_.id == k + 1).get.title -> i)
        case _ => None
      }

      val t = net.evaluate(x).map(i => i / 0.2)
      val all = z.map { i => i -> t(i) }.sortBy(_._2).reverse.map {
        case (i, rat) => movies.find(_.id == i + 1).get.title -> rat
      }

      val (top, flop) = (all.take(5), all.takeRight(5))

      println(s"rated: $o")
      println(s"top: $top")
      println(s"flop: $flop")
      println()
    }

  }

}
