package neuroflow.playground

import java.io.{File, FileOutputStream, PrintWriter}

import neuroflow.application.plugin.Extensions._
import neuroflow.application.plugin.IO
import neuroflow.application.plugin.IO.Jvm._
import neuroflow.application.plugin.IO._
import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Util._
import neuroflow.application.processor.{Normalizer, Util}
import neuroflow.common.~>
import neuroflow.core.Activators.Double._
import neuroflow.core._
import neuroflow.dsl._
import neuroflow.dsl.Generic._
import neuroflow.nets.cpu.DenseNetwork._

import scala.io.{Source, StdIn}

/**
  * @author bogdanski
  * @since 15.04.17
  */

object MovieCluster {

  case class Movie(id: Int, title: String, vec: scala.Vector[Double])
  case class Rating(user: Int, movieId: Int, rating: Int)

  val netFile = "/Users/felix/github/unversioned/movies.nf"
  val clusterOutput = "/Users/felix/github/unversioned/clusters.txt"

  val dimensionLimit = 300
  val observationLimit = 50000

  val movies: List[Movie] =
    ~>(Source.fromFile(getResourceFile("file/ml-100k/u.item")).getLines.toList.take(dimensionLimit)).map { ms =>
      ms.map { line =>
        val r = line.replace("|", ";").split(";")
        Movie(r(0).toInt, r(1), ζ[Double](ms.size).toScalaVector.updated(r(0).toInt - 1, 1.0))
      }
    }

  val observations: List[Rating] = Source.fromFile(getResourceFile("file/ml-100k/u.data"))
    .getLines.map(_.split("\t")).map(r => Rating(r(0).toInt, r(1).toInt, r(2).toInt)).toList

  val layout = Vector(movies.size) :: Dense(3, Linear) :: Dense(movies.size, Sigmoid) :: SquaredMeanError()

  def apply = {

    import neuroflow.application.plugin.Extensions.VectorOps
    implicit val breeder = neuroflow.core.WeightBreeder[Double].random(-1, 1)

    val topByUser = observations.take(observationLimit).filter(_.rating == 5).groupBy(_.user).map {
      case (user, ratings) =>
        val vecs = ratings.flatMap(r => if (r.movieId <= dimensionLimit) Some(movies(r.movieId - 1).vec) else None)
        Util.shuffle(vecs).map {
          case (k, v) => k.denseVec -> Normalizer.MaxUnit(v.reduce(_ + _).denseVec)
        }
    }.toList.flatten

    println("Training samples: " + topByUser.size)

    val net = Network(layout, Settings[Double](iterations = 500, learningRate = { case (_, _) => 1E-4 }))

    net.train(topByUser.map(_._1), topByUser.map(_._2))

    IO.File.writeWeights(net.weights, netFile)

  }

  def find = {

    val net = {
      implicit val breeder = IO.File.weightBreeder[Double](netFile)
      Network(layout, Settings[Double]())
    }

    val res = movies.map(m => m.copy(vec = net.focus(layout.tail.head).apply(m.vec.denseVec).scalaVec))

    val outputFile = ~>(new File(clusterOutput)).io(_.delete)
    ~>(new PrintWriter(new FileOutputStream(outputFile, true))).io { writer =>
      res.foreach(v => writer.println(prettyPrint(v.vec, ";") + ";" + v.title))
    }.io(_.close)

    var findId: Int = 0
    while ({ print("Find movieId: "); findId = StdIn.readInt(); findId >= 0 }) {
      val target = res(findId)
      val all = res.map {
        case Movie(_, title, vec) =>
          (title, cosineSimilarity(target.vec.denseVec, vec.denseVec))
      }.sortBy(_._2)
      val best = all.reverse.take(10)
      val worst = all.take(10)
      println("The 10 most (un-)similar movies for: " + target.title)
      best.foreach(m => println(m))
      println("...")
      worst.foreach(m => println(m))
      println()
      println()
    }

  }

}


/*

    See:
        - resources/MovieCloud.png

    Find movieId: 100
    The 10 most (un-)similar movies for: Heavy Metal (1981)
    (Heavy Metal (1981),1.0)
    (Priest (1994),0.9996337330358226)
    (Get Shorty (1995),0.9993119619452804)
    (Up Close and Personal (1996),0.9980979346218072)
    (Billy Madison (1995),0.997593692885892)
    (Beavis and Butt-head Do America (1996),0.9969605830805026)
    (Crimson Tide (1995),0.9966695525631081)
    (Ulee's Gold (1997),0.996190468249715)
    (Shall We Dance? (1996),0.9955180716800702)
    (Lost World: Jurassic Park, The (1997),0.9944897656524266)
    ...
    (Free Willy 2: The Adventure Home (1995),-0.9592297809777249)
    (Kansas City (1996),-0.8039040484169)
    (Operation Dumbo Drop (1995),-0.7549209605650805)
    (French Twist (Gazon maudit) (1995),-0.7332665724074914)
    (Muppet Treasure Island (1996),-0.6747395427916062)
    (Theodore Rex (1995),-0.5375612877734781)
    (Unhook the Stars (1996),-0.47010580617422415)
    (Santa Clause, The (1994),-0.3543725709772031)
    (Mimic (1997),-0.09361883933955371)
    (Mad Love (1995),0.1907817999640347)



 */
