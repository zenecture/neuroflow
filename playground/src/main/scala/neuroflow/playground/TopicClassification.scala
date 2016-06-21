package neuroflow.playground

import neuroflow.application.plugin.IO.File
import neuroflow.application.plugin.Style._
import neuroflow.application.processor.Util._
import neuroflow.core.Activator.Tanh
import neuroflow.core._
import neuroflow.nets.LBFGSNetwork._
import shapeless._

/**
  * @author bogdanski
  * @since 20.06.16
  */

object TopicClassification {

  val netFile = "/Users/felix/github/unversioned/ct.nf"
  val maxSamples = 100
  val dict = word2vec(getResourceFile("file/newsgroup/all-vec.txt"))

  def readAll(dir: String, max: Int = maxSamples, offset: Int = 0) =
    getResourceFiles(dir).drop(offset).take(max).map(scala.io.Source.fromFile)
      .flatMap(bs => try { Some(strip(bs.mkString)) } catch { case _ => None })

  def readSingle(file: String) = Seq(strip(scala.io.Source.fromFile(getResourceFile(file)).mkString))

  def strip(s: String) = s.replaceAll("[^a-zA-Z ]+", "").toLowerCase

  def normalize(xs: Seq[String]): Seq[Seq[String]] = xs.map(_.split(" ").distinct.toSeq)

  def vectorize(xs: Seq[Seq[String]]) = xs.map(_.flatMap(dict.get)).map { v =>
    val vs = v.reduce((l, r) => l.zip(r).map(l => l._1 + l._2))
    val n = v.size.toDouble
    vs.map(_ / n)
  }

  def apply = {

    val cars = normalize(readAll("file/newsgroup/cars/"))
    val med = normalize(readAll("file/newsgroup/med/"))

    val trainCars = vectorize(cars).map((_, ->(1.0, -1.0)))
    val trainMed = vectorize(med).map((_, ->(-1.0, 1.0)))

    val allTrain = trainCars ++ trainMed

    println("No. of samples: " + allTrain.size)

    val net = Network(Input(20) :: Hidden(40, Tanh) :: Hidden(40, Tanh) :: Output(2, Tanh) :: HNil,
      Settings(maxIterations = 500, specifics = Some(Map("m" -> 7))))

    net.train(allTrain.map(_._1), allTrain.map(_._2))

    File.write(net, netFile)

  }

  def test = {

    val net = {
      implicit val wp = File.read(netFile)
      Network(Input(20) :: Hidden(40, Tanh) :: Hidden(40, Tanh) :: Output(2, Tanh) :: HNil, Settings())
    }

    val cars = normalize(readAll("file/newsgroup/cars/", offset = 0, max = maxSamples * 2))
    val med = normalize(readAll("file/newsgroup/med/", offset = 0, max = maxSamples * 2))
    val free = normalize(readSingle("file/newsgroup/free.txt"))

    val trainCars = vectorize(cars)
    val trainMed = vectorize(med)
    val trainFree = vectorize(free)

    def eval(id: String, maxIndex: Int, xs: Seq[Seq[Double]]) = {
      val (ok, fail) = xs.map(net.evaluate).map(k => k.indexOf(k.max) == maxIndex).partition(l => l)
      println(s"Correctly classified $id: ${ok.size.toDouble / (ok.size.toDouble + fail.size.toDouble) * 100.0} % !")
    }

    eval("cars", 0, trainCars)
    eval("med", 1, trainMed)

    trainFree.map(net.evaluate).foreach(k => println(s"Free classified as: ${if (k.indexOf(k.max) == 0) "cars" else "med"}"))

  }

}
