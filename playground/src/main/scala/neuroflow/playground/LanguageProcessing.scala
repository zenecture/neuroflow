package neuroflow.playground

import java.io.File

import neuroflow.application.plugin.IO
import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Util._
import neuroflow.common.VectorTranslation._
import neuroflow.core.Activator._
import neuroflow.core._
import neuroflow.nets.cpu.LBFGSNetwork._
import shapeless._

/**
  * @author bogdanski
  * @since 20.06.16
  */

object LanguageProcessing {


  /*

       Here the goal is to classify the context of arbitrary text.
       The classes used for training are C = { cars, med }.
       The data is an aggregate of newsgroup posts found at the internet.
         (Source: qwone.com/%7Ejason/20Newsgroups)

       Feel free to read this article for the full story:
         http://znctr.com/blog/language-processing

   */


  val netFile = "/Users/felix/github/unversioned/langprocessing.nf"
  val maxSamples = 100
  val dict = word2vec(getResourceFile("file/newsgroup/all-vec.txt"))

  def readAll(dir: String, max: Int = maxSamples, offset: Int = 0) =
    getResourceFiles(dir).drop(offset).take(max).map(scala.io.Source.fromFile)
      .flatMap(bs => try { Some(strip(bs.mkString)) } catch { case _: Throwable => None })

  def readSingle(file: String) = Seq(strip(scala.io.Source.fromFile(getResourceFile(file)).mkString))

  def normalize(xs: Seq[String]): scala.Vector[scala.Vector[String]] = xs.map(_.split(" ").distinct.toVector).toVector

  def vectorize(xs: Seq[Seq[String]]): scala.Vector[scala.Vector[Double]] = xs.map(_.flatMap(dict.get)).map { v =>
    val vs = v.reduce((l, r) => l.zip(r).map(l => l._1 + l._2))
    val n = v.size.toDouble
    vs.map(_ / n)
  }.toVector

  /**
    * Parses a word2vec skip-gram `file` to give a map of word -> vector.
    * Fore more information about word2vec: https://code.google.com/archive/p/word2vec/
    * Use `dimension` to enforce that all vectors have the same dimension.
    */
  def word2vec(file: File, dimension: Option[Int] = None): Map[String, scala.Vector[Double]] =
    scala.io.Source.fromFile(file).getLines.map { l =>
      val raw = l.split(" ")
      (raw.head, raw.tail.map(_.toDouble).toVector)
    }.toMap.filter(l => dimension.forall(l._2.size == _))

  def apply = {

    implicit val wp = neuroflow.core.WeightProvider.FFN[Double].random(-1, 1)

    val cars = normalize(readAll("file/newsgroup/cars/"))
    val med = normalize(readAll("file/newsgroup/med/"))

    val trainCars = vectorize(cars).map((_, ->(1.0, -1.0)))
    val trainMed = vectorize(med).map((_, ->(-1.0, 1.0)))

    val allTrain = trainCars ++ trainMed

    println("No. of samples: " + allTrain.size)

    val net = Network(Input(20) :: Dense(40, Tanh) :: Dense(40, Tanh) :: Output(2, Tanh) :: HNil,
      Settings[Double](iterations = 500, specifics = Some(Map("m" -> 7))))

    net.train(allTrain.map(_._1.dv), allTrain.map(_._2))

    IO.File.write(net.weights, netFile)

  }

  def test = {

    val net = {
      implicit val wp = IO.File.readDouble(netFile)
      Network(Input(20) :: Dense(40, Tanh) :: Dense(40, Tanh) :: Output(2, Tanh) :: HNil, Settings[Double]())
    }

    val cars = normalize(readAll("file/newsgroup/cars/", offset = maxSamples, max = maxSamples))
    val med = normalize(readAll("file/newsgroup/med/", offset = maxSamples, max = maxSamples))
    val free = normalize(readSingle("file/newsgroup/free.txt"))

    val testCars = vectorize(cars)
    val testMed = vectorize(med)
    val testFree = vectorize(free)

    def eval(id: String, maxIndex: Int, xs: scala.Vector[scala.Vector[Double]]) = {
      val (ok, fail) = xs.map(x => net(x.dv)).map(k => k.toScalaVector.indexOf(k.max) == maxIndex).partition(l => l)
      println(s"Correctly classified $id: ${ok.size.toDouble / (ok.size.toDouble + fail.size.toDouble) * 100.0} % !")
    }

    eval("cars", 0, testCars)
    eval("med", 1, testMed)

    testFree.map(x => net(x.dv)).foreach(k =>
      println(s"Free classified as: ${if (k.toScalaVector.indexOf(k.max) == 0) "cars" else "med"}")
    )

  }

}
