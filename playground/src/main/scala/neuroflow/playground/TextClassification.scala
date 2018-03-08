package neuroflow.playground

import neuroflow.application.plugin.Extensions._
import neuroflow.application.plugin.IO.Jvm._
import neuroflow.application.plugin.IO._
import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Util._
import neuroflow.core.Activator._
import neuroflow.core._
import neuroflow.dsl._
import neuroflow.nets.cpu.DenseNetwork._

/**
  * @author bogdanski
  * @since 20.06.16
  */

object TextClassification {


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
  def word2vec(file: java.io.File, dimension: Option[Int] = None): Map[String, scala.Vector[Double]] =
    scala.io.Source.fromFile(file).getLines.map { l =>
      val raw = l.split(" ")
      (raw.head, raw.tail.map(_.toDouble).toVector)
    }.toMap.filter(l => dimension.forall(l._2.size == _))

  def apply = {

    implicit val breeder = neuroflow.core.WeightBreeder[Double].random(-1, 1)

    val cars = normalize(readAll("file/newsgroup/cars/"))
    val med = normalize(readAll("file/newsgroup/med/"))

    val trainCars = vectorize(cars).map((_, ->(1.0, 0.0)))
    val trainMed = vectorize(med).map((_, ->(0.0, 1.0)))

    val allTrain = trainCars ++ trainMed

    println("No. of samples: " + allTrain.size)

    val net = Network(Vector(20) :: Dense(40, Tanh) :: Dense(40, Tanh) :: Dense(2, Sigmoid) :: Softmax(),
      Settings[Double](iterations = 15000, learningRate = { case _ => 1E-4 }))

    net.train(allTrain.map(_._1.denseVec), allTrain.map(_._2))

    File.writeWeights(net.weights, netFile)

    neuroflow.application.plugin.IO.File.writeWeights(net.weights, netFile)

  }

  def test = {

    val net = {
      implicit val breeder = File.readWeights[Double](netFile)
      Network(Vector(20) :: Dense(40, Tanh) :: Dense(40, Tanh) :: Dense(2, Sigmoid) :: Softmax())
    }

    val cars = normalize(readAll("file/newsgroup/cars/", offset = maxSamples, max = maxSamples))
    val med = normalize(readAll("file/newsgroup/med/", offset = maxSamples, max = maxSamples))
    val free = normalize(readSingle("file/newsgroup/free.txt"))

    val testCars = vectorize(cars)
    val testMed = vectorize(med)
    val testFree = vectorize(free)

    def eval(id: String, maxIndex: Int, xs: scala.Vector[scala.Vector[Double]]) = {
      val (ok, fail) = xs.map(x => net(x.denseVec)).map(k => k.toScalaVector.indexOf(k.max) == maxIndex).partition(l => l)
      println(s"Correctly classified $id: ${ok.size.toDouble / (ok.size.toDouble + fail.size.toDouble) * 100.0} % !")
    }

    eval("cars", 0, testCars)
    eval("med", 1, testMed)

    testFree.map(x => net(x.denseVec)).foreach(k =>
      println(s"Free classified as: ${if (k.toScalaVector.indexOf(k.max) == 0) "cars" else "med"}")
    )

  }

}

/*




                 _   __                      ________
                / | / /__  __  ___________  / ____/ /___ _      __
               /  |/ / _ \/ / / / ___/ __ \/ /_  / / __ \ | /| / /
              / /|  /  __/ /_/ / /  / /_/ / __/ / / /_/ / |/ |/ /
             /_/ |_/\___/\__,_/_/   \____/_/   /_/\____/|__/|__/
                                                                1.5.7


                Network : neuroflow.nets.cpu.DenseNetwork

                Weights : 2.480 (≈ 0,0189209 MB)
              Precision : Double

                   Loss : neuroflow.core.Softmax
                 Update : neuroflow.core.Vanilla

                 Layout : 20 Vector
                          40 Dense (φ)
                          40 Dense (φ)
                          2 Dense (σ)






                   O     O
                   O     O
             O     O     O
             O     O     O
             O     O     O     O
             O     O     O     O
             O     O     O
             O     O     O
                   O     O
                   O     O



    Mär 08, 2018 8:55:20 PM com.github.fommil.jni.JniLoader liberalLoad
    INFORMATION: successfully loaded /var/folders/t_/plj660gn6ps0546vj6xtx92m0000gn/T/jniloader2727974297877248970netlib-native_system-osx-x86_64.jnilib
    Correctly classified cars: 98.98989898989899 % !
    Correctly classified med: 97.0 % !
    Free classified as: cars



 */