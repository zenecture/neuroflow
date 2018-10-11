package neuroflow.playground

import java.io.{File, FileOutputStream, PrintWriter}

import neuroflow.application.plugin.Extensions._
import neuroflow.application.plugin.IO
import neuroflow.application.plugin.IO.Jvm._
import neuroflow.application.plugin.IO._
import neuroflow.application.plugin.Notation.ζ
import neuroflow.application.processor.Normalizer.ScaledVectorSpace
import neuroflow.application.processor.Util._
import neuroflow.common.~>
import neuroflow.core.Activators.Float._
import neuroflow.core._
import neuroflow.dsl._
import neuroflow.dsl.Implicits._
import neuroflow.nets.gpu.DenseNetwork._

import scala.io.{Source, StdIn}
import scala.util.{Failure, Success, Try}

/**
  * @author bogdanski
  * @since 24.09.17
  */
object Word2Vec {

  /*

      This model is a sketch to illustrate the principle of Word2Vec skip-gram,
      i.e. using a linear bottleneck projection layer for clustering words.

      It produces a word vector dictionary, sliding a window over the text corpus: << windowL | target word | windowR >>
      The resulting vectors have dimension `wordDim`. Words without a minimum word count are `cutOff`.

   */

  def apply = {

    val wordDim = 20

    val windowL = 4
    val windowR = 4
    val cutOff  = 16

    val output = "/Users/felix/github/unversioned/word2vec.txt"
    val wps = "/Users/felix/github/unversioned/word2vecWp.nf"

    implicit val weights = WeightBreeder[Float].normal(μ = 0.01, σ = 0.01)
//     implicit val weights = IO.File.weightBreeder[Float](wps)

    val corpus = Source.fromFile(getResourceFile("file/newsgroup/reduced.txt")).mkString.split(" ").map(_ -> 1)

    val wordCount = collection.mutable.HashMap.empty[String, Int]
    corpus.foreach { c =>
      wordCount.get(c._1) match {
        case Some(i) => wordCount += c._1 -> (i + 1)
        case None    => wordCount += c._1 -> 1
      }
    }

    val words = wordCount.filter(_._2 >= cutOff)

    val vecs  = words.zipWithIndex.map {
      case (w, i) => w._1 -> {
        val v = ζ[Float](words.size)
        v.update(i, 1.0f)
        v
      }
    }.toMap

    val xsys = corpus.zipWithIndex.drop(windowL).dropRight(windowR).flatMap { case ((w, c), i) =>
      val l = ((i - windowL) until i).map(j => corpus(j)._1)
      val r = ((i + 1) until (i + 1 + windowR)).map(j => corpus(j)._1)
      Try {
        val s = l ++ r
        (w, vecs(w), s.map(vecs).reduce(_ + _))
      } match {
        case Success(res) => Some(res)
        case Failure(f)   => None // Cutoff.
      }
    }

    val dim = xsys.head._2.length

    val L =
      Vector(dim)                 ::
      Dense(wordDim, Linear)      ::
      Dense(dim, ReLU)            ::  SoftmaxLogMultEntropy[Float](N = windowL + windowR)

    val net =
      Network(
        layout = L,
        Settings[Float](
          learningRate    = { case (_, _) => 1E-4 },
          updateRule      = Momentum(0.9f),
          iterations      = 10000,
          prettyPrint     = true,
          batchSize       = Some(10070),
          gcThreshold     = Some(256L * 1024L * 1024L),
          waypoint        = Some(Waypoint(nth = 200, (_, ws) => IO.File.writeWeights(ws, wps))))
      )

    net.train(xsys.map(_._2), xsys.map(_._3))

    val focused = net focus L.tail.head

    val resRaw = vecs.map {
      case (w, v) => w -> focused(v)
    }.toSeq

    val res = resRaw.map(_._1).zip(ScaledVectorSpace(resRaw.map(_._2.double)))

    val outputFile = ~>(new File(output)).io(_.delete)
    ~>(new PrintWriter(new FileOutputStream(outputFile, true))).io { writer =>
      res.foreach { v =>
        writer.println(v._1 + " " + prettyPrint(v._2.toScalaVector, " "))
      }
    }.io(_.close)

    var find: String = ""
    while ({ print("Find word: "); find = StdIn.readLine(); find != "!q" }) {
      val found = res.find(_._1 == find).getOrElse(???)
      val sims  = res.map { r =>
        r._1 -> cosineSimilarity(found._2, r._2)
      }.sortBy(_._2).reverse.take(10)
      println(s"10 close words for ${found._1}:")
      sims.foreach(println)
    }

  }

}

/*

    Example:

    Find word: that
    10 close words for that:
    (that,0.9999999999999998)
    (this,0.9561189745530446)
    (so,0.9422961435537252)
    (said,0.9336007168283127)
    (only,0.9313980547893304)
    (if,0.9088163257358929)
    (i,0.9059398681693549)
    (do,0.9006138989645291)
    (was,0.8910568555158572)
    (what,0.8885351948991972)

 */

