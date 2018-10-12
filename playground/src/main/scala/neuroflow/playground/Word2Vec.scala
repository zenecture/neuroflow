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

      This model illustrates the principle of Word2Vec skip-gram,
      i.e. using a linear bottleneck projection layer for clustering words.

      It produces a word vector dictionary, sliding a window over the text corpus: << windowL | target word | windowR >>
      The resulting vectors have dimension `wordDim`. Words without a minimum word count are `cutOff`.

   */

  def apply = {

    val wordDim = 20

    val windowL = 10
    val windowR = 10
    val cutOff  = 6

    val output = "/Users/felix/github/unversioned/word2vec.txt"
    val wps = "/Users/felix/github/unversioned/word2vecWp.nf"

    implicit val weights = WeightBreeder[Float].normal(Map(1 -> (0.0, 0.2), 2 -> (0.0, 0.01)))
//    implicit val weights = IO.File.weightBreeder[Float](wps)

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
          batchSize       = Some(1825),
          gcThreshold     = Some(256L * 1024L * 1024L),
          waypoint        = Some(Waypoint(nth = 100, (_, ws) => IO.File.writeWeights(ws, wps))))
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

    println("Search for words... type '!q' to exit.")

    var find: String = ""
    while ({ print("Find word: "); find = StdIn.readLine(); find != "!q" }) {
      val search = res.find(_._1 == find)
      if (search.isEmpty) println(s"Couldn't find '$find'")
      search.foreach { found =>
        val sims = res.map { r =>
          r._1 -> cosineSimilarity(found._2, r._2)
        }.sortBy(_._2).reverse.take(10)
        println(s"10 close words for ${found._1}:")
        sims.foreach(println)
      }
    }

  }

}

/*

    Find word: this
    10 close words for this:
    (this,1.0000000000000002)
    (that,0.8019692350827597)
    (then,0.7708570153186066)
    (in,0.7629352475531863)
    (but,0.7576960299465183)
    (might,0.74985702457262)
    (is,0.7424459047375567)
    (be,0.7340690849691263)
    (case,0.7307796938811121)
    (if,0.721725324047652)

    Find word: infection
    10 close words for infection:
    (infection,0.9999999999999999)
    (hiv,0.843220308944862)
    (studies,0.8215011064456148)
    (mother,0.7876219542008291)
    (infected,0.7824488149704468)
    (addition,0.7757412190382657)
    (biological,0.7757090456677228)
    (properties,0.7734318008356436)
    (virus,0.7669225673339366)
    (oral,0.7648847268495547)

    Find word: honda
    10 close words for honda:
    (honda,1.0)
    (owners,0.8856899891138512)
    (accord,0.8742929494371982)
    (clutch,0.8378747101812247)
    (amount,0.7719844800542742)
    (shifting,0.7336407449236684)
    (concerns,0.6942505302932392)
    (speed,0.6938484692655489)
    (warm,0.686383342403993)
    (chatter,0.66346535477902)

 */

