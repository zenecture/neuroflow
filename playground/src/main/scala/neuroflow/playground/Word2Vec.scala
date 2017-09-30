package neuroflow.playground

import java.io.{File, FileOutputStream, PrintWriter}

import neuroflow.application.plugin.IO
import neuroflow.application.plugin.Notation.ζ
import neuroflow.application.processor.Extensions.Breeze.cosineSimilarity
import neuroflow.application.processor.Normalizer.ScaledVectorSpace
import neuroflow.application.processor.Util._
import neuroflow.common.~>
import neuroflow.core.Activator._
import neuroflow.core._
import neuroflow.nets.cpu.DenseNetwork._
import shapeless._

import scala.io.{Source, StdIn}
import scala.util.{Failure, Success, Try}

/**
  * @author bogdanski
  * @since 24.09.17
  */
object Word2Vec {

  def apply = {

    val windowL = 5
    val windowR = 5
    val cutOff  = 5

    val output = "/Users/felix/github/unversioned/word2vec.txt"
    val wps = "/Users/felix/github/unversioned/word2vecWp.nf"

    implicit val wp = neuroflow.core.WeightProvider.Double.FFN(-1, 1)
//     implicit val wp = IO.File.readDouble(wps)

    val corpus = Source.fromFile(getResourceFile("file/newsgroup/reduced.txt")).mkString.split(" ").map(_ -> 1)

    val wc = collection.mutable.HashMap.empty[String, Int]
    corpus.foreach { c =>
      wc.get(c._1) match {
        case Some(i) => wc += c._1 -> (i + 1)
        case None    => wc += c._1 -> 1
      }
    }

    val words = wc.filter(_._2 >= cutOff)

    val vecs  = words.zipWithIndex.map {
      case (w, i) => w._1 -> {
        val v = ζ[Double](words.size)
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

    val net =
      Network(
        Input(dim)                 ::
        Focus(Dense(20, Linear))   ::
        Output(dim, Sigmoid)       :: HNil,
        Settings[Double](
          learningRate    = { case (_, _) => 1E-4 },
          updateRule      = Momentum(0.9),
          iterations      = 10000,
          batchSize       = Some(64),
          regularization  = Some(KeepBest),
          prettyPrint     = true,
          waypoint        = Some(Waypoint(nth = 10, ws => IO.File.write(ws, wps))))
      )

    net.train(xsys.map(_._2), xsys.map(_._3))

    val resRaw = vecs.map {
      case (w, v) => w -> net(v)
    }.toSeq

    val res = resRaw.map(_._1).zip(ScaledVectorSpace(resRaw.map(_._2)))

    val outputFile = ~>(new File(output)).io(_.delete)
    ~>(new PrintWriter(new FileOutputStream(outputFile, true))).io { writer =>
      res.foreach { v =>
        writer.println(prettyPrint(v._2.toScalaVector, ";") + s";${v._1}")
      }
    }.io(_.close)

    var findId: Int = 0
    while ({ print("Find wordId: "); findId = StdIn.readInt(); findId >= 0 }) {
      val found = res(findId)
      val sims  = res.map { r =>
        r._1 -> cosineSimilarity(found._2, r._2)
      }.sortBy(_._2).reverse.take(10)
      println(s"10 similar words for ${found._1}:")
      sims.foreach(println)
    }

  }

}
