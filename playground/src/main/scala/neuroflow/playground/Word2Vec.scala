package neuroflow.playground

import java.io.{File, FileOutputStream, PrintWriter}

import neuroflow.application.plugin.IO
import neuroflow.application.plugin.Notation.ζ
import neuroflow.application.processor.Normalizer.ScaledVectorSpace
import neuroflow.application.processor.Util._
import neuroflow.common.~>
import neuroflow.core.Activator._
import neuroflow.core._
import neuroflow.nets.gpu.DenseNetwork._
import shapeless._

import scala.io.Source
import scala.util.{Failure, Success, Try}

/**
  * @author bogdanski
  * @since 24.09.17
  */
object Word2Vec {

  def apply = {

    val windowL = 3
    val windowR = 3
    val cutOff  = 5

    val output = "/Users/felix/github/unversioned/word2vec.txt"
    val wps = "/Users/felix/github/unversioned/word2vecWp.nf"

    implicit val wp = neuroflow.core.WeightProvider.Float.FFN(-1, 1)
//     implicit val wp = IO.File.readFloat(wps)

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

    val net =
      Network(
        Input(dim)                 ::
        Focus(Dense(3, Linear))    ::
        Output(dim, Sigmoid)       :: HNil,
        Settings[Float](
          learningRate    = { case (_, _) => 1E-4 },
          updateRule      = Momentum(0.9f),
          iterations      = 10000,
          batchSize       = Some(8),
          parallelism     = 8,
          regularization  = Some(KeepBest),
          prettyPrint     = true,
          waypoint        = Some(Waypoint(nth = 10, ws => IO.File.write(ws, wps))))
      )

    net.train(xsys.map(_._2), xsys.map(_._3))

    val resRaw = vecs.map {
      case (w, v) => w -> net(v)
    }.toSeq

    val res = resRaw.map(_._1).zip(ScaledVectorSpace(resRaw.map(_._2.map(_.toDouble))))

    val outputFile = ~>(new File(output)).io(_.delete)
    ~>(new PrintWriter(new FileOutputStream(outputFile, true))).io { writer =>
      res.foreach { v =>
        writer.println(prettyPrint(v._2.toScalaVector, ";") + s";${v._1}")
      }
    }.io(_.close)

  }

}
