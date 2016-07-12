package neuroflow.application.processor

import java.io.{BufferedInputStream, File, FileInputStream}

import neuroflow.common.~>
import neuroflow.core.Network

/**
  * @author bogdanski
  * @since 12.06.16
  */

object Util {

  /**
    * Gets the `File` residing at `path`.
    */
  def getResourceFile(path: String): File = new File(getClass.getClassLoader.getResource(path).toURI)

  /**
    * Gets all files within `path`.
    */
  def getResourceFiles(path: String): Seq[File] = new File(getClass.getClassLoader.getResource(path).toURI).listFiles.filter(_.isFile)

  /**
    * Gets the plain bytes from `file`.
    */
  def getBytes(file: File): Seq[Byte] = ~> (new BufferedInputStream(new FileInputStream(file))) map
    (s => (s, Stream.continually(s.read).takeWhile(_ != -1).map(_.toByte).toList)) io (_._1.close) map(_._2)

  /**
    * Parses a word2vec skip-gram `file` to give a map of word -> vector.
    * Fore more information about word2vec: https://code.google.com/archive/p/word2vec/
    * Use `dimension` to enforce that all vectors have the same dimension.
    */
  def word2vec(file: File, dimension: Option[Int] = None): Map[String, Vector[Double]] =
    scala.io.Source.fromFile(file).getLines.map { l =>
      val raw = l.split(" ")
      (raw.head, raw.tail.map(_.toDouble).toVector)
    }.toMap.filter(l => dimension.forall(l._2.size == _))

  /**
    * Strips given string `s` to only contain lower case letters.
    */
  def strip(s: String) = s.replaceAll("[^a-zA-Z ]+", "").toLowerCase

  /**
    * Builds the cartesian of two traversables.
    */
  implicit class Crossable[X](xs: Traversable[X]) {
    def cross[Y](ys: Traversable[Y]) = for (x <- xs; y <- ys) yield (x, y)
  }

}
