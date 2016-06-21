package neuroflow.application.processor

import java.io.{BufferedInputStream, File, FileInputStream}

import neuroflow.common.~>

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
    */
  def word2vec(file: File): Map[String, Vector[Double]] =
    scala.io.Source.fromFile(file).getLines.map { l =>
      val raw = l.split(" ")
      (raw.head, raw.tail.map(_.toDouble).toVector)
    }.toMap

}
