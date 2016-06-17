package neuroflow.application.preprocessor

import java.io.{BufferedInputStream, File, FileInputStream}

import scala.collection.JavaConversions._

import neuroflow.common.~>

/**
  * @author bogdanski
  * @since 12.06.16
  */

object Util {

  /**
    * Gets the `File` residing at `path`.
    */
  def getFile(path: String): File = new File(getClass.getClassLoader.getResource(path).toURI)

  /**
    * Gets all files within `path`.
    */
  def getFiles(path: String): Seq[File] = new File(getClass.getClassLoader.getResource(path).toURI).listFiles

  /**
    * Gets the plain bytes from `file`.
    */
  def getBytes(file: File): Seq[Byte] = ~> (new BufferedInputStream(new FileInputStream(file))) map
    (s => (s, Stream.continually(s.read).takeWhile(_ != -1).map(_.toByte).toList)) io (_._1.close) map(_._2)

}
