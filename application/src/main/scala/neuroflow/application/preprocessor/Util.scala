package neuroflow.application.preprocessor

import java.io.{BufferedInputStream, File, FileInputStream}

import neuroflow.common.~>

/**
  * @author bogdanski
  * @since 12.06.16
  */

object Util {

  /**
    * Gets the `File` specified by `path`
    */
  def getFile(path: String): File = new File(getClass.getClassLoader.getResource(path).toURI)

  /**
    * Gets the plain bytes from `file`.
    */
  def getBytes(file: File): Seq[Byte] = ~> (new BufferedInputStream(new FileInputStream(file))) map
    (s => (s, Stream.continually(s.read).takeWhile(_ != -1).map(_.toByte).toList)) io (_._1.close) map(_._2)

}
