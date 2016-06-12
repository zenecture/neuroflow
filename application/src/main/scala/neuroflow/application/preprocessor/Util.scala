package neuroflow.application.preprocessor

import java.io.{BufferedInputStream, File, FileInputStream}

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
  def getBytes(file: File): Seq[Byte] = {
    val bis = new BufferedInputStream(new FileInputStream(file))
    Stream.continually(bis.read).takeWhile(_ != -1).map(_.toByte).toList
  }

}
