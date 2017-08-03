package neuroflow.core

import java.io.{File, FileOutputStream, PrintWriter}

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.util.Try

/**
  * @author bogdanski
  * @since 10.07.16
  */


trait ErrorFuncGrapher { self: Network[_] =>

  /**
    * Appends the `error` to the specified output file, if any,
    * and executes the given closure, if any. This does not block.
    */
  def maybeGraph(error: Double): Unit =
    self.settings.errorFuncOutput.foreach {
      efo =>
        Future {
          Try {
            val handleOpt = efo.file
              .map(f => new PrintWriter(new FileOutputStream(new File(f), true)))
            handleOpt.foreach(_.println(error))
            handleOpt.foreach(_.close())
            efo.closure.foreach(_ (error))
          }
        }
    }

}

case class ErrorFuncOutput(file: Option[String] = None, closure: Option[Double => Unit] = None)
