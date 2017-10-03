package neuroflow.core

import java.io.{File, FileOutputStream, PrintWriter}

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.util.Try

/**
  * @author bogdanski
  * @since 10.07.16
  */


trait LossFuncGrapher { self: Network[_, _, _] =>

  /**
    * Appends the `loss` to the specified output file, if any,
    * and executes given `action`, if any. This does not block.
    */
  def maybeGraph(loss: Double): Unit =
    self.settings.lossFuncOutput.foreach {
      lfo =>
        Future {
          Try {
            val handleOpt = lfo.file
              .map(f => new PrintWriter(new FileOutputStream(new File(f), true)))
            handleOpt.foreach(_.println(loss))
            handleOpt.foreach(_.close())
            lfo.action.foreach(_ (loss))
          }
        }
    }

}

case class LossFuncOutput(file: Option[String] = None, action: Option[Double => Unit] = None)
