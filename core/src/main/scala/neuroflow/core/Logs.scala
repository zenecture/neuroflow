package neuroflow.core

import org.joda.time.DateTime

import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

/**
  * @author bogdanski
  * @since 29.09.15
  */

/**
  * Base trait for any loggers. May use monadic structure
  * in `Return` to chain things.
  */
trait Loggable[Return] {

  def warn(message: String): Return
  def error(message: String): Return
  def info(message: String): Return

}

/**
  * Instead of depending on these heavy-metal logging frameworks,
  * the idea is to just print to console, while redirecting this output
  * to a file on OS-level.
  */
trait Logs extends Loggable[Unit] {

  private val datePattern = "dd.MM.yyyy HH:mm:ss:SSS"
  private def format(s: String) = s"[${DateTime.now.toString(datePattern)}] [${Thread.currentThread.getName}] $s"
  private def print(message: String) = Future(println(message))

  def warn(message: String): Unit = print("[WARN] " + format(message))
  def error(message: String): Unit = print("[ERROR] " + format(message))
  def info(message: String): Unit = print("[INFO] " + format(message))

}
