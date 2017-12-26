package neuroflow.common

import scala.language.implicitConversions

/**
  * @author bogdanski
  * @since 09.01.16
  */

case class ~>[T](t: T) {

  def next[B](f: => B): ~>[B] = ~>(f)
  def io(f: T => Unit): ~>[T] = ~>(f(t)) flatMap (_ => this)
  def map[B](f: T => B): ~>[B] = flatMap(g => ~>(f(g)))
  def flatMap[B](f: T => ~>[B]): ~>[B] = f(t)

}

object ~> {

  implicit def lift[T](v: T): ~>[T] = ~>(v)
  implicit def ground[T](monadic: ~>[T]): T = monadic.t

}
