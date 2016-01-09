package neuroflow.common

/**
  * @author bogdanski
  * @since 09.01.16
  */

case class ~>[T](grounded: T) {
  def next[B](f: => B): ~>[B] = ~>(f)
  def io(f: T => Unit): ~>[T] = ~>(f(grounded)) flatMap (_ => this)
  def map[B](f: T => B): ~>[B] = ~>(f(grounded))
  def flatMap[B](f: T => ~>[B]): ~>[B] = f(grounded)
}

object ~> {

  implicit def lift[T](v: T): ~>[T] = ~>(v)
  implicit def ground[T](monadic: ~>[T]): T = monadic grounded

}
