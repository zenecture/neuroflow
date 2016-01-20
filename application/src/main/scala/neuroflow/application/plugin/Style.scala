package neuroflow.application.plugin

/**
  * @author bogdanski
  * @since 20.01.16
  */
object Style {

  /**
    * Just a little candy.
    */
  def ->[A](elems: A*): Vector[A] = elems.toVector
  def -->[A](elems: A*) = elems

}
