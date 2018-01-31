package neuroflow.core

/**
  * @author bogdanski
  * @since 03.01.16
  */


trait IllusionBreaker { self: Network[_, _, _] =>

  /**
    * Checks if the [[Settings]] are properly defined for this network.
    * Throws a [[neuroflow.core.IllusionBreaker.SettingsNotSupportedException]] if not. Default behavior is no op.
    */
  def checkSettings(): Unit = ()

}


object IllusionBreaker {

  class SettingsNotSupportedException(message: String) extends Exception(message)
  class NotSoundException(message: String) extends Exception(message)

}
