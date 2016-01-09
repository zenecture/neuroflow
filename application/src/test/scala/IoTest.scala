import neuroflow.application.io.IO
import neuroflow.core.Network
import org.specs2.Specification
import org.specs2.specification.core.SpecStructure

/**
  * @author bogdanski
  * @since 09.01.16
  */
class IoTest extends Specification {
  def is: SpecStructure = s2"""

    This spec will test IO related functionality.

    It should:
      - Serialize a net                       $serialize
      - De-serialize a net                    $deserialize

  """

  def serialize = {
    val net = Network(Nil)
    val serialized = IO.save(net)
    serialized.size must be greaterThan 0
  }

  def deserialize = {
    val net = Network(Nil)
    val serialized = IO.save(net)
    val deserialized = IO.load(serialized).getOrElse(throw new Exception("Couldn't deserialize the net."))
    net.layers.size must be equalTo deserialized.layers.size
  }

}