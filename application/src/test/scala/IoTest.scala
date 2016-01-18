import neuroflow.application.io.IO
import neuroflow.core.Activator.Sigmoid
import neuroflow.core.{Hidden, Output, Input, Network}
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
      - Do file IO                            $fileIo

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

  def fileIo = {
    val file = "/Users/felix/github/unversioned/savednets/net.nf"
    val net = Network(Input(2) :: Hidden(3, Sigmoid.apply) :: Output(2, Sigmoid.apply) :: Nil)
    IO.save(net, file)
    val success = IO.load(file) exists (_.layers === net.layers)
    success must beTrue
  }

}