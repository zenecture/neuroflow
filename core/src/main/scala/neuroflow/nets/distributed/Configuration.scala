package neuroflow.nets.distributed

import neuroflow.core.{Node, Settings}
import com.typesafe.config.{Config, ConfigFactory}

/**
  * @author bogdanski
  * @since 11.09.17
  */
object Configuration {

  def apply[V](node: Node, settings: Settings[V]): Config = {
    ConfigFactory.parseString(
      s"""
         |akka {
         |  log-dead-letters = 0
         |  extensions = ["com.romix.akka.serialization.kryo.KryoSerializationExtension$$"]
         |  actor {
         |    provider = remote
         |    kryo {
         |      type = "nograph"
         |      idstrategy = "incremental"
         |      implicit-registration-logging = true
         |    }
         |    serializers {
         |      kryo = "com.twitter.chill.akka.AkkaSerializer"
         |    }
         |    serialization-bindings {
         |       "neuroflow.nets.distributed.Message" = kryo
         |    }
         |  }
         |  remote {
         |    artery {
         |      enabled = on
         |      canonical.hostname = "${node.host}"
         |      canonical.port = ${node.port}
         |      advanced {
         |        maximum-frame-size = ${settings.transport.frameSize}
         |        maximum-large-frame-size = ${settings.transport.frameSize}
         |      }
         |    }
         |  }
         |}
    """.stripMargin)
  }

}
