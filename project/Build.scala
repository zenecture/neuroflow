import sbt.Keys._
import sbt._
import sbtassembly.AssemblyPlugin.autoImport._

object NeuroflowBuild extends Build {

  val neuroflowSettings = Defaults.coreDefaultSettings ++ Seq(

    name in ThisBuild         := "neuroflow",
    organization in ThisBuild := "com.zenecture",
    version                   := "1.00.8",
    scalaVersion              := "2.12.3",
    assemblyMergeStrategy
                  in assembly := {
      case PathList("META-INF", xs @ _*) => MergeStrategy.discard
      case "reference.conf"              => MergeStrategy.concat
      case x                             => MergeStrategy.first
    }

  )

  lazy val core        = Project(id = "neuroflow-core",        base = file("core"),        settings = neuroflowSettings)
  lazy val application = Project(id = "neuroflow-application", base = file("application"), settings = neuroflowSettings) dependsOn core
  lazy val playground  = Project(id = "neuroflow-playground",  base = file("playground"),  settings = neuroflowSettings) dependsOn application

}
