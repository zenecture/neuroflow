import sbt.Keys._
import sbt._
import sbtassembly.AssemblyPlugin.autoImport._
import sbtassembly.PathList

object ZeduceBuild extends Build {

  val neuroflowSettings = Defaults.coreDefaultSettings ++ Seq(
    name in ThisBuild := "neuroflow",
    organization in ThisBuild := "com.zenecture",
    version := "0.1",
    scalaVersion := "2.11.7",
    assemblyMergeStrategy in assembly := {
      case x => MergeStrategy defaultMergeStrategy x
    }
  )

  lazy val core = Project(id = "neuroflow-core", base = file("core"), settings = neuroflowSettings)
  lazy val application = Project(id = "neuroflow-application", base = file("application"), settings = neuroflowSettings) dependsOn core
}
