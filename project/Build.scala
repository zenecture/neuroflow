import sbt.Keys.{autoCompilerPlugins, _}
import sbt._
import sbtassembly.AssemblyPlugin.autoImport._

object NeuroflowBuild extends Build {

  val neuroflowSettings = Defaults.coreDefaultSettings ++ Seq(

    name in ThisBuild         := "neuroflow",
    organization in ThisBuild := "com.zenecture",
    version                   := "1.7.7",
    scalaVersion              := "2.12.3",
    assemblyMergeStrategy
                  in assembly := {
      case PathList("META-INF", xs @ _*) => MergeStrategy.discard
      case "reference.conf"              => MergeStrategy.concat
      case _                             => MergeStrategy.first
    },
    autoCompilerPlugins := true,
    addCompilerPlugin("org.scalamacros" %% "paradise" % "2.1.0" cross CrossVersion.full),
    parallelExecution in ThisBuild := false

  )

  lazy val core        = Project(id = "neuroflow-core",        base = file("core"),        settings = neuroflowSettings)
  lazy val application = Project(id = "neuroflow-application", base = file("application"), settings = neuroflowSettings) dependsOn core
  lazy val playground  = Project(id = "neuroflow-playground",  base = file("playground"),  settings = neuroflowSettings) dependsOn application

}
