val circeVersion = "0.8.0"

libraryDependencies  ++= Seq(
  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly(),
  "org.scalanlp" %% "breeze" % "0.13",
  "joda-time" % "joda-time" % "2.8.2",
  "io.circe" %% "circe-core" % circeVersion,
  "io.circe" %% "circe-generic" % circeVersion,
  "io.circe" %% "circe-parser" % circeVersion,
  "org.slf4j" % "slf4j-simple" % "1.7.5",
  "org.specs2" %% "specs2-core" % "3.8.9" % "test",
  "org.scalatest" %% "scalatest" % "3.0.1" % "test"
)

resolvers ++= Seq(
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)
