val circeVersion = "0.9.0"

libraryDependencies  ++= Seq(
  "io.circe"               %%  "circe-core"                     %  circeVersion,
  "io.circe"               %%  "circe-generic"                  %  circeVersion,
  "io.circe"               %%  "circe-parser"                   %  circeVersion,
  "org.specs2"             %%  "specs2-core" % "3.8.9" % "test",
  "org.scalatest"          %%  "scalatest" % "3.0.1" % "test"
)

resolvers ++= Seq(
  Resolver.sonatypeRepo("snapshots"),
  Resolver.sonatypeRepo("releases")
)
