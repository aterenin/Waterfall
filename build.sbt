name := "Waterfall"

version := "1.0"

scalaVersion := "2.11.8"

assemblyJarName in assembly := "Waterfall.jar"
mainClass in assembly := Some("GPUGibbs")
test in assembly := {}
assemblyExcludedJars in assembly := {
  val cp = (fullClasspath in assembly).value
  cp.filter{_.data.getName.contains("jcu")}
}