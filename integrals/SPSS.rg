import "regent"
require("generate_integral")

__demand(__inline)
task SPSS(R000   : double[2],
          P      : double[1],
          a      : double,
          b      : double,
          c      : double) : double[4]
  var R1000 : double = a * R000[1]
  var R0100 : double = b * R000[1]
  var R0010 : double = c * R000[1]

  var result : double[4]

  result[0] = R000[0] * P[0]
  result[1] = R1000 * P[0]
  result[2] = R0100 * P[0]
  result[3] = R0010 * P[0]

  return result
end

coulombSPSS = generateTaskCoulombIntegral(1, 0, SPSS)
