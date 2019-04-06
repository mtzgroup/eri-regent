import "regent"
require("generate_integral")

__demand(__inline)
task PPSS(R000   : double[3],
          P      : double[1],
          a      : double,
          b      : double,
          c      : double) : double[10]
  var R1000 : double = a * R000[1]
  var R0100 : double = b * R000[1]
  var R0010 : double = c * R000[1]

  var R1001 : double = a * R000[2]
  var R0101 : double = b * R000[2]
  var R0011 : double = c * R000[2]

  var R1100 : double = a * R0101
  var R1010 : double = a * R0011
  var R0110 : double = b * R0011

  var R2000 : double = a * R1001 + R000[1]
  var R0200 : double = b * R0101 + R000[1]
  var R0020 : double = c * R0011 + R000[1]

  var result : double[10]

  result[0] = R000[0] * P[0]
  result[1] = R1000 * P[0]
  result[2] = R0100 * P[0]
  result[3] = R0010 * P[0]
  result[4] = R1100 * P[0]
  result[5] = R1010 * P[0]
  result[6] = R0110 * P[0]
  result[7] = R2000 * P[0]
  result[8] = R0200 * P[0]
  result[9] = R0020 * P[0]

  return result
end

coulombPPSS = generateTaskCoulombIntegral(2, 0, PPSS)
