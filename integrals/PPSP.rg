import "regent"
require("generate_integral")

__demand(__inline)
task PPSP(R000   : double[4],
          P      : double[4],
          a      : double,
          b      : double,
          c      : double) : double[10]
  var R1000 : double = a * R000[1]
  var R0100 : double = b * R000[1]
  var R0010 : double = c * R000[1]

  var R1001 : double = a * R000[2]
  var R0101 : double = b * R000[2]
  var R0011 : double = c * R000[2]

  var R1002 : double = a * R000[3]
  var R0102 : double = b * R000[3]
  var R0012 : double = c * R000[3]

  var R1100 : double = a * R0101
  var R1010 : double = a * R0011
  var R0110 : double = b * R0011

  var R1101 : double = a * R0102
  var R1011 : double = a * R0012
  var R0111 : double = b * R0012

  var R2000 : double = a * R1001 + R000[1]
  var R0200 : double = b * R0101 + R000[1]
  var R0020 : double = c * R0011 + R000[1]

  var R2001 : double = a * R1002 + R000[2]
  var R0201 : double = b * R0102 + R000[2]
  var R0021 : double = c * R0012 + R000[2]

  var R2100 : double = a * R1101 + R0101
  var R1200 : double = a * R0201
  var R1110 : double = a * R0111

  var R2010 : double = a * R1011 + R0011
  var R1020 : double = a * R0021

  var R0210 : double = b * R0111 + R0011
  var R0120 : double = b * R0021

  var R3000 : double = a * R2001 + R1001 + R1001
  var R0300 : double = b * R0201 + R0101 + R0101
  var R0030 : double = c * R0021 + R0011 + R0011

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

  result[0] -= R1000 * P[1]
  result[1] -= R2000 * P[1]
  result[2] -= R1100 * P[1]
  result[3] -= R1010 * P[1]
  result[4] -= R2100 * P[1]
  result[5] -= R2010 * P[1]
  result[6] -= R1110 * P[1]
  result[7] -= R3000 * P[1]
  result[8] -= R1200 * P[1]
  result[8] -= R1020 * P[1]

  result[0] -= R0100 * P[2]
  result[1] -= R1100 * P[2]
  result[2] -= R0200 * P[2]
  result[3] -= R0110 * P[2]
  result[4] -= R1200 * P[2]
  result[5] -= R1110 * P[2]
  result[6] -= R0210 * P[2]
  result[7] -= R2100 * P[2]
  result[8] -= R0300 * P[2]
  result[9] -= R0120 * P[2]

  result[0] -= R0010 * P[3]
  result[1] -= R1010 * P[3]
  result[2] -= R0110 * P[3]
  result[3] -= R0020 * P[3]
  result[4] -= R1110 * P[3]
  result[5] -= R1020 * P[3]
  result[6] -= R0120 * P[3]
  result[7] -= R2010 * P[3]
  result[8] -= R0210 * P[3]
  result[9] -= R0030 * P[3]

  return result
end

coulombPPSP = generateTaskCoulombIntegral(2, 1, PPSP)
