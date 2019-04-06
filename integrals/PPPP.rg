import "regent"
require("generate_integral")

__demand(__inline)
task PPPP(R000   : double[5],
          P      : double[10],
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
  var R1003 : double = a * R000[4]
  var R0103 : double = b * R000[4]
  var R0013 : double = c * R000[4]

  var R1100 : double = a * R0101
  var R1010 : double = a * R0011
  var R0110 : double = b * R0011
  var R1101 : double = a * R0102
  var R1011 : double = a * R0012
  var R0111 : double = b * R0012
  var R1102 : double = a * R0103
  var R1012 : double = a * R0013
  var R0112 : double = b * R0013

  var R2000 : double = a * R1001 + R000[1]
  var R0200 : double = b * R0101 + R000[1]
  var R0020 : double = c * R0011 + R000[1]
  var R2001 : double = a * R1002 + R000[2]
  var R0201 : double = b * R0102 + R000[2]
  var R0021 : double = c * R0012 + R000[2]
  var R2002 : double = a * R1003 + R000[3]
  var R0202 : double = b * R0103 + R000[3]
  var R0022 : double = c * R0013 + R000[3]

  var R2100 : double = a * R1101 + R0101
  var R1200 : double = a * R0201
  var R1110 : double = a * R0111
  var R2010 : double = a * R1011 + R0011
  var R1020 : double = a * R0021
  var R0210 : double = b * R0111 + R0011
  var R0120 : double = b * R0021
  var R2101 : double = a * R1102 + R0102
  var R1201 : double = a * R0202
  var R1111 : double = a * R0112
  var R2011 : double = a * R1012 + R0012
  var R1021 : double = a * R0022
  var R0211 : double = b * R0112 + R0012
  var R0121 : double = b * R0022

  var R3000 : double = a * R2001 + R1001 + R1001
  var R0300 : double = b * R0201 + R0101 + R0101
  var R0030 : double = c * R0021 + R0011 + R0011
  var R3001 : double = a * R2002 + R1002 + R1002
  var R0301 : double = b * R0202 + R0102 + R0102
  var R0031 : double = c * R0022 + R0012 + R0012

  var R2200 : double = a * R1201 + R0201
  var R2110 : double = a * R1111 + R0111
  var R1210 : double = a * R0211
  var R3100 : double = a * R2101 + R1101 + R1101
  var R1300 : double = a * R0301
  var R1120 : double = a * R0121

  var R2020 : double = a * R1021 + R0021
  var R3010 : double = a * R2011 + R1011 + R1011
  var R1030 : double = a * R0031

  var R0220 : double = b * R0121 + R0021
  var R0310 : double = b * R0211 + R0111 + R0111
  var R0130 : double = b * R0031

  var R4000 : double = a * R3001 + R2001 + R2001 + R2001
  var R0400 : double = b * R0301 + R0201 + R0201 + R0201
  var R0040 : double = c * R0031 + R0021 + R0021 + R0021

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
  result[9] -= R1020 * P[1]

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

  result[0] += R1100 * P[4]
  result[1] += R2100 * P[4]
  result[2] += R1200 * P[4]
  result[3] += R1110 * P[4]
  result[4] += R2200 * P[4]
  result[5] += R2110 * P[4]
  result[6] += R1210 * P[4]
  result[7] += R3100 * P[4]
  result[8] += R1300 * P[4]
  result[9] += R1120 * P[4]

  result[0] += R1010 * P[5]
  result[1] += R2010 * P[5]
  result[2] += R1110 * P[5]
  result[3] += R1020 * P[5]
  result[4] += R2110 * P[5]
  result[5] += R2020 * P[5]
  result[6] += R1120 * P[5]
  result[7] += R3010 * P[5]
  result[8] += R1210 * P[5]
  result[9] += R1030 * P[5]

  result[0] += R0110 * P[6]
  result[1] += R1110 * P[6]
  result[2] += R0210 * P[6]
  result[3] += R0120 * P[6]
  result[4] += R1210 * P[6]
  result[5] += R1120 * P[6]
  result[6] += R0220 * P[6]
  result[7] += R2110 * P[6]
  result[8] += R0310 * P[6]
  result[9] += R0130 * P[6]

  result[0] += R2000 * P[7]
  result[1] += R3000 * P[7]
  result[2] += R2100 * P[7]
  result[3] += R2010 * P[7]
  result[4] += R3100 * P[7]
  result[5] += R3010 * P[7]
  result[6] += R2110 * P[7]
  result[7] += R4000 * P[7]
  result[8] += R2200 * P[7]
  result[9] += R2020 * P[7]

  result[0] += R0200 * P[8]
  result[1] += R1200 * P[8]
  result[2] += R0300 * P[8]
  result[3] += R0210 * P[8]
  result[4] += R1300 * P[8]
  result[5] += R1210 * P[8]
  result[6] += R0310 * P[8]
  result[7] += R2200 * P[8]
  result[8] += R0400 * P[8]
  result[9] += R0220 * P[8]

  result[0] += R0020 * P[9]
  result[1] += R1020 * P[9]
  result[2] += R0120 * P[9]
  result[3] += R0030 * P[9]
  result[4] += R1120 * P[9]
  result[5] += R1030 * P[9]
  result[6] += R0130 * P[9]
  result[7] += R2020 * P[9]
  result[8] += R0220 * P[9]
  result[9] += R0040 * P[9]

  return result
end

coulombPPPP = generateTaskCoulombIntegral(2, 2, PPPP)
