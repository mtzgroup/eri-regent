import "regent"
require("generate_integral")

__demand(__inline)
task SSSP(R000   : double[2],
          P      : double[4],
          a      : double,
          b      : double,
          c      : double) : double[1]
  var R1000 : double = a * R000[1]
  var R0100 : double = b * R000[1]
  var R0010 : double = c * R000[1]

  var result : double[1]
  result[0] = R000[0] * P[0]
  result[0] -= R1000 * P[1]
  result[0] -= R0100 * P[2]
  result[0] -= R0010 * P[3]

  return result
end

coulombSSSP = generateTaskCoulombIntegral(0, 1, SSSP)
