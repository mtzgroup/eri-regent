import "regent"
require("generate_integral")

__demand(__inline)
task SSSP(R000   : double[2],
          P      : double[4],
          a      : double,
          b      : double,
          c      : double) : double[1]
  var result : double[1]

  result[0] = [generateRExpression(0, 0, 0, a, b, c, R000)] * P[0]
  result[0] -= [generateRExpression(1, 0, 0, a, b, c, R000)] * P[1]
  result[0] -= [generateRExpression(0, 1, 0, a, b, c, R000)] * P[1]
  result[0] -= [generateRExpression(0, 0, 1, a, b, c, R000)] * P[3]

  return result
end

coulombSSSP = generateTaskCoulombIntegral(0, 1, SSSP)
