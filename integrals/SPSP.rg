import "regent"
require("generate_integral")

__demand(__inline)
task SPSP(R000   : double[3],
          P      : double[4],
          a      : double,
          b      : double,
          c      : double) : double[4]
  var result : double[4]

  result[0] = [generateRExpression(0, 0, 0, 0, a, b, c, R000)] * P[0]
  result[1] = [generateRExpression(1, 0, 0, 0, a, b, c, R000)] * P[0]
  result[2] = [generateRExpression(0, 1, 0, 0, a, b, c, R000)] * P[0]
  result[3] = [generateRExpression(0, 0, 1, 0, a, b, c, R000)] * P[0]

  result[0] -= [generateRExpression(1, 0, 0, 0, a, b, c, R000)] * P[1]
  result[1] -= [generateRExpression(2, 0, 0, 0, a, b, c, R000)] * P[1]
  result[2] -= [generateRExpression(1, 1, 0, 0, a, b, c, R000)] * P[1]
  result[3] -= [generateRExpression(1, 0, 1, 0, a, b, c, R000)] * P[1]

  result[0] -= [generateRExpression(0, 1, 0, 0, a, b, c, R000)] * P[2]
  result[1] -= [generateRExpression(1, 1, 0, 0, a, b, c, R000)] * P[2]
  result[2] -= [generateRExpression(0, 2, 0, 0, a, b, c, R000)] * P[2]
  result[3] -= [generateRExpression(0, 1, 1, 0, a, b, c, R000)] * P[2]

  result[0] -= [generateRExpression(0, 0, 1, 0, a, b, c, R000)] * P[3]
  result[1] -= [generateRExpression(1, 0, 1, 0, a, b, c, R000)] * P[3]
  result[2] -= [generateRExpression(0, 1, 1, 0, a, b, c, R000)] * P[3]
  result[3] -= [generateRExpression(0, 0, 2, 0, a, b, c, R000)] * P[3]

  return result
end

coulombSPSP = generateTaskCoulombIntegral(1, 1, SPSP)
