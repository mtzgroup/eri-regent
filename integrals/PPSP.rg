import "regent"
require("generate_integral")

__demand(__inline)
task PPSP(R000   : double[4],
          P      : double[4],
          a      : double,
          b      : double,
          c      : double) : double[10]
  var result : double[10]

  result[0] = [generateRExpression(0, 0, 0, 0, a, b, c, R000)] * P[0]
  result[1] = [generateRExpression(1, 0, 0, 0, a, b, c, R000)] * P[0]
  result[2] = [generateRExpression(0, 1, 0, 0, a, b, c, R000)] * P[0]
  result[3] = [generateRExpression(0, 0, 1, 0, a, b, c, R000)] * P[0]
  result[4] = [generateRExpression(1, 1, 0, 0, a, b, c, R000)] * P[0]
  result[5] = [generateRExpression(1, 0, 1, 0, a, b, c, R000)] * P[0]
  result[6] = [generateRExpression(0, 1, 1, 0, a, b, c, R000)] * P[0]
  result[7] = [generateRExpression(2, 0, 0, 0, a, b, c, R000)] * P[0]
  result[8] = [generateRExpression(0, 2, 0, 0, a, b, c, R000)] * P[0]
  result[9] = [generateRExpression(0, 0, 2, 0, a, b, c, R000)] * P[0]

  result[0] -= [generateRExpression(1, 0, 0, 0, a, b, c, R000)] * P[1]
  result[1] -= [generateRExpression(2, 0, 0, 0, a, b, c, R000)] * P[1]
  result[2] -= [generateRExpression(1, 1, 0, 0, a, b, c, R000)] * P[1]
  result[3] -= [generateRExpression(1, 0, 1, 0, a, b, c, R000)] * P[1]
  result[4] -= [generateRExpression(2, 1, 0, 0, a, b, c, R000)] * P[1]
  result[5] -= [generateRExpression(2, 0, 1, 0, a, b, c, R000)] * P[1]
  result[6] -= [generateRExpression(1, 1, 1, 0, a, b, c, R000)] * P[1]
  result[7] -= [generateRExpression(3, 0, 0, 0, a, b, c, R000)] * P[1]
  result[8] -= [generateRExpression(1, 2, 0, 0, a, b, c, R000)] * P[1]
  result[8] -= [generateRExpression(1, 0, 2, 0, a, b, c, R000)] * P[1]

  result[0] -= [generateRExpression(0, 1, 0, 0, a, b, c, R000)] * P[2]
  result[1] -= [generateRExpression(1, 1, 0, 0, a, b, c, R000)] * P[2]
  result[2] -= [generateRExpression(0, 2, 0, 0, a, b, c, R000)] * P[2]
  result[3] -= [generateRExpression(0, 1, 1, 0, a, b, c, R000)] * P[2]
  result[4] -= [generateRExpression(1, 2, 0, 0, a, b, c, R000)] * P[2]
  result[5] -= [generateRExpression(1, 1, 1, 0, a, b, c, R000)] * P[2]
  result[6] -= [generateRExpression(0, 2, 1, 0, a, b, c, R000)] * P[2]
  result[7] -= [generateRExpression(2, 1, 0, 0, a, b, c, R000)] * P[2]
  result[8] -= [generateRExpression(0, 3, 0, 0, a, b, c, R000)] * P[2]
  result[9] -= [generateRExpression(0, 1, 2, 0, a, b, c, R000)] * P[2]

  result[0] -= [generateRExpression(0, 0, 1, 0, a, b, c, R000)] * P[3]
  result[1] -= [generateRExpression(1, 0, 1, 0, a, b, c, R000)] * P[3]
  result[2] -= [generateRExpression(0, 1, 1, 0, a, b, c, R000)] * P[3]
  result[3] -= [generateRExpression(0, 0, 2, 0, a, b, c, R000)] * P[3]
  result[4] -= [generateRExpression(1, 1, 1, 0, a, b, c, R000)] * P[3]
  result[5] -= [generateRExpression(1, 0, 2, 0, a, b, c, R000)] * P[3]
  result[6] -= [generateRExpression(0, 1, 2, 0, a, b, c, R000)] * P[3]
  result[7] -= [generateRExpression(2, 0, 1, 0, a, b, c, R000)] * P[3]
  result[8] -= [generateRExpression(0, 2, 1, 0, a, b, c, R000)] * P[3]
  result[9] -= [generateRExpression(0, 0, 3, 0, a, b, c, R000)] * P[3]

  return result
end

coulombPPSP = generateTaskCoulombIntegral(2, 1, PPSP)
