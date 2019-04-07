import "regent"
require("generate_integral")

__demand(__inline)
task SPPP(R000   : double[4],
          P      : double[10],
          a      : double,
          b      : double,
          c      : double) : double[4]
  var result : double[4]

  result[0] = [generateRExpression(0, 0, 0, a, b, c, R000)] * P[0]
  result[1] = [generateRExpression(1, 0, 0, a, b, c, R000)] * P[0]
  result[2] = [generateRExpression(0, 1, 0, a, b, c, R000)] * P[0]
  result[3] = [generateRExpression(0, 0, 1, a, b, c, R000)] * P[0]

  result[0] -= [generateRExpression(1, 0, 0, a, b, c, R000)] * P[1]
  result[1] -= [generateRExpression(2, 0, 0, a, b, c, R000)] * P[1]
  result[2] -= [generateRExpression(1, 1, 0, a, b, c, R000)] * P[1]
  result[3] -= [generateRExpression(1, 0, 1, a, b, c, R000)] * P[1]

  result[0] -= [generateRExpression(0, 1, 0, a, b, c, R000)] * P[2]
  result[1] -= [generateRExpression(1, 1, 0, a, b, c, R000)] * P[2]
  result[2] -= [generateRExpression(0, 2, 0, a, b, c, R000)] * P[2]
  result[3] -= [generateRExpression(0, 1, 1, a, b, c, R000)] * P[2]

  result[0] -= [generateRExpression(0, 0, 1, a, b, c, R000)] * P[3]
  result[1] -= [generateRExpression(1, 0, 1, a, b, c, R000)] * P[3]
  result[2] -= [generateRExpression(0, 1, 1, a, b, c, R000)] * P[3]
  result[3] -= [generateRExpression(0, 0, 2, a, b, c, R000)] * P[3]

  result[0] += [generateRExpression(1, 1, 0, a, b, c, R000)] * P[4]
  result[1] += [generateRExpression(2, 1, 0, a, b, c, R000)] * P[4]
  result[2] += [generateRExpression(1, 2, 0, a, b, c, R000)] * P[4]
  result[3] += [generateRExpression(1, 1, 1, a, b, c, R000)] * P[4]

  result[0] += [generateRExpression(1, 0, 1, a, b, c, R000)] * P[5]
  result[1] += [generateRExpression(2, 0, 1, a, b, c, R000)] * P[5]
  result[2] += [generateRExpression(1, 1, 1, a, b, c, R000)] * P[5]
  result[3] += [generateRExpression(1, 0, 2, a, b, c, R000)] * P[5]

  result[0] += [generateRExpression(0, 1, 1, a, b, c, R000)] * P[6]
  result[1] += [generateRExpression(1, 1, 1, a, b, c, R000)] * P[6]
  result[2] += [generateRExpression(0, 2, 1, a, b, c, R000)] * P[6]
  result[3] += [generateRExpression(0, 1, 2, a, b, c, R000)] * P[6]

  result[0] += [generateRExpression(2, 0, 0, a, b, c, R000)] * P[7]
  result[1] += [generateRExpression(3, 0, 0, a, b, c, R000)] * P[7]
  result[2] += [generateRExpression(2, 1, 0, a, b, c, R000)] * P[7]
  result[3] += [generateRExpression(2, 0, 1, a, b, c, R000)] * P[7]

  result[0] += [generateRExpression(0, 2, 0, a, b, c, R000)] * P[8]
  result[1] += [generateRExpression(1, 2, 0, a, b, c, R000)] * P[8]
  result[2] += [generateRExpression(0, 3, 0, a, b, c, R000)] * P[8]
  result[3] += [generateRExpression(0, 2, 1, a, b, c, R000)] * P[8]

  result[0] += [generateRExpression(0, 0, 2, a, b, c, R000)] * P[9]
  result[1] += [generateRExpression(1, 0, 2, a, b, c, R000)] * P[9]
  result[2] += [generateRExpression(0, 1, 2, a, b, c, R000)] * P[9]
  result[3] += [generateRExpression(0, 0, 3, a, b, c, R000)] * P[9]

  return result
end

coulombSPPP = generateTaskCoulombIntegral(1, 2, SPPP)
