import "regent"
require("generate_integral")

__demand(__inline)
task SSPP(R000   : double[3],
          P      : double[10],
          a      : double,
          b      : double,
          c      : double) : double[1]
  var result : double[1]

  result[0] = [generateRExpression(0, 0, 0, 0, a, b, c, R000)] * P[0]
  result[0] -= [generateRExpression(1, 0, 0, 0, a, b, c, R000)] * P[1]
  result[0] -= [generateRExpression(0, 1, 0, 0, a, b, c, R000)] * P[2]
  result[0] -= [generateRExpression(0, 0, 1, 0, a, b, c, R000)] * P[3]
  result[0] += [generateRExpression(1, 1, 0, 0, a, b, c, R000)] * P[4]
  result[0] += [generateRExpression(1, 0, 1, 0, a, b, c, R000)] * P[5]
  result[0] += [generateRExpression(0, 1, 1, 0, a, b, c, R000)] * P[6]
  result[0] += [generateRExpression(2, 0, 0, 0, a, b, c, R000)] * P[7]
  result[0] += [generateRExpression(0, 2, 0, 0, a, b, c, R000)] * P[8]
  result[0] += [generateRExpression(0, 0, 2, 0, a, b, c, R000)] * P[9]

  return result
end

coulombSSPP = generateTaskCoulombIntegral(0, 2, SSPP)
