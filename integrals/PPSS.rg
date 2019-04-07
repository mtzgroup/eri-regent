import "regent"
require("generate_integral")

__demand(__inline)
task PPSS(R000   : double[3],
          P      : double[1],
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

  return result
end

coulombPPSS = generateTaskCoulombIntegral(2, 0, PPSS)
