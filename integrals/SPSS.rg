import "regent"
require("generate_integral")

__demand(__inline)
task SPSS(R000   : double[2],
          P      : double[1],
          a      : double,
          b      : double,
          c      : double) : double[4]
  var result : double[4]

  result[0] = [generateRExpression(0, 0, 0, a, b, c, R000)] * P[0]
  result[1] = [generateRExpression(1, 0, 0, a, b, c, R000)] * P[0]
  result[2] = [generateRExpression(0, 1, 0, a, b, c, R000)] * P[0]
  result[3] = [generateRExpression(0, 0, 1, a, b, c, R000)] * P[0]

  return result
end

coulombSPSS = generateTaskCoulombIntegral(1, 0, SPSS)
