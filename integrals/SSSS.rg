import "regent"
require("generate_integral")

__demand(__inline)
task SSSS(R000   : double[1],
          P      : double[1],
          a      : double,
          b      : double,
          c      : double) : double[1]
  var result : double[1]
  result[0] = R000[0] * P[0]
  return result
end

coulombSSSS = generateTaskCoulombIntegral(0, 0, SSSS)
