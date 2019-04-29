-- TODO: Rename this file
import "regent"
require "fields"

local cmath = terralib.includec("math.h")
local M_PI = cmath.M_PI
local sqrt = regentlib.sqrt(double)
local floor = regentlib.floor(double)
local exp = regentlib.exp(double)

-- Generates a task that computes a list of size `length` of auxiliary values.
-- R000[j] = (-2*alpha)^j * F_j(t)
-- where F_j(t) is the Boys function.
function generateTaskComputeR000(length)
  assert(length <= 16, "Only accurate for j <= 16")
  -- TODO: Much of this code can be metaprogrammed if performance is an issue here
  local
  __demand(__inline)
  task computeR000(t      : double,
                   alpha  : double,
                   r_boys : region(ispace(int2d), double)) : double[length]
  where
    reads(r_boys)
  do
    var R000 : double[length]
    if t < 12 then
      var t_idx : int = floor(10.0 * t + 0.5)
      var t_est : double = t_idx / 10.0
      R000[length-1] = 0.0
      var factor : double = 1.0
      for k = 0, 7 do -- exclusive
        var boys_est : double = r_boys[{t_idx, length-1+k}]
        R000[length-1] = R000[length-1] + factor * boys_est
        factor = factor * (t_est - t) / (k + 1)
      end
      for j = length-2, -1, -1 do -- exclusive
        R000[j] = (2.0 * t * R000[j+1] + exp(-t)) / (2.0 * j + 1.0)
      end
    else
      R000[0] = sqrt(M_PI) / (2.0 * sqrt(t))
      var g : double
      if t < 15 then
        g = 0.4999489092 - 0.2473631686 / t + 0.321180909 / (t * t) - 0.3811559346 / (t * t * t)
      elseif t < 18 then
        g = 0.4998436875 - 0.24249438 / t + 0.24642845 / (t * t);
      elseif t < 24 then
        g = 0.499093162 - 0.2152832 / t
      elseif t < 30 then
        g = 0.490
      end

      if t < 30 then
        R000[0] = R000[0] - exp(-t) * g / t
      end

      for j = 0, length-1 do -- exclusive
        R000[j+1] = ((2.0 * j + 1.0) * R000[j] - exp(-t)) / (2.0 * t)
      end
    end

    var factor : double = 1.0
    for j = 0, length do -- exclusive
      R000[j] = R000[j] * factor
      factor = factor * -2.0 * alpha
    end
    return R000
  end
  return computeR000
end
