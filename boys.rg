import "regent"

local assert = regentlib.assert
local cmath = terralib.includec("math.h")
local M_PI = cmath.M_PI
local sqrt = cmath.sqrt
local floor = cmath.floor
local exp = cmath.exp

-- Use a python script to generate `precomputedBoys.h`
-- `python gen_precomputed_header.py`
-- local getPrecomputedBoys = terralib.includec("precomputedBoys.h").getPrecomputedBoys
-- FIXME: Hack to allow docker to see header file
local getPrecomputedBoys = terralib.includec("precomputedBoys.h", {"-I", "/eri"}).getPrecomputedBoys

-- Computes a list of size `length` of auxiliary values.
-- R000[j] = (-2*alpha)^j * F_j(t)
-- where F_j(t) is the boys function.
terra computeR000(t : double, alpha : double, R000 : &double, length : int)
  assert(t >= 0, "t must be non-negative!")

  if t < 12 then
    assert(length-1 <= 16, "Only accurate for j <= 16")
    var t_est : double = floor(10.0 * t + 0.5) / 10.0
    R000[length-1] = 0
    var factor : double = 1
    for k = 0, 7 do
      var boys_est : double = getPrecomputedBoys(t_est, length-1+k)
      R000[length-1] = R000[length-1] + factor * boys_est
      factor = factor * (t_est - t) / (k + 1)
    end
    for j = length-2, -1, -1 do
      R000[j] = (2.0 * t * R000[j+1] + exp(-t)) / (2.0 * j + 1)
    end
  else
    assert(length-1 <= 16, "Only accurate for j <= 16")
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

    for j = 1, length do
      R000[j] = 2.0 / t * ((2 * j + 1) * R000[j-1] - exp(-t))
    end
  end

  var factor : double = 1
  for j = 0, length do
    R000[j] = R000[j] * factor
    factor = factor * -2 * alpha
  end
end
