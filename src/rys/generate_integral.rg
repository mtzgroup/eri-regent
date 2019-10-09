import "regent"

require "fields"
require "generate_spin_pattern"
require "helper"

local sqrt = regentlib.sqrt(double)
local rsqrt = regentlib.rsqrt(double)

-- Generates the statements to produce a constant region needed for the Rys algorithm.
function generateSpinPatternRegion(L)
  local H = (L + 1) * (L + 2) * (L + 3) / 6
  local r_spin_pattern = regentlib.newsymbol("r_spin_pattern")
  -- TODO: Since the largest value is `L`, we can significantly reduce the size
  --       of `r_spin_pattern` if needed.
  local statements = terralib.newlist({rquote
    var [r_spin_pattern] = region(ispace(int1d, H), int3d)
  end})
  local pattern_lua = generateSpinPattern(L)
  for i = 0, H-1 do -- inclusive
    local N, L, M = unpack(pattern_lua[i+1])
    statements:insert(rquote
      r_spin_pattern[i] = {x=N, y=L, z=M}
    end)
  end
  return {statements, r_spin_pattern}
end

-- Given a pair of angular momentums, this returns a task
-- to compute electron repulsion integrals between BraKets
-- using the Rys algorithm.
function generateTaskRysIntegral(L12, L34)
  local nrys = math.floor((L12 + L34) / 2) + 1
  local L = L12 + L34
  local H12 = (L12 + 1) * (L12 + 2) * (L12 + 3) / 6
  local H34 = (L34 + 1) * (L34 + 2) * (L34 + 3) / 6

  local
  __demand(__inline)
  task interpolate_t(T : double, i : int) : double
    return 1.0 -- TODO
  end

  local
  __demand(__inline)
  task interpolate_w(T : double, i : int) : double
    return 1.0 -- TODO
  end

  local
  __demand(__leaf)
  __demand(__cuda)
  task integral(r_bra_gausses  : region(ispace(int1d), getHermiteGaussianPacked(L12)),
                r_ket_gausses  : region(ispace(int1d), getHermiteGaussianPacked(L34)),
                r_spin_pattern : region(ispace(int1d), int3d))
  where
    reads(r_bra_gausses.{x, y, z, eta, C, bound}),
    reads(r_ket_gausses.{x, y, z, eta, C, density, bound}),
    reads(r_spin_pattern),
    reduces +(r_bra_gausses.j)
  do
    var alpha : double = 1.0 -- TODO
    var omega : double = 1.0 -- TODO
    var ket_idx_bounds_lo : int = r_ket_gausses.ispace.bounds.lo
    var ket_idx_bounds_hi : int = r_ket_gausses.ispace.bounds.hi
    for bra_idx in r_bra_gausses.ispace do
      var bra_x : double = r_bra_gausses[bra_idx].x
      var bra_y : double = r_bra_gausses[bra_idx].y
      var bra_z : double = r_bra_gausses[bra_idx].z
      var bra_eta : double = r_bra_gausses[bra_idx].eta
      var bra_C : double = r_bra_gausses[bra_idx].C
      var bra_bound : double = r_bra_gausses[bra_idx].bound
      var accumulator : double[H12]
      for i = 0, H12 do -- exclusive
        accumulator[i] = 0.0
      end

      for ket_idx = ket_idx_bounds_lo, ket_idx_bounds_hi + 1 do
        var ket_x : double = r_ket_gausses[ket_idx].x
        var ket_y : double = r_ket_gausses[ket_idx].y
        var ket_z : double = r_ket_gausses[ket_idx].z
        var ket_eta : double = r_ket_gausses[ket_idx].eta
        var ket_C : double = r_ket_gausses[ket_idx].C
        var density : double[H34] = r_ket_gausses[ket_idx].density
        var ket_bound : double = r_ket_gausses[ket_idx].bound

        -- TODO: Bounds
        var a : double = bra_x - ket_x
        var b : double = bra_y - ket_y
        var c : double = bra_z - ket_z

        var lambda : double = alpha * bra_C * ket_C * rsqrt(bra_eta + ket_eta)
        -- TODO: Rename to alpha to be consistent with McMurchie
        var rho : double = bra_eta * ket_eta / (bra_eta + ket_eta)
        var d2 : double = 1
        if omega ~= -1 then
          d2 = omega * omega / (rho + omega * omega)
          lambda *= sqrt(d2)
        end
        var T : double = rho * d2 * (a*a + b*b + c*c)

        -- TODO: Unroll
        for i = 0, nrys do
          var t : double = __demand(__inline, interpolate_t(T, i))
          var w : double = __demand(__inline, interpolate_w(T, i))
          var U : double = -2.0 * rho * d2 * t * t
          var Hx : double[L+1]
          var Hy : double[L+1]
          var Hz : double[L+1]
          Hx[0] = lambda * w
          Hy[0] = 1.0
          Hz[0] = 1.0
          if L > 0 then
            Hx[1] = U * a * Hx[0]
            Hy[1] = U * b
            Hz[1] = U * c
          end
          for i = 1, L do -- exclusive
            Hx[i+1] = U * (a * Hx[i] + i * Hx[i-1])
            Hy[i+1] = U * (b * Hy[i] + i * Hy[i-1])
            Hz[i+1] = U * (c * Hz[i] + i * Hz[i-1])
          end
          for t_index = 0, H12 do -- exclusive
            var sum : double = 0.0
            var t_spin : int3d = r_spin_pattern[t_index]
            for u_index = 0, H34 do -- exclusive
              var u_spin : int3d = r_spin_pattern[u_index]
              var N = t_spin.x + u_spin.x
              var L = t_spin.y + u_spin.y
              var M = t_spin.z + u_spin.z
              sum += Hx[N] * Hy[L] * Hz[M] * density[u_index]
            end
            accumulator[t_index] += sum
          end
        end
      end

      r_bra_gausses[bra_idx].j += accumulator
    end
  end
  integral:set_name("Rys"..LToStr[L12]..LToStr[L34])
  return integral
end
