import "regent"
require("fields")
require("generate_spin_pattern")

local sqrt = regentlib.sqrt(double)

function generateSetPatternStatements(L, r_spin_pattern)
  local H = (L + 1) * (L + 2) * (L + 3) / 6
  local statements = terralib.newlist()
  local pattern_lua = generateSpinPattern(L)
  for i = 0, H-1 do -- inclusive
    statements:insert(rquote
      r_spin_pattern[{i, 0}] = [pattern_lua[i+1][1]]
      r_spin_pattern[{i, 1}] = [pattern_lua[i+1][2]]
      r_spin_pattern[{i, 2}] = [pattern_lua[i+1][3]]
    end)
  end
  return statements
end

function generateTaskRysIntegral(L12, L34)
  local nrys = math.floor((L12 + L34) / 2) + 1
  local L = L12 + L34
  local H12 = (L12 + 1) * (L12 + 2) * (L12 + 3) / 6
  local H34 = (L34 + 1) * (L34 + 2) * (L34 + 3) / 6
  local PI_5_2 = math.pow(math.pi, 2.5)

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
  task integral(r_bra_gausses  : region(ispace(int1d), HermiteGaussian),
                r_ket_gausses  : region(ispace(int1d), HermiteGaussian),
                r_density      : region(ispace(int1d), Double),
                r_j_values     : region(ispace(int1d), Double),
                r_spin_pattern : region(ispace(int2d), int),
                alpha          : double,
                omega          : double)
  where
    reads(r_bra_gausses, r_ket_gausses, r_density, r_spin_pattern),
    reduces +(r_j_values),
    r_density * r_j_values
  do
    var ket_idx_bounds_lo : int = r_ket_gausses.ispace.bounds.lo
    var ket_idx_bounds_hi : int = r_ket_gausses.ispace.bounds.hi
    for bra_idx in r_bra_gausses.ispace do
      for ket_idx = ket_idx_bounds_lo, ket_idx_bounds_hi + 1 do
        var bra = r_bra_gausses[bra_idx]
        var ket = r_ket_gausses[ket_idx]
        -- TODO: Bounds
        var a : double = bra.x - ket.x
        var b : double = bra.y - ket.y
        var c : double = bra.z - ket.z

        var lambda : double = alpha * 2.0 * PI_5_2 / (bra.eta * ket.eta * sqrt(bra.eta + ket.eta))
        var rho : double = bra.eta * ket.eta / (bra.eta + ket.eta)
        var d2 : double = 1
        if omega ~= -1 then
          d2 = omega * omega / (rho + omega * omega)
          lambda *= sqrt(d2)
        end
        var T : double = rho * d2 * (a*a + b*b + c*c)

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
          var j_offset = bra.data_rect.lo
          var d_offset = ket.data_rect.lo
          var P : double[H34]
          for i = 0, H34 do -- exclusive
            P[i] = r_density[ket.data_rect.lo + i].value
          end
          for t_index = 0, H12 do -- exclusive
            for u_index = 0, H34 do -- exclusive
              var N = r_spin_pattern[{t_index, 0}] + r_spin_pattern[{u_index, 0}]
              var L = r_spin_pattern[{t_index, 1}] + r_spin_pattern[{u_index, 1}]
              var M = r_spin_pattern[{t_index, 2}] + r_spin_pattern[{u_index, 2}]
              -- TODO: Maybe accumulate into array before storing into `r_j_values`
              r_j_values[bra.data_rect.lo + t_index].value += Hx[N] * Hy[L] * Hz[M] * P[u_index]
            end
          end
        end
      end
    end
  end
  local LToStr = {"SS", "SP", "PP", "PD", "DD", "FD", "FF"}
  integral:set_name("Rys"..LToStr[L12+1]..LToStr[L34+1])
  return integral
end

-- mytask = generateTaskRysIntegral(6, 6)
--
-- task top()
--   var r_spin_pattern = region(ispace(int2d, {10, 3}), int)
--   ;[generateSetPatternStatements(2, r_spin_pattern)];
-- end
