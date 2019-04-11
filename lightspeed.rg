import "regent"
require("fields")

local sqrt = regentlib.sqrt(double)

function generateLightspeedIntegral(L12, L34)
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
    return 1.0
  end

  local function generateSumStatements(a, b, c, U, lambda,
                                       r_j_values, j_offset,
                                       r_density, d_offset)
    local Hx = regentlib.newsymbol(double[L + 1], "Hx")
    local Hy = regentlib.newsymbol(double[L + 1], "Hy")
    local Hz = regentlib.newsymbol(double[L + 1], "Hz")
    local statements = terralib.newlist({rquote
      var [Hx]
      var [Hy]
      var [Hz]
      Hx[0] = lambda
      Hy[0] = 1.0
      Hz[0] = 1.0
    end})
    if L > 0 then
      statements:insert(rquote
        Hx[1] = U * a * Hx[0] -- Hx[0] holds an extra constant
        Hy[1] = U * b
        Hz[1] = U * c
      end)
    end
    for i = 2, L do -- inclusive
      statements:insert(rquote
        Hx[i] = U * (a * Hx[i-1] + (i-1) * Hx[i-2])
        Hy[i] = U * (b * Hy[i-1] + (i-1) * Hy[i-2])
        Hz[i] = U * (c * Hz[i-1] + (i-1) * Hz[i-2])
      end)
    end

    local P = regentlib.newsymbol(double[H34], "P")
    statements:insert(rquote
      var [P]
    end)
    for i = 0, H34-1 do --inclusive
      statements:insert(rquote
        P[i] = r_density[d_offset + i].value
      end)
    end
    -- NOTE: This is based on the format of the input data from TeraChem
    local pattern = {
      {0; 0; 0;};
      {1; 0; 0;};
      {0; 1; 0;};
      {0; 0; 1;};
      {1; 1; 0;};
      {1; 0; 1;};
      {0; 1; 1;};
      {2; 0; 0;};
      {0; 2; 0;};
      {0; 0; 2;};
    }
    for t = 0, H12-1 do -- inclusive
      for u = 0, H34-1 do -- inclusive
        -- TODO: Need to check if this is correct
        local N = pattern[t + 1][1] + pattern[u + 1][1]
        local L = pattern[t + 1][2] + pattern[u + 1][2]
        local M = pattern[t + 1][3] + pattern[u + 1][3]
        statements:insert(rquote
          r_j_values[j_offset + t].value += Hx[N] * Hy[L] * Hz[M] * P[u]
        end)
      end
    end
    return statements
  end

  __demand(__leaf)
  __demand(__cuda)
  task lightspeed(r_bra_gausses : region(ispace(int1d), HermiteGaussian),
                  r_ket_gausses : region(ispace(int1d), HermiteGaussian),
                  r_density     : region(ispace(int1d), Double),
                  r_j_values    : region(ispace(int1d), Double),
                  alpha         : double,
                  omega         : double)
  where
    reads(r_bra_gausses, r_ket_gausses, r_density),
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
          var lambda_w : double = lambda * w
          var j_offset = bra.data_rect.lo
          var d_offset = ket.data_rect.lo
          ;[generateSumStatements(a, b, c, U, lambda_w,
                                  r_j_values, j_offset,
                                  r_density, d_offset)];
        end
      end
    end
  end
  return lightspeed
end

mytask = generateLightspeedIntegral(1, 2)
