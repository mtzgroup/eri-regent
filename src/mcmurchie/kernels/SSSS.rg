import "regent"

local function SSSS(a, b, c, R000, lambda,
                    r_j_values, j_offset,
                    r_density, d_offset)
  return rquote
    var P : double = r_density[d_offset].value
    r_j_values[j_offset].value += lambda * R000[0] * P
  end
end

return SSSS
