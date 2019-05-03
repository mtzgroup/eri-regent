import "regent"

local function SPSS(a, b, c, R000, lambda,
                    r_j_values, j_offset,
                    r_density, d_offset)
  return rquote
    var R1000 : double = a * [R000[1]]
    var R0100 : double = b * [R000[1]]
    var R0010 : double = c * [R000[1]]

    var P0 : double = r_density[d_offset].value

    r_j_values[j_offset].value += lambda * [R000[0]] * P0
    r_j_values[j_offset + 1].value += lambda * R1000 * P0
    r_j_values[j_offset + 2].value += lambda * R0100 * P0
    r_j_values[j_offset + 3].value += lambda * R0010 * P0
  end
end

return SPSS
