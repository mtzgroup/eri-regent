import "regent"

local function SSSP(a, b, c, R000, lambda,
                    r_j_values, j_offset,
                    r_density, d_offset)
  return rquote
    var R1000 : double = a * R000[1]
    var R0100 : double = b * R000[1]
    var R0010 : double = c * R000[1]

    var P0 : double = r_density[d_offset].value
    var P1 : double = r_density[d_offset + 1].value
    var P2 : double = r_density[d_offset + 2].value
    var P3 : double = r_density[d_offset + 3].value

    var result : double
    result = R000[0] * P0
    result -= R1000 * P1
    result -= R0100 * P2
    result -= R0010 * P3
    result *= lambda
    r_j_values[j_offset].value += result
  end
end

return SSSP
