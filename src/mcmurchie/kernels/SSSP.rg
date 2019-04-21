import "regent"

local function SSSP(a, b, c, R000, lambda,
                    r_j_values, j_offset,
                    r_density, d_offset)
  return rquote
    var R1000 : double = a * R000[1]
    var R0100 : double = b * R000[1]
    var R0010 : double = c * R000[1]

    var P : double
    var result : double

    P = r_density[d_offset].value
    result = R000[0] * P
    P = r_density[d_offset + 1].value
    result -= R1000 * P
    P = r_density[d_offset + 2].value
    result -= R0100 * P
    P = r_density[d_offset + 3].value
    result -= R0010 * P

    result *= lambda

    r_j_values[j_offset].value += result
  end
end

return SSSP
