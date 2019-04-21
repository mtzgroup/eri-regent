import "regent"

local function SPSP(a, b, c, R000, lambda,
                    r_j_values, j_offset,
                    r_density, d_offset)
  return rquote
    var R1000 : double = a * R000[1]
    var R0100 : double = b * R000[1]
    var R0010 : double = c * R000[1]

    var R1001 : double = a * R000[2]
    var R0101 : double = b * R000[2]
    var R0011 : double = c * R000[2]

    var R1100 : double = a * R0101
    var R1010 : double = a * R0011
    var R0110 : double = b * R0011

    var R2000 : double = a * R1001 + R000[1]
    var R0200 : double = b * R0101 + R000[1]
    var R0020 : double = c * R0011 + R000[1]

    var P0 : double = r_density[d_offset].value
    var P1 : double = r_density[d_offset + 1].value
    var P2 : double = r_density[d_offset + 2].value
    var P3 : double = r_density[d_offset + 3].value

    var result : double[4]
    result[0] = R000[0] * P0
    result[1] = R1000 * P0
    result[2] = R0100 * P0
    result[3] = R0010 * P0

    result[0] -= R1000 * P1
    result[1] -= R2000 * P1
    result[2] -= R1100 * P1
    result[3] -= R1010 * P1

    result[0] -= R0100 * P2
    result[1] -= R1100 * P2
    result[2] -= R0200 * P2
    result[3] -= R0110 * P2

    result[0] -= R0010 * P3
    result[1] -= R1010 * P3
    result[2] -= R0110 * P3
    result[3] -= R0020 * P3

    result[0] *= lambda
    result[1] *= lambda
    result[2] *= lambda
    result[3] *= lambda

    r_j_values[j_offset].value += result[0]
    r_j_values[j_offset + 1].value += result[1]
    r_j_values[j_offset + 2].value += result[2]
    r_j_values[j_offset + 3].value += result[3]
  end
end

return SPSP
