import "regent"

local function SSPP(a, b, c, R000, lambda,
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
    var P4 : double = r_density[d_offset + 4].value
    var P5 : double = r_density[d_offset + 5].value
    var P6 : double = r_density[d_offset + 6].value
    var P7 : double = r_density[d_offset + 7].value
    var P8 : double = r_density[d_offset + 8].value
    var P9 : double = r_density[d_offset + 9].value

    var result : double
    result = R000[0] * P0
    result -= R1000 * P1
    result -= R0100 * P2
    result -= R0010 * P3
    result += R1100 * P4
    result += R1010 * P5
    result += R0110 * P6
    result += R2000 * P7
    result += R0200 * P8
    result += R0020 * P9
    result *= lambda
    r_j_values[j_offset].value += result
  end
end

return SSPP
