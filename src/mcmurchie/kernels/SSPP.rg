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
    P = r_density[d_offset + 4].value
    result += R1100 * P
    P = r_density[d_offset + 5].value
    result += R1010 * P
    P = r_density[d_offset + 6].value
    result += R0110 * P
    P = r_density[d_offset + 7].value
    result += R2000 * P
    P = r_density[d_offset + 8].value
    result += R0200 * P
    P = r_density[d_offset + 9].value
    result += R0020 * P

    result *= lambda

    r_j_values[j_offset].value += result
  end
end

return SSPP
