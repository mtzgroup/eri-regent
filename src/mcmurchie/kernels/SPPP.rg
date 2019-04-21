import "regent"

local function SPPP(a, b, c, R000, lambda,
                    r_j_values, j_offset,
                    r_density, d_offset)
  return rquote
    var R1000 : double = a * R000[1]
    var R0100 : double = b * R000[1]
    var R0010 : double = c * R000[1]

    var R1001 : double = a * R000[2]
    var R0101 : double = b * R000[2]
    var R0011 : double = c * R000[2]

    var R1002 : double = a * R000[3]
    var R0102 : double = b * R000[3]
    var R0012 : double = c * R000[3]

    var R1100 : double = a * R0101
    var R1010 : double = a * R0011
    var R0110 : double = b * R0011

    var R1101 : double = a * R0102
    var R1011 : double = a * R0012
    var R0111 : double = b * R0012

    var R2000 : double = a * R1001 + R000[1]
    var R0200 : double = b * R0101 + R000[1]
    var R0020 : double = c * R0011 + R000[1]

    var R2001 : double = a * R1002 + R000[2]
    var R0201 : double = b * R0102 + R000[2]
    var R0021 : double = c * R0012 + R000[2]

    var R2100 : double = a * R1101 + R0101
    var R1200 : double = a * R0201
    var R1110 : double = a * R0111

    var R2010 : double = a * R1011 + R0011
    var R1020 : double = a * R0021

    var R0210 : double = b * R0111 + R0011
    var R0120 : double = b * R0021

    var R3000 : double = a * R2001 + R1001 + R1001
    var R0300 : double = b * R0201 + R0101 + R0101
    var R0030 : double = c * R0021 + R0011 + R0011

    var P : double
    var result0 : double
    var result1 : double
    var result2 : double
    var result3 : double
    var result4 : double
    var result5 : double
    var result6 : double
    var result7 : double
    var result8 : double
    var result9 : double

    P = r_density[d_offset].value
    result0 = R000[0] * P
    result1 = R1000 * P
    result2 = R0100 * P
    result3 = R0010 * P

    P = r_density[d_offset + 1].value
    result0 -= R1000 * P
    result1 -= R2000 * P
    result2 -= R1100 * P
    result3 -= R1010 * P

    P = r_density[d_offset + 2].value
    result0 -= R0100 * P
    result1 -= R1100 * P
    result2 -= R0200 * P
    result3 -= R0110 * P

    P = r_density[d_offset + 3].value
    result0 -= R0010 * P
    result1 -= R1010 * P
    result2 -= R0110 * P
    result3 -= R0020 * P

    P = r_density[d_offset + 4].value
    result0 += R1100 * P
    result1 += R2100 * P
    result2 += R1200 * P
    result3 += R1110 * P

    P = r_density[d_offset + 5].value
    result0 += R1010 * P
    result1 += R2010 * P
    result2 += R1110 * P
    result3 += R1020 * P

    P = r_density[d_offset + 6].value
    result0 += R0110 * P
    result1 += R1110 * P
    result2 += R0210 * P
    result3 += R0120 * P

    P = r_density[d_offset + 7].value
    result0 += R2000 * P
    result1 += R3000 * P
    result2 += R2100 * P
    result3 += R2010 * P

    P = r_density[d_offset + 8].value
    result0 += R0200 * P
    result1 += R1200 * P
    result2 += R0300 * P
    result3 += R0210 * P

    P = r_density[d_offset + 9].value
    result0 += R0020 * P
    result1 += R1020 * P
    result2 += R0120 * P
    result3 += R0030 * P

    result0 *= lambda
    result1 *= lambda
    result2 *= lambda
    result3 *= lambda

    r_j_values[j_offset].value += result0
    r_j_values[j_offset + 1].value += result1
    r_j_values[j_offset + 2].value += result2
    r_j_values[j_offset + 3].value += result3
  end
end

return SPPP
