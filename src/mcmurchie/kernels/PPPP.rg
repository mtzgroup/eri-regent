import "regent"

local function PPPP(a, b, c, R000, lambda,
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
    var R1003 : double = a * R000[4]
    var R0103 : double = b * R000[4]
    var R0013 : double = c * R000[4]

    var R1100 : double = a * R0101
    var R1010 : double = a * R0011
    var R0110 : double = b * R0011
    var R1101 : double = a * R0102
    var R1011 : double = a * R0012
    var R0111 : double = b * R0012
    var R1102 : double = a * R0103
    var R1012 : double = a * R0013
    var R0112 : double = b * R0013

    var R2000 : double = a * R1001 + R000[1]
    var R0200 : double = b * R0101 + R000[1]
    var R0020 : double = c * R0011 + R000[1]
    var R2001 : double = a * R1002 + R000[2]
    var R0201 : double = b * R0102 + R000[2]
    var R0021 : double = c * R0012 + R000[2]
    var R2002 : double = a * R1003 + R000[3]
    var R0202 : double = b * R0103 + R000[3]
    var R0022 : double = c * R0013 + R000[3]

    var R2100 : double = a * R1101 + R0101
    var R1200 : double = a * R0201
    var R1110 : double = a * R0111
    var R2010 : double = a * R1011 + R0011
    var R1020 : double = a * R0021
    var R0210 : double = b * R0111 + R0011
    var R0120 : double = b * R0021
    var R2101 : double = a * R1102 + R0102
    var R1201 : double = a * R0202
    var R1111 : double = a * R0112
    var R2011 : double = a * R1012 + R0012
    var R1021 : double = a * R0022
    var R0211 : double = b * R0112 + R0012
    var R0121 : double = b * R0022

    var R3000 : double = a * R2001 + R1001 + R1001
    var R0300 : double = b * R0201 + R0101 + R0101
    var R0030 : double = c * R0021 + R0011 + R0011
    var R3001 : double = a * R2002 + R1002 + R1002
    var R0301 : double = b * R0202 + R0102 + R0102
    var R0031 : double = c * R0022 + R0012 + R0012

    var R2200 : double = a * R1201 + R0201
    var R2110 : double = a * R1111 + R0111
    var R1210 : double = a * R0211
    var R3100 : double = a * R2101 + R1101 + R1101
    var R1300 : double = a * R0301
    var R1120 : double = a * R0121

    var R2020 : double = a * R1021 + R0021
    var R3010 : double = a * R2011 + R1011 + R1011
    var R1030 : double = a * R0031

    var R0220 : double = b * R0121 + R0021
    var R0310 : double = b * R0211 + R0111 + R0111
    var R0130 : double = b * R0031

    var R4000 : double = a * R3001 + R2001 + R2001 + R2001
    var R0400 : double = b * R0301 + R0201 + R0201 + R0201
    var R0040 : double = c * R0031 + R0021 + R0021 + R0021

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
    result4 = R1100 * P
    result5 = R1010 * P
    result6 = R0110 * P
    result7 = R2000 * P
    result8 = R0200 * P
    result9 = R0020 * P

    P = r_density[d_offset + 1].value
    result0 -= R1000 * P
    result1 -= R2000 * P
    result2 -= R1100 * P
    result3 -= R1010 * P
    result4 -= R2100 * P
    result5 -= R2010 * P
    result6 -= R1110 * P
    result7 -= R3000 * P
    result8 -= R1200 * P
    result9 -= R1020 * P

    P = r_density[d_offset + 2].value
    result0 -= R0100 * P
    result1 -= R1100 * P
    result2 -= R0200 * P
    result3 -= R0110 * P
    result4 -= R1200 * P
    result5 -= R1110 * P
    result6 -= R0210 * P
    result7 -= R2100 * P
    result8 -= R0300 * P
    result9 -= R0120 * P

    P = r_density[d_offset + 3].value
    result0 -= R0010 * P
    result1 -= R1010 * P
    result2 -= R0110 * P
    result3 -= R0020 * P
    result4 -= R1110 * P
    result5 -= R1020 * P
    result6 -= R0120 * P
    result7 -= R2010 * P
    result8 -= R0210 * P
    result9 -= R0030 * P

    P = r_density[d_offset + 4].value
    result0 += R1100 * P
    result1 += R2100 * P
    result2 += R1200 * P
    result3 += R1110 * P
    result4 += R2200 * P
    result5 += R2110 * P
    result6 += R1210 * P
    result7 += R3100 * P
    result8 += R1300 * P
    result9 += R1120 * P

    P = r_density[d_offset + 5].value
    result0 += R1010 * P
    result1 += R2010 * P
    result2 += R1110 * P
    result3 += R1020 * P
    result4 += R2110 * P
    result5 += R2020 * P
    result6 += R1120 * P
    result7 += R3010 * P
    result8 += R1210 * P
    result9 += R1030 * P

    P = r_density[d_offset + 6].value
    result0 += R0110 * P
    result1 += R1110 * P
    result2 += R0210 * P
    result3 += R0120 * P
    result4 += R1210 * P
    result5 += R1120 * P
    result6 += R0220 * P
    result7 += R2110 * P
    result8 += R0310 * P
    result9 += R0130 * P

    P = r_density[d_offset + 7].value
    result0 += R2000 * P
    result1 += R3000 * P
    result2 += R2100 * P
    result3 += R2010 * P
    result4 += R3100 * P
    result5 += R3010 * P
    result6 += R2110 * P
    result7 += R4000 * P
    result8 += R2200 * P
    result9 += R2020 * P

    P = r_density[d_offset + 8].value
    result0 += R0200 * P
    result1 += R1200 * P
    result2 += R0300 * P
    result3 += R0210 * P
    result4 += R1300 * P
    result5 += R1210 * P
    result6 += R0310 * P
    result7 += R2200 * P
    result8 += R0400 * P
    result9 += R0220 * P

    P = r_density[d_offset + 9].value
    result0 += R0020 * P
    result1 += R1020 * P
    result2 += R0120 * P
    result3 += R0030 * P
    result4 += R1120 * P
    result5 += R1030 * P
    result6 += R0130 * P
    result7 += R2020 * P
    result8 += R0220 * P
    result9 += R0040 * P

    result0 *= lambda
    result1 *= lambda
    result2 *= lambda
    result3 *= lambda
    result4 *= lambda
    result5 *= lambda
    result6 *= lambda
    result7 *= lambda
    result8 *= lambda
    result9 *= lambda

    r_j_values[j_offset].value += result0
    r_j_values[j_offset + 1].value += result1
    r_j_values[j_offset + 2].value += result2
    r_j_values[j_offset + 3].value += result3
    r_j_values[j_offset + 4].value += result4
    r_j_values[j_offset + 5].value += result5
    r_j_values[j_offset + 6].value += result6
    r_j_values[j_offset + 7].value += result7
    r_j_values[j_offset + 8].value += result8
    r_j_values[j_offset + 9].value += result9
  end
end

return PPPP
