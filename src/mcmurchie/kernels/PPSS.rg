import "regent"

local function PPSS(a, b, c, R000, lambda,
                    r_j_values, j_offset,
                    r_density, d_offset)
  return rquote
    var R1000 : double = a * [R000[1]]
    var R0100 : double = b * [R000[1]]
    var R0010 : double = c * [R000[1]]

    var R1001 : double = a * [R000[2]]
    var R0101 : double = b * [R000[2]]
    var R0011 : double = c * [R000[2]]

    var R1100 : double = a * R0101
    var R1010 : double = a * R0011
    var R0110 : double = b * R0011

    var R2000 : double = a * R1001 + [R000[1]]
    var R0200 : double = b * R0101 + [R000[1]]
    var R0020 : double = c * R0011 + [R000[1]]

    var P0 : double = r_density[d_offset].value

    r_j_values[j_offset].value += lambda * [R000[0]] * P0
    r_j_values[j_offset + 1].value += lambda * R1000 * P0
    r_j_values[j_offset + 2].value += lambda * R0100 * P0
    r_j_values[j_offset + 3].value += lambda * R0010 * P0
    r_j_values[j_offset + 4].value += lambda * R1100 * P0
    r_j_values[j_offset + 5].value += lambda * R1010 * P0
    r_j_values[j_offset + 6].value += lambda * R0110 * P0
    r_j_values[j_offset + 7].value += lambda * R2000 * P0
    r_j_values[j_offset + 8].value += lambda * R0200 * P0
    r_j_values[j_offset + 9].value += lambda * R0020 * P0
  end
end

return PPSS
