import "regent"

require "fields"
require "helper"

local assert = regentlib.assert
local c = regentlib.c
local fabs = regentlib.fabs(double)

-- TODO: Verify these functions are correct!

local terra checkFile(filep : &c.FILE)
  assert(filep ~= nil, "Could not open file!")
end

terra readParametersFile(filename : &int8, data : &double)
  var filep = c.fopen(filename, "r")
  checkFile(filep)
  var num_values = c.fscanf(filep,
    "scalfr=%lf\nscallr=%lf\nomega=%lf\nthresp=%lf\nthredp=%lf\n",
    data, data+1, data+2, data+3, data+4)
  assert(num_values == 5, "Could not read all of parameters file!")
  c.fclose(filep)
end

-- Writes data found in `filename` to an array of regions given by `region_vars`.
function writeJBrasToRegions(filename, region_vars)
  local filep = regentlib.newsymbol()
  local statements = terralib.newlist({rquote
    var [filep] = c.fopen(filename, "r")
    checkFile(filep)
  end})
  for L1 = 0, getCompiledMaxMomentum() do -- inclusive
    for L2 = L1, getCompiledMaxMomentum() do -- inclusive
      local H = computeH(L1 + L2)
      local field_space = getJBra(L1 + L2)
      local r_jbras = region_vars[L1][L2]
      statements:insert(rquote
        var int_data : int[3]
        var double_data : double[6]
        var num_values = c.fscanf(filep, "L1=%d,L2=%d,N=%d\n", int_data, int_data+1, int_data+2)
        assert(num_values == 3, "Did not read all values in header!")
        var N = int_data[2]
        assert(L1 == int_data[0] and L2 == int_data[1], "Unexpected angular momentum!")
        var [r_jbras] = region(ispace(int1d, N), field_space)
        var zeros : double[H]
        for i = 0, H do -- exclusive
          zeros[i] = 0
        end
        for i = 0, N do -- exclusive
          num_values = c.fscanf(filep,
            "x=%lf,y=%lf,z=%lf,eta=%lf,c=%lf,bound=%lf",
            double_data+0, double_data+1, double_data+2,
            double_data+3, double_data+4, double_data+5
          )
          assert(num_values == 6, "Did not read all values in line!");
          r_jbras[i] = {
            x=double_data[0], y=double_data[1], z=double_data[2],
            eta=double_data[3], C=double_data[4], bound=double_data[5],
            output=zeros
          }
          assert(c.fscanf(filep, "\n") == 0, "Could not read newline!")
        end
      end)
    end
  end
  statements:insert(rquote
    c.fclose(filep)
  end)
  return statements
end

-- Writes data found in `filename` to an array of regions given by `region_vars`.
function writeJKetsToRegions(filename, region_vars)
  local filep = regentlib.newsymbol()
  local statements = terralib.newlist({rquote
    var [filep] = c.fopen(filename, "r")
    checkFile(filep)
  end})
  for L1 = 0, getCompiledMaxMomentum() do -- inclusive
    for L2 = L1, getCompiledMaxMomentum() do -- inclusive
      local H = computeH(L1 + L2)
      local field_space = getJKet(L1 + L2)
      local r_jkets = region_vars[L1][L2]
      statements:insert(rquote
        var int_data : int[3]
        var double_data : double[6]
        var num_values = c.fscanf(filep, "L1=%d,L2=%d,N=%d\n", int_data, int_data+1, int_data+2)
        assert(num_values == 3, "Did not read all values in header!")
        var N = int_data[2]
        assert(L1 == int_data[0] and L2 == int_data[1], "Unexpected angular momentum!")
        var [r_jkets] = region(ispace(int1d, N), field_space)
        for i = 0, N do -- exclusive
          num_values = c.fscanf(filep,
            "x=%lf,y=%lf,z=%lf,eta=%lf,c=%lf,bound=%lf,density=",
            double_data+0, double_data+1, double_data+2,
            double_data+3, double_data+4, double_data+5
          )
          assert(num_values == 6, "Did not read all values in line!");
          var density : double[H]
          for j = 0, H do -- exclusive
            num_values = c.fscanf(filep, "%lf;", density+j)
            assert(num_values == 1, "Could not read density value!")
          end
          r_jkets[i] = {
            x=double_data[0], y=double_data[1], z=double_data[2],
            eta=double_data[3], C=double_data[4], bound=double_data[5],
            density=density
          }
          assert(c.fscanf(filep, "\n") == 0, "Could not read newline!")
        end
      end)
    end
  end
  statements:insert(rquote
    c.fclose(filep)
  end)
  return statements
end

-- Writes the output to `filename`
function writeOutput(r_jbras_list, filename)
  local filep = regentlib.newsymbol()
  local statements = terralib.newlist({rquote
    var [filep] = c.fopen(filename, "w")
    checkFile(filep)
  end})
  for L1 = 0, getCompiledMaxMomentum() do -- inclusive
    for L2 = L1, getCompiledMaxMomentum() do -- inclusive
      local H = computeH(L1 + L2)
      statements:insert(rquote
        c.fprintf(filep, "L1=%d,L2=%d,N=%d\n", L1, L2, [r_jbras_list[L1][L2]].volume)
        for bra in [r_jbras_list[L1][L2]] do
          for i = 0, H do -- exclusive
            c.fprintf(filep, "%A\t", bra.output[i])
          end
          c.fprintf(filep, "\n")
        end
      end)
    end
  end
  statements:insert(rquote
    c.fclose(filep)
  end)
  return statements
end

-- Verify the output is within `epsilon` of data from `filename`
function verifyOutput(r_jbras_list, epsilon, filename)
  local filep = regentlib.newsymbol()
  local statements = terralib.newlist({rquote
    var [filep] = c.fopen(filename, "r")
    checkFile(filep)
  end})
  for L1 = 0, getCompiledMaxMomentum() do -- inclusive
    for L2 = L1, getCompiledMaxMomentum() do -- inclusive
      local H = computeH(L1 + L2)
      local r_jbras = r_jbras_list[L1][L2]
      statements:insert(rquote
        var max_error : double = 0 -- TODO
        var int_data : int[3]
        var double_data : double[1]
        var num_values = c.fscanf(filep, "L1=%d,L2=%d,N=%d\n", int_data+0, int_data+1, int_data+2)
        assert(num_values == 3, "Did not read angular momentum!")
        assert(L1 == int_data[0] and L2 == int_data[1], "Wrong angular momentum!")
        var N = int_data[2]
        for i = 0, N do -- exclusive
          for j = 0, H do -- exclusive
            num_values = c.fscanf(filep, "%lf\t", double_data)
            assert(num_values == 1, "Did not read value!")
            var expected = double_data[0]
            var actual = r_jbras[i].output[j]
            var error = fabs(actual - expected)
            if [bool](c.isnan(actual)) or [bool](c.isinf(actual)) or error > epsilon then
              c.printf("Value differs at L1 = %d, L2 = %d, JBra[%d].output[%d]: actual = %.12f, expected = %.12f\n",
                       L1, L2, i, j, actual, expected)
              assert(false, "Wrong output!")
            end
          end
          assert(c.fscanf(filep, "\n") == 0, "Did not read newline")
        end
      end)
    end
  end
  statements:insert(rquote
    c.fclose(filep)
    c.printf("Values are correct!\n")
  end)
  return statements
end
