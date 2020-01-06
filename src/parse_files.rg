import "regent"

require "fields"
require "helper"

local assert = regentlib.assert
local c = regentlib.c
local fabs = regentlib.fabs(double)

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

-- Writes data found in `filename` to an array of regions given by `region_vars`
function writeJBrasToRegions(filename, region_vars)
  local filep = regentlib.newsymbol()
  local statements = terralib.newlist({rquote
    var [filep] = c.fopen(filename, "r")
    checkFile(filep)
  end})
  for L1 = 0, getCompiledMaxMomentum() do -- inclusive
    for L2 = L1, getCompiledMaxMomentum() do -- inclusive
      local L12 = L1 + L2
      local H = computeH(L12)
      local field_space = getJBra(L12)
      local r_jbras = region_vars[L1][L2]
      statements:insert(rquote
        var int_data : int[3]
        var double_data : double[6]
        var num_values = c.fscanf(filep, "L1=%d,L2=%d,N=%d\n",
                                  int_data, int_data+1, int_data+2)
        assert(num_values == 3, "Did not read all values in header!")
        var N = int_data[2]
        assert(L1 == int_data[0] and L2 == int_data[1],
               "Unexpected angular momentum!")
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
            location={x=double_data[0], y=double_data[1], z=double_data[2]},
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

-- Writes data found in `filename` to an array of regions given by `region_vars`
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
        var num_values = c.fscanf(filep, "L1=%d,L2=%d,N=%d\n",
                                  int_data, int_data+1, int_data+2)
        assert(num_values == 3, "Did not read all values in header!")
        var N = int_data[2]
        assert(L1 == int_data[0] and L2 == int_data[1],
               "Unexpected angular momentum!")
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
            location={x=double_data[0], y=double_data[1], z=double_data[2]},
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

-- Writes data found in `filename` to an array of regions given by `region_vars`
function writeKFockToRegions(filename, region_vars)
  local filep = regentlib.newsymbol()
  local statements = terralib.newlist({rquote
    var [filep] = c.fopen(filename, "r")
    checkFile(filep)
  end})
  for L1 = 0, getCompiledMaxMomentum() do -- inclusive
    for L2 = L1, getCompiledMaxMomentum() do -- inclusive
      local field_space = getKFockPair(L1, L2)
      local r_kpairs = region_vars[L1][L2]
      statements:insert(rquote
        var int_data : int[3]
        var double_data : double[6]
        var num_values = c.fscanf(filep, "L1=%d,L2=%d,N=%d\n",
                                  int_data, int_data+1, int_data+2)
        assert(num_values == 3, "Did not read all values in header!")
        var N = int_data[2]
        assert(L1 == int_data[0] and L2 == int_data[1],
               "Unexpected angular momentum!")
        var [r_kpairs] = region(ispace(int1d, N), field_space)
        for i = 0, N do -- exclusive
          num_values = c.fscanf(filep,
            "x=%lf,y=%lf,z=%lf,eta=%lf,c=%lf,bound=%lf,use_upper_diag=%d,shell_idx=%d\n",
            double_data+0, double_data+1, double_data+2,
            double_data+3, double_data+4, double_data+5,
            int_data+0, int_data+1
          )
          assert(num_values == 8, "Did not read all values in line!");
          -- TODO
          r_kpairs[i] = {
            location={x=double_data[0], y=double_data[1], z=double_data[2]},
            eta=double_data[3], C=double_data[4], bound=double_data[5],
            ishell_location={x=0, y=0, z=0}, jshell_location={x=0, y=0, z=0},
            ishell_index=-1, jshell_index=-1,
          }
        end
      end)
    end
  end
  statements:insert(rquote
    c.fclose(filep)
  end)
  return statements
end

-- Writes data found in `filename` to an array of regions given by `region_vars`
function writeKFockDensityToRegions(filename, region_vars)
  local filep = regentlib.newsymbol()
  local statements = terralib.newlist({rquote
    var [filep] = c.fopen(filename, "r")
    checkFile(filep)
  end})
  for L1 = 0, getCompiledMaxMomentum() do -- inclusive
    for L2 = L1, getCompiledMaxMomentum() do -- inclusive
      local field_space = getKFockDensity(L1, L2)
      local r_density = region_vars[L1][L2]
      statements:insert(rquote
        var int_data : int[4]
        var double_data : double[6]
        var num_values = c.fscanf(filep, "L1=%d,L2=%d,N1=%d,N2=%d\n",
                                  int_data, int_data+1, int_data+2, int_data+3)
        assert(num_values == 4, "Did not read all values in header!")
        var N1, N2 = int_data[2], int_data[3]
        assert(L1 == int_data[0] and L2 == int_data[1],
               "Unexpected angular momentum!")
        var [r_density] = region(ispace(int2d, {N1, N2}), field_space)
        for i = 0, N1 do -- exclusive
          for j = 0, N2 do -- exclusive
            num_values = c.fscanf(filep, "i=%d,j=%d,value=%lf\n",
                                  int_data+0, int_data+1, double_data+0)
            assert(num_values == 3, "Did not read all values in line!");
            -- TODO
            -- r_density[i, j] = {
            -- }
          end
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
      local L12 = L1 + L2
      local H = computeH(L12)
      statements:insert(rquote
        c.fprintf(filep, "L1=%d,L2=%d,N=%d\n",
                  L1, L2, [r_jbras_list[L1][L2]].volume)
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

-- Verify the output is correct
-- `delta` is the maximum allowed absolute error
-- `epsilon` is the maximum allowed relative error
function verifyOutput(r_jbras_list, delta, epsilon, filename)
  local filep = regentlib.newsymbol()
  local max_absolute_error = regentlib.newsymbol(double)
  local max_relative_error = regentlib.newsymbol(double)
  local statements = terralib.newlist({rquote
    var [filep] = c.fopen(filename, "r")
    checkFile(filep)
    var [max_absolute_error] = -1
    var [max_relative_error] = -1
  end})
  for L1 = 0, getCompiledMaxMomentum() do -- inclusive
    for L2 = L1, getCompiledMaxMomentum() do -- inclusive
      local L12 = L1 + L2
      local H = computeH(L12)
      local r_jbras = r_jbras_list[L1][L2]
      statements:insert(rquote
        var int_data : int[3]
        var double_data : double[1]
        var num_values = c.fscanf(filep, "L1=%d,L2=%d,N=%d\n",
                                  int_data+0, int_data+1, int_data+2)
        assert(num_values == 3, "Did not read angular momentum!")
        assert(L1 == int_data[0] and L2 == int_data[1],
               "Wrong angular momentum!")
        var N = int_data[2]
        for i = 0, N do -- exclusive
          for j = 0, H do -- exclusive
            num_values = c.fscanf(filep, "%lf\t", double_data)
            assert(num_values == 1, "Did not read value!")
            var expected = double_data[0]
            var result = r_jbras[i].output[j]
            var absolute_error = fabs(result - expected)
            var relative_error = fabs(absolute_error / expected)
            if absolute_error > max_absolute_error then
              max_absolute_error = absolute_error
            end
            if relative_error > max_relative_error then
              max_relative_error = relative_error
            end
            if [bool](c.isnan(result)) or [bool](c.isinf(result))
                or (absolute_error > delta and relative_error > epsilon) then
              c.printf("Value differs at L1 = %d, L2 = %d, JBra[%d].output[%d]: result = %.12f, expected = %.12f, absolute_error = %.12g, relative_error = %.12g\n",
                       L1, L2, i, j, result, expected, absolute_error, relative_error)
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
    c.printf("Values are correct! max_absolue_error = %.12g, max_relative_error = %.12g\n",
             max_absolute_error, max_relative_error)
  end)
  return statements
end

-- Verify the output is correct
-- `delta` is the maximum allowed absolute error
-- `epsilon` is the maximum allowed relative error
task verifyKFockOutput(r_output : region(ispace(int2d), double),
                       delta : double, epsilon : double,
                       filename : int8[512])
where
  reads(r_output)
do
  var filep = c.fopen(filename, "r")
  checkFile(filep)
  var max_absolute_error = -1
  var max_relative_error = -1
  var int_data : int[3]
  var double_data : double[1]
  var num_values = c.fscanf(filep, "N=%d\n", int_data + 0)
  assert(num_values == 1, "Did not read N!")
  var N = int_data[0]
  assert(N * N == r_output.volume, "Output does not have correct size!")
  for i = 0, N do -- exclusive
    for j = 0, N do -- exclusive
      num_values = c.fscanf(filep, "i=%d,j=%d,value=%lf\n",
                            int_data + 0, int_data + 1, double_data + 0)
      assert(num_values == 3, "Did not read all values!")
      assert(int_data[0] == i and int_data[1] == j, "Index is not correct!")
      var expected = double_data[0]
      var result = r_output[{i, j}]
      var absolute_error = fabs(result - expected)
      var relative_error = fabs(absolute_error / expected)
      if absolute_error > max_absolute_error then
        max_absolute_error = absolute_error
      end
      if relative_error > max_relative_error then
        max_relative_error = relative_error
      end
      if [bool](c.isnan(result)) or [bool](c.isinf(result))
          or (absolute_error > delta and relative_error > epsilon) then
        c.printf("Value differs at output[%d, %d]: result = %.12f, expected = %.12f, absolute_error = %.12g, relative_error = %.12g\n",
                 i, j, result, expected, absolute_error, relative_error)
        assert(false, "Wrong output!")
      end
    end
  end
  c.fclose(filep)
  c.printf("Values are correct!\n")
  c.printf("Values are correct! max_absolute_error = %.12g, max_relative_error = %.12g\n",
           max_absolute_error, max_relative_error)
end
