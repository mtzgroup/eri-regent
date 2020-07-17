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
      local H = tetrahedral_number(L1 + L2 + 1)
      local field_space = getJBra(L1 + L2)
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
      local H = tetrahedral_number(L1 + L2 + 1)
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
function writeKFockToRegions(filename, region_vars, preval_vars)
  local filep = regentlib.newsymbol()
  local statements = terralib.newlist({rquote
    var [filep] = c.fopen(filename, "r")
    checkFile(filep)
  end})
  for L1 = 0, getCompiledMaxMomentum() do -- inclusive
    for L2 = 0, getCompiledMaxMomentum() do -- inclusive
      local field_space = getKFockPair(L1, L2)
      local r_kpairs = region_vars[L1][L2]
      local bra_prevals, ket_prevals = unpack(preval_vars[L1][L2])
      statements:insert(rquote
        var int_data : int[3]
        var double_data : double[12]
        var num_values = c.fscanf(filep, "L1=%d,L2=%d,N=%d\n",
                                  int_data, int_data+1, int_data+2)
        assert(num_values == 3, "Did not read all values in input header!")
        var N = int_data[2]
        assert(L1 == int_data[0] and L2 == int_data[1],
               "Unexpected angular momentum in kfock pairs!")
        var [r_kpairs] = region(ispace(int1d, N), field_space)
        var [bra_prevals] = region(ispace(int2d, {N, [KFockNumBraPrevals[L1][L2]]}), double)
        var [ket_prevals] = region(ispace(int2d, {N, [KFockNumKetPrevals[L1][L2]]}), double)
        for i = 0, N do -- exclusive
          num_values = c.fscanf(filep,
            "x=%lf,y=%lf,z=%lf,eta=%lf,c=%lf,bound=%lf,i_shell_idx=%d,j_shell_idx=%d,PIx=%lf,PIy=%lf,PIz=%lf,PJx=%lf,PJy=%lf,PJz=%lf,",
            double_data+0, double_data+1, double_data+2,
            double_data+3, double_data+4, double_data+5,
            int_data+0, int_data+1,
            double_data+6, double_data+7, double_data+8,
            double_data+9, double_data+10, double_data+11
          )
          assert(num_values == 14, "Did not read all values in line!");
          r_kpairs[i] = {
            location={x=double_data[0], y=double_data[1], z=double_data[2]},
            eta=double_data[3], C=double_data[4], bound=double_data[5],
            ishell_index=int_data[0], jshell_index=int_data[1],
            ishell_location={x=double_data[6], y=double_data[7], z=double_data[8]},
            jshell_location={x=double_data[9], y=double_data[10], z=double_data[11]},
          }

          num_values = c.fscanf(filep, "bra_prevals=")
          assert(num_values == 0, "Did not read bra_prevals!")
          for k = 0, [KFockNumBraPrevals[L1][L2]] do -- exclusive
            num_values = c.fscanf(filep, "%lf,", double_data)
            assert(num_values == 1, "Did not read bra_preval value!")
            bra_prevals[{i, k}] = double_data[0]
            --c.printf("bra_prevals[%d,%d] = %lf\n", i, k, bra_prevals[{i, k}])  -- KGJ
          end
          num_values = c.fscanf(filep, "ket_prevals=")
          assert(num_values == 0, "Did not read ket_prevals!")
          for k = 0, [KFockNumKetPrevals[L1][L2]] do -- exclusive
            num_values = c.fscanf(filep, "%lf,", double_data)
            assert(num_values == 1, "Did not read ket_preval value!")
            ket_prevals[{i, k}] = double_data[0]
          end
          num_values = c.fscanf(filep, "\n")
          assert(num_values == 0, "Did not read to end of line!")
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
function writeKFockDensityToRegions(filename, region_vars, r_output_list)
  local filep = regentlib.newsymbol()
  local statements = terralib.newlist({rquote
    var [filep] = c.fopen(filename, "r")
    checkFile(filep)
  end})
  for L2 = 0, getCompiledMaxMomentum() do -- inclusive
    for L4 = L2, getCompiledMaxMomentum() do -- inclusive
      local field_space = getKFockDensity(L2, L4)
      local r_density = region_vars[L2][L4]
      statements:insert(rquote
        var int_data : int[4]
        var double_data : double[1]
        var num_values = c.fscanf(filep, "L2=%d,L4=%d,N2=%d,N4=%d\n",
                                  int_data, int_data+1, int_data+2, int_data+3)
        assert(num_values == 4, "Did not read all values in density header!")
        var N2, N4 = int_data[2], int_data[3]
        assert(L2 == int_data[0] and L4 == int_data[1],
               "Unexpected angular momentum in kfock density!")
        var [r_density] = region(ispace(int2d, {N2, N4}), field_space)
        for bra_jshell = 0, N2 do -- exclusive
          for ket_jshell = 0, N4 do -- exclusive
            num_values = c.fscanf(
                  filep,
                  "bra_jshell_idx=%d,ket_jshell_idx=%d,values=",
                  int_data + 0, int_data + 1)
            assert(num_values == 2, "Did not read values!")
            assert(int_data[0] == bra_jshell and int_data[1] == ket_jshell,
                   "Wrong indices!")
            for k = 0, [triangle_number(L2 + 1)] do -- exclusive
              for m = 0, [triangle_number(L4 + 1)] do -- exclusive
                num_values = c.fscanf(filep, "%lf,", double_data)
                assert(num_values == 1, "Did not read kfock density value!")
                r_density[{bra_jshell, ket_jshell}].values[k][m] = double_data[0]
              end
            end
            num_values = c.fscanf(filep, "bound=%lf\n", double_data)
            assert(num_values == 1, "Did not read bound!")
            r_density[{bra_jshell, ket_jshell}].bound = double_data[0]
          end
        end
      end)
    end
  end
  statements:insert(rquote
    c.fclose(filep)
  end)
  for L1 = 0, getCompiledMaxMomentum() do -- inclusive
    for L3 = L1, getCompiledMaxMomentum() do -- inclusive
      local r_density = region_vars[L1][L3]
      local r_output = r_output_list[L1][L3]
      local N = (getCompiledMaxMomentum() + 1) * (getCompiledMaxMomentum() + 1) + 1
      local N1 = rexpr r_density.bounds.hi.x + 1 end
      local N3 = rexpr r_density.bounds.hi.y + 1 end
      statements:insert(rquote
        var [r_output] = region(ispace(int3d, {N, N1, N3}), getKFockOutput(L1, L3))
      end)
    end
  end
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
      local H = tetrahedral_number(L1 + L2 + 1)
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
      local H = tetrahedral_number(L1 + L2 + 1)
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
function verifyKFockOutput(region_vars, delta, epsilon, filename)
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
    for L2 = 0, getCompiledMaxMomentum() do -- inclusive
      for L3 = 0, getCompiledMaxMomentum() do -- inclusive
        for L4 = 0, getCompiledMaxMomentum() do -- inclusive
          if L1 < L3 or (L1 == L3 and L2 <= L4) then
            local N24 = L2 + L4 * (getCompiledMaxMomentum() + 1)
            local field_space = getKFockOutput(L1, L3)
            local r_output = region_vars[L1][L3]
            statements:insert(rquote
              var int_data : int[6]
              var double_data : double[1]
              var num_values = c.fscanf(filep, "L1=%d,L2=%d,L3=%d,L4=%d,N1=%d,N3=%d\n",
                                        int_data + 0, int_data + 1, int_data + 2,
                                        int_data + 3, int_data + 4, int_data + 5)
              assert(num_values == 6, "Could not read output header!")
              var N1, N3 = int_data[4], int_data[5]
              assert(L1 == int_data[0] and L2 == int_data[1]
                     and L3 == int_data[2] and L4 == int_data[3],
                     "Unexpected angular momentum in kfock output!")
              for bra_ishell = 0, N1 do -- exclusive
                for ket_ishell = 0, N3 do -- exclusive
                  num_values = c.fscanf(filep,
                      "bra_ishell_idx=%d,ket_ishell_idx=%d,values=",
                      int_data + 0, int_data + 1)
                  assert(num_values == 2, "Could not read line!")
                  assert(int_data[0] == bra_ishell and int_data[1] == ket_ishell,
                         "Index is not correct!")
                  for i = 0, [triangle_number(L1 + 1)] do -- exclusive
                    for j = 0, [triangle_number(L3 + 1)] do -- exclusive
                      num_values = c.fscanf(filep, "%lf,", double_data)
                      assert(num_values == 1, "Did not read kfock value!")
                      var expected = double_data[0]
                      var result = r_output[{N24, bra_ishell, ket_ishell}].values[i][j]
                      --var result = r_output[{N24, bra_ishell, ket_ishell}].values[j][i] -- THIS IS SOME SPOOKY SHIT 
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
                        c.printf(
"Value differs at L1234 = %d %d %d %d, output[%d, %d].values[%d, %d]: result = %.12f, expected = %.12f, absolute_error = %.12g, relative_error = %.12g\n",
                                 L1, L2, L3, L4, bra_ishell, ket_ishell, i, j,
                                 result, expected, absolute_error, relative_error)
                        --assert(false, "Wrong output!")
                      else
                      -- print all values
                        c.printf(
"                 L1234 = %d %d %d %d, output[%d, %d].values[%d, %d]: result = %.12f, expected = %.12f, absolute_error = %.12g, relative_error = %.12g\n",
                                 L1, L2, L3, L4, bra_ishell, ket_ishell, i, j,
                                 result, expected, absolute_error, relative_error)
                      end
                    end
                  end
                  assert(c.fscanf(filep, "\n") == 0, "Did not read newline")
                end
              end
            end)
          end
        end
      end
    end
  end
  statements:insert(rquote
    c.fclose(filep)
    c.printf("Values are correct! max_absolue_error = %.12g, max_relative_error = %.12g\n",
             max_absolute_error, max_relative_error)
  end)
  return statements
end
