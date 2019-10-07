import "regent"
require "fields"
require "helper"

local c = regentlib.c
local assert = regentlib.assert

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
function writeGaussiansToRegions(filename, region_vars)
  local filep = regentlib.newsymbol()
  local int_data = regentlib.newsymbol(int[2])
  local double_data = regentlib.newsymbol(double[6])
  local statements = terralib.newlist({rquote
    var [filep] = c.fopen(filename, "r")
    checkFile(filep)
    var [int_data]
    var [double_data]
  end})
  for L_lua = 0, #region_vars do -- inclusive
    local region_var = region_vars[L_lua]
    statements:insert(rquote
      var num_values = c.fscanf(filep, "L=%d,N=%d\n", int_data, int_data+1)
      assert(num_values == 2, "Did not read all values in header!")
      var L = int_data[0]
      var N = int_data[1]
      assert(L == L_lua, "Unexpected angular momentum!")
      var [region_var] = region(ispace(int1d, N), Gaussian)
      for j = 0, N do -- exclusive
        num_values = c.fscanf(filep,
          "x=%lf,y=%lf,z=%lf,eta=%lf,c=%lf,bound=%lf",
          double_data+0, double_data+1, double_data+2,
          double_data+3, double_data+4, double_data+5
        )
        assert(num_values == 6, "Did not read all values in line!");
        region_var[j] = {
          x=double_data[0], y=double_data[1], z=double_data[2],
          eta=double_data[3], C=double_data[4], bound=double_data[5]
        }
        assert(c.fscanf(filep, "\n") == 0, "Could not read newline!")
      end
    end)
  end
  statements:insert(rquote
    c.fclose(filep)
  end)
  return statements
end

-- Writes data found in `filename` to an array of regions given by `region_vars`.
function writeGaussiansWithDensityToRegions(filename, region_vars)
  local filep = regentlib.newsymbol()
  local int_data = regentlib.newsymbol(int[2])
  local double_data = regentlib.newsymbol(double[6])
  local statements = terralib.newlist({rquote
    var [filep] = c.fopen(filename, "r")
    checkFile(filep)
    var [int_data]
    var [double_data]
  end})
  for L_lua = 0, #region_vars do -- inclusive
    local region_var = region_vars[L_lua]
    local H = computeH(L_lua)
    local field_space = getGaussianWithDensity(L_lua)
    statements:insert(rquote
      var num_values = c.fscanf(filep, "L=%d,N=%d\n", int_data, int_data+1)
      assert(num_values == 2, "Did not read all values in header!")
      var L = int_data[0]
      var N = int_data[1]
      assert(L == L_lua, "Unexpected angular momentum!")
      var [region_var] = region(ispace(int1d, N), field_space)
      for j = 0, N do -- exclusive
        num_values = c.fscanf(filep,
          "x=%lf,y=%lf,z=%lf,eta=%lf,c=%lf,bound=%lf,density=",
          double_data+0, double_data+1, double_data+2,
          double_data+3, double_data+4, double_data+5
        )
        assert(num_values == 6, "Did not read all values in line!");
        var density : double[H]
        for k = 0, H do -- exclusive
          num_values = c.fscanf(filep, "%lf;", density+k)
          assert(num_values == 1, "Could not read density value!")
        end
        region_var[j] = {
          x=double_data[0], y=double_data[1], z=double_data[2],
          eta=double_data[3], C=double_data[4], bound=double_data[5],
          density=density
        }
        assert(c.fscanf(filep, "\n") == 0, "Could not read newline!")
      end
    end)
  end
  statements:insert(rquote
    c.fclose(filep)
  end)
  return statements
end

-- local r_jbras0 = regentlib.newsymbol()
-- local r_jbras1 = regentlib.newsymbol()
-- local r_jbras2 = regentlib.newsymbol()
-- local task test()
--   var args = c.legion_runtime_get_input_args()
--   var filename = args.argv[1];
--   [write_gaussians_with_density_to_regions(filename, {r_jbras0, r_jbras1, r_jbras2})]
--   c.printf("L=0\n")
--   for bra in [r_jbras0] do
--     c.printf("(%f, %f, %f), eta: %f, C: %f, bound: %f\n",
--              bra.x, bra.y, bra.z, bra.eta, bra.C, bra.bound)
--   end
--   c.printf("L=1\n")
--   for bra in [r_jbras1] do
--     c.printf("(%f, %f, %f), eta: %f, C: %f, bound: %f\n",
--              bra.x, bra.y, bra.z, bra.eta, bra.C, bra.bound)
--   end
--   c.printf("L=2\n")
--   for bra in [r_jbras2] do
--     c.printf("(%f, %f, %f), eta: %f, C: %f, bound: %f\n",
--              bra.x, bra.y, bra.z, bra.eta, bra.C, bra.bound)
--   end
-- end
--
-- regentlib.start(test)
