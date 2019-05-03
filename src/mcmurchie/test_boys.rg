import "regent"

require "mcmurchie.generate_R_table"
require "mcmurchie.populate_boys_region"

local R000 = {}
for j = 0, 15 do -- inclusive
  R000[j] = regentlib.newsymbol(double, "R000"..j)
end


local function printValues()
  local statements = terralib.newlist()
  for j, v in pairs(R000) do
    statements:insert(rquote
      regentlib.c.printf("R000%d = %.16g\n", j, v)
    end)
  end
  return statements
end


task toplevel()
  var args = regentlib.c.legion_runtime_get_input_args()
  regentlib.assert(args.argc == 3, "Usage: regent mcmurchie/test_boys.rg t alpha")
  var t : double = regentlib.c.atof(args.argv[1])
  var alpha : double = regentlib.c.atof(args.argv[2])
  var r_boys = region(ispace(int2d, {121, getBoysLargestJ() + 1}), double)
  populateBoysRegion(r_boys);
  [generateStatementsComputeR000(R000, 16, t, alpha, r_boys)];
  [printValues()]
end

regentlib.start(toplevel)
