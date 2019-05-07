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
      regentlib.c.printf("R000%d=%.16g ", j, v)
    end)
  end
  return statements
end


terra readInput(t : &float, alpha : &float) : bool
  return regentlib.c.scanf("%f %f", t, alpha) == 2
end


task toplevel()
  var r_boys = region(ispace(int2d, {121, getBoysLargestJ() + 1}), double)
  populateBoysRegion(r_boys)
  var data : float[2]
  while readInput(data, data+1) do
    var t : float = data[0]
    var alpha : float = data[1];
    [generateStatementsComputeR000(R000, 16, t, alpha, r_boys)]
    regentlib.c.printf("t=%.16g alpha=%.16g ", t, alpha);
    [printValues()]
    regentlib.c.printf("\n")
  end
end

regentlib.start(toplevel)
