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


terra readInput() : float
  var input : float
  if regentlib.c.scanf("%f", &input) == 1 then
    return input
  else
    return -1
  end
end


task toplevel()
  var r_boys = region(ispace(int2d, {251, getBoysLargestJ() + 1}), double)
  populateBoysRegion(r_boys)
  while true do
    var t : float = readInput()
    if t < 0 then
      break
    end
    [generateStatementsComputeBoys(R000, 16, t, r_boys)]
    regentlib.c.printf("t=%.16g ", t);
    [printValues()]
    regentlib.c.printf("\n")
  end
end

regentlib.start(toplevel)
