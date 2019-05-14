import "regent"

require "mcmurchie.generate_R_table"
require "mcmurchie.populate_boys_region"

if arg[1] == nil then
  assert(false, "Usage: regent "..arg[0].." [max j]")
end
local max_j = tonumber(arg[1])

local boys = {}
for j = 0, max_j do -- inclusive
  boys[j] = regentlib.newsymbol(double, "boys"..j)
end


local function printValues()
  local statements = terralib.newlist()
  for j, v in pairs(boys) do
    statements:insert(rquote
      regentlib.c.printf("boys%d=%.16g ", j, v)
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
    [generateStatementsComputeBoys(boys, max_j+1, t, r_boys)]
    regentlib.c.printf("t=%.16g ", t);
    [printValues()]
    regentlib.c.printf("\n")
  end
end

regentlib.start(toplevel)
