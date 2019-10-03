-- Returns the maximum momentum determined at Regent compile time
local _max_momentum = nil
function getCompiledMaxMomentum()
  if _max_momentum == nil then
    for i, arg_value in ipairs(arg) do
      if arg[i] == "-L" then
        _max_momentum = tonumber(arg[i+1])
      end
    end
    assert(_max_momentum ~= nil, "Must have argument `-L [max_momentum]`!")
  end
  return _max_momentum
end

-- The string representation of a given angular momentum
LToStr = {
  [0]="SS", [1]="SP", [2]="PP",
  [3]="PD", [4]="DD", [5]="DF",
  [6]="FF", [7]="FG", [8]="GG"
}

-- TODO: Rename and add documentation
function computeH(L)
  return (L + 1) * (L + 2) * (L + 3) / 6
end
