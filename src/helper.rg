-- Returns the maximum momentum determined at Regent compile time
local _max_momentum = nil
function getCompiledMaxMomentum()
  if _max_momentum == nil then
    for i, arg_value in ipairs(arg) do
      if arg[i] == "-L" then
        _max_momentum = StrToL[arg[i+1]]
      end
    end
    assert(_max_momentum ~= nil, "Must give max angular momentum `-L [S|P|D|F|G]`")
  end
  return _max_momentum
end

-- The string representation of a given angular momentum
LToStr = {
  [0] = "S", [1] = "P", [2] = "D", [3] = "F", [4] = "G"
}
StrToL = {
  ["S"] = 0, ["P"] = 1, ["D"] = 2, ["F"] = 3, ["G"] = 4
}

-- Return the number of atomic orbital functions in shells of momentum 0 to L
-- TODO: Find a better name
function computeH(L12)
  return (L12 + 1) * (L12 + 2) * (L12 + 3) / 6
end

-- Returns a list of lists where `pattern[i] = {N, L, M}`.
--
-- For example, `generateSpinPattern(2)` returns the table
-- 0 0 0
-- 1 0 0
-- 0 1 0
-- 0 0 1
-- 1 1 0
-- 1 0 1
-- 0 1 1
-- 2 0 0
-- 0 2 0
-- 0 0 2
-- Remember that indices of lua lists start with one.
function generateSpinPattern(level)
  local pvec = {}
  for M = 0, level do -- inclusive
    for L = 0, level - M do -- inclusive
      for N = 0, level - M - L do -- inclusive
        table.insert(pvec, {N, L, M})
      end
    end
  end

  -- TODO: This can be done with a stable sort
  local pattern = {}
  for k = 0, level * level do -- inclusive
    for _, v in pairs(pvec) do
      if v[1] * v[1] + v[2] * v[2] + v[3] * v[3] == k then
        table.insert(pattern, v)
      end
    end
  end
  return pattern
end
