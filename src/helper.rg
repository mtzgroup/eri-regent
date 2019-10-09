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