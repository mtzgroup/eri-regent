-- Returns the maximum momentum determined at Regent compile time
local _max_momentum = nil
function getCompiledMaxMomentum()
  if _max_momentum == nil then
    for i, arg_value in ipairs(arg) do
      if arg[i] == "-L" then
        _max_momentum = StrToL[arg[i+1]]
      end
    end
    assert(_max_momentum ~= nil,
           "Must give max angular momentum `-L [S|P|D|F|G]`")
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
LPairToStr = {
  [0] = "SS", [1] = "SP", [2] = "PP",
  [3] = "PD", [4] = "DD", [5] = "DF",
  [6] = "FF", [7] = "FG", [8] = "GG"
}

function tetrahedral_number(n)
  return n * (n + 1) * (n + 2) / 6
end

function triangle_number(n)
  return n * (n + 1) / 2
end

-- Returns a list of lists where `pattern[i] = {N, L, M}`.
--
-- For example, `generateJFockSpinPattern(2)` returns the table
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
function generateJFockSpinPattern(level)
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

-- Returns a list of lists where `pattern[i] = {N, L, M}`.
--
-- sorted by increasing value of N+L+M
--
-- Remember that indices of lua lists start with one.
function generateJFockSpinPatternSorted(level)
  local pvec = {}
  for M = 0, level do -- inclusive
    for L = 0, level - M do -- inclusive
      for N = 0, level - M - L do -- inclusive
        table.insert(pvec, {N, L, M})
      end
    end
  end

  local pattern = {}
  for k = 0, level * level do -- inclusive
    for _, v in pairs(pvec) do
      if v[1] * v[1] + v[2] * v[2] + v[3] * v[3] == k then
        table.insert(pattern, v)
      end
    end
  end
  
  local pattern2 = {}
  for k = 0, level do -- inclusive
    for _, v in pairs(pattern) do
      if v[1] + v[2] + v[3] == k then
        table.insert(pattern2, v)
      end
    end
  end
  return pattern2
end

-- Returns a list of lists where `pattern[i] = {N, L, M}`.
--
-- For example, `generateJFockSpinPatternRestricted(2)` returns the table
-- 1 1 0
-- 1 0 1
-- 0 1 1
-- 2 0 0
-- 0 2 0
-- 0 0 2
-- Remember that indices of lua lists start with one.
function generateJFockSpinPatternRestricted(level)
  local pvec = {}
  for M = 0, level do -- inclusive
    for L = 0, level - M do -- inclusive
      for N = 0, level - M - L do -- inclusive
        table.insert(pvec, {N, L, M})
      end
    end
  end

  local pattern = {}
  for k = 0, level * level do -- inclusive
    for _, v in pairs(pvec) do
      if v[1] * v[1] + v[2] * v[2] + v[3] * v[3] == k and v[1]+v[2]+v[3] == level then
        table.insert(pattern, v)
      end
    end
  end
  return pattern
end

-- Returns a list of lists where `pattern[i] = {N, L, M}`.
--
-- For example, `generateKFockSpinPattern(2)` returns the table
-- 2 0 0
-- 1 1 0
-- 1 0 1
-- 0 2 0
-- 0 1 1
-- 0 0 2
-- Remember that indices of lua lists start with one.
function generateKFockSpinPattern(level)
  local pvec = {}
  for N = 0, level do -- inclusive
    for L = 0, level - N do -- inclusive
      for M = 0, level - N - L do -- inclusive
        table.insert(pvec, {N, L, M})
      end
    end
  end

  local pattern = {}
  local b = level + 1
  for k = b * b * b, 0, -1 do -- inclusive and backwards
    for _, v in pairs(pvec) do
      if v[1] + v[2] + v[3] == level then
        if v[1] * b * b + v[2] * b + v[3] == k then
          table.insert(pattern, v)
        end
      end
    end
  end
  return pattern
end
