-- Returns a list of lists where `pattern[i] = {N, L, M}`.
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
