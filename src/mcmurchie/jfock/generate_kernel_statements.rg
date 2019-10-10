import "regent"

require "helper"

-- Returns a list of regent statements that implements the McMurchie algorithm
function generateJFockKernelStatements(R, L12, L34, density, accumulator)
  local H12, H34 = computeH(L12), computeH(L34)
  local statements = terralib.newlist()
  local results = {}
  for i = 0, H12-1 do -- inclusive
    results[i] = regentlib.newsymbol(double, "result"..i)
    statements:insert(rquote var [results[i]] = 0.0 end)
  end

  local pattern12 = generateSpinPattern(L12)
  local pattern34 = generateSpinPattern(L34)
  for u = 0, H34-1 do -- inclusive
    for t = 0, H12-1 do -- inclusive
      local Nt, Lt, Mt = unpack(pattern12[t+1])
      local Nu, Lu, Mu = unpack(pattern34[u+1])
      local N, L, M = Nt + Nu, Lt + Lu, Mt + Mu
      if (Nu + Lu + Mu) % 2 == 0 then
        statements:insert(rquote
          [results[t]] += density[u] * [R[N][L][M][0]]
        end)
      else
        statements:insert(rquote
          [results[t]] -= density[u] * [R[N][L][M][0]]
        end)
      end
    end
  end
  for i = 0, H12-1 do -- inclusive
    statements:insert(rquote
      accumulator[i] += [results[i]]
    end)
  end
  return statements
end
