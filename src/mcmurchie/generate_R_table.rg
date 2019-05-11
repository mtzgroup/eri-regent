import "regent"

local sqrt = regentlib.sqrt(double)
local round = regentlib.round(double)
local exp = regentlib.exp(double)
local SQRT_PI = math.sqrt(math.pi)

-- Generates statements that compute `length` auxiliary values.
-- R000j = (-2*alpha)^j * F_j(t)
-- where F_j(t) is the Boys function.
function generateStatementsComputeR000(R000, length, t, alpha, lambda, r_boys)
  assert(length <= 16, "Only accurate for j <= 16")


  local function taylorExpansion()
    local factor = regentlib.newsymbol(double, "factor")
    local t_idx = regentlib.newsymbol(int, "t_idx")
    local t_est = regentlib.newsymbol(double, "t_est")
    local statements = terralib.newlist({rquote
      var [factor] = 1.0
      var [t_idx] = round(10.0 * t)
      var [t_est] = t_idx / 10.0
      [R000[length-1]] = r_boys[{t_idx, length-1}]
    end})
    for k = 1, 6 do -- inclusive
      statements:insert(rquote
        var boys_est : double = r_boys[{t_idx, length-1+k}];
        factor *= (t_est - t) / k;
        [R000[length-1]] += factor * boys_est
      end)
    end
    return statements
  end


  local function downwardsRecursion()
    local statements = terralib.newlist()
    for j = length-2, 0, -1 do -- inclusive
      statements:insert(rquote
        [R000[j]] = (2.0 * t * [R000[j+1]] + exp(-t)) / (2.0 * j + 1.0)
      end)
    end
    return statements
  end


  local function upwardsRecursion()
    local statements = terralib.newlist()
    for j = 0, length-2 do -- inclusive
      statements:insert(rquote
        [R000[j+1]] = ((2.0 * j + 1.0) * [R000[j]] - exp(-t)) / (2.0 * t)
      end)
    end
    return statements
  end


  local statements = terralib.newlist()
  for j = 0, length-1 do -- inclusive
    statements:insert(rquote var [R000[j]] end)
  end
  statements:insert(rquote
    if t < 12 then
      [taylorExpansion()];
      [downwardsRecursion()];
    else
      [R000[0]] = SQRT_PI / (2.0 * sqrt(t))
      var g : double
      -- if t < 15 then
      g = 0.4999489092 - 0.2473631686 / t + 0.321180909 / (t * t)
                                          - 0.3811559346 / (t * t * t);
      -- elseif t < 18 then
      --   g = 0.4998436875 - 0.24249438 / t + 0.24642845 / (t * t);
      -- elseif t < 24 then
      --   g = 0.499093162 - 0.2152832 / t;
      -- elseif t < 30 then
      --   g = 0.490
      -- end

      -- if t < 30 then
      [R000[0]] -= exp(-t) * g / t;
      -- end

      [upwardsRecursion()];
    end
  end)

  local factor = regentlib.newsymbol(double, "factor")
  statements:insert(rquote
    var [factor] = lambda; 
    [R000[0]] *= factor
  end)
  for j = 1, length-1 do -- inclusive
    statements:insert(rquote
      factor *= -2.0 * alpha;
      [R000[j]] *= factor
    end)
  end
  return statements
end


-- Generates statements that compute Hermite polynomials given by
-- R00MJ = c * R00(M-1)(J+1) + (M-1) * R00(M-2)(J+1)
-- R0LMJ = b * R0(L-1)M(J+1) + (L-1) * R0(L-2)M(J+1)
-- RNLMJ = a * R(N-1)LM(J+1) + (N-1) * R(N-2)LM(J+1)
function generateStatementsComputeRTable(R, length, t, alpha, lambda, r_boys, a, b, c)
  local statements = generateStatementsComputeR000(R[0][0][0], length, t, alpha, lambda, r_boys)

  for N = 0, length-1 do -- inclusive
    for L = 0, length-1-N do -- inclusive
      for M = 0, length-1-N-L do -- inclusive
        for j = 0, length-1-N-L-M do -- inclusive
          if N == 0 and L == 0 and M == 0 then
            -- Do nothing. R000j has already been computed.
          elseif N == 0 and L == 0 and M == 1 then
            statements:insert(rquote
              var [R[0][0][1][j]] = c * [R[0][0][0][j+1]]
            end)
          elseif N == 0 and L == 0 then
            statements:insert(rquote
              -- TODO: These small integer multiplications can be replaced
              --       with repeated additions
              var [R[0][0][M][j]] = c * [R[0][0][M-1][j+1]] + (M-1) * [R[0][0][M-2][j+1]]
            end)
          elseif N == 0 and L == 1 then
            statements:insert(rquote
              var [R[0][1][M][j]] = b * [R[0][0][M][j+1]]
            end)
          elseif N == 0 then
            statements:insert(rquote
              var [R[0][L][M][j]] = b * [R[0][L-1][M][j+1]] + (L-1) * [R[0][L-2][M][j+1]]
            end)
          elseif N == 1 then
            statements:insert(rquote
              var [R[1][L][M][j]] = a * [R[0][L][M][j+1]]
            end)
          else
            statements:insert(rquote
              var [R[N][L][M][j]] = a * [R[N-1][L][M][j+1]] + (N-1) * [R[N-2][L][M][j+1]]
            end)
          end
        end
      end
    end
  end

  return statements
end
