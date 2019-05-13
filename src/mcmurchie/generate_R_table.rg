import "regent"

local sqrt = regentlib.sqrt(double)
local round = regentlib.round(double)
local exp = regentlib.exp(double)
local SQRT_PI = math.sqrt(math.pi)

-- Generates statements that compute `length` Boys values
function generateStatementsComputeBoys(boys, length, t, r_boys)

  local factor = regentlib.newsymbol(double, "factor")
  local t_idx = regentlib.newsymbol(int, "t_idx")
  local t_est = regentlib.newsymbol(double, "t_est")
  local downwardsRecursion = terralib.newlist({rquote
    var [factor] = 1.0
    var [t_idx] = round(10.0 * t)
    var [t_est] = t_idx / 10.0
    [boys[length-1]] = r_boys[{t_idx, length-1}]
  end})
  for k = 1, 5 do -- inclusive
    downwardsRecursion:insert(rquote
      var boys_est : double = r_boys[{t_idx, length-1+k}];
      factor *= (t_est - t) / k;
      [boys[length-1]] += factor * boys_est
    end)
  end
  for j = length-2, 0, -1 do -- inclusive
    downwardsRecursion:insert(rquote
      [boys[j]] = (2.0 * t * [boys[j+1]] + exp(-t)) / (2 * j + 1)
    end)
  end

  local upwardsRecursionAsymptotic = terralib.newlist({rquote
    [boys[0]] = SQRT_PI / (2 * sqrt(t)); -- TODO: Use inverse sqrt
  end})
  for j = 0, length-2 do -- inclusive
    upwardsRecursionAsymptotic:insert(rquote
      [boys[j+1]] = (2 * j + 1) * [boys[j]] / (2.0 * t)
    end)
  end

  local statements = terralib.newlist()
  for j = 0, length-1 do -- inclusive
    statements:insert(rquote var [boys[j]] end)
  end
  statements:insert(rquote
    if t < 25 then
      [downwardsRecursion]
    else
      [upwardsRecursionAsymptotic]
    end
  end)
  return statements
end


-- Generates statements that compute Hermite polynomials given by
-- R000J = lambda * (-2*alpha)^J * F_J(t)
-- R00MJ = c * R00(M-1)(J+1) + (M-1) * R00(M-2)(J+1)
-- R0LMJ = b * R0(L-1)M(J+1) + (L-1) * R0(L-2)M(J+1)
-- RNLMJ = a * R(N-1)LM(J+1) + (N-1) * R(N-2)LM(J+1)
-- where F_J(t) is the Boys function.
function generateStatementsComputeRTable(R, length, t, alpha, lambda, r_boys, a, b, c)
  local statements = generateStatementsComputeBoys(R[0][0][0], length, t, r_boys)

  for j = 0, length-1 do -- inclusive
    statements:insert(rquote
      [R[0][0][0][j]] *= lambda
      lambda *= -2.0 * alpha;
    end)
  end

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
