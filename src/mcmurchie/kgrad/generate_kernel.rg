import "regent"

require "helper"
local c = regentlib.c

local function get_ket_count(L3, L4, k)
  local values = {
    {{  0},            -- SS (not used)
     {  0},            -- SP (not used)
     {  0}},           -- SD

    {{  0,   2,   4},  -- PS
     {  0,   9,  18},  -- PP
     {  0,  31,  62}}, -- PD

    {{  0,   4,   8,  12,  15,  18},  -- DS
     {  0,  18,  36,  54,  68,  82},  -- DP
     {  0,  56, 112, 168, 214, 260}}  -- DD
  }
  return values[L3+1][L4+1][k+1]
end


function generateKGradKernelStatements(R,L1,L2,L3,L4,k_idx,bra,ket,bra_EGP,ket_prevals,EGPmap,bra_idx,ket_idx,denik,denjl,output)
 
  local statements = terralib.newlist()

  local function getR(N, L, M)
    if R[N] == nil or R[N][L] == nil or R[N][L][M] == nil or R[N][L][M][0] == nil then
      return 0
    else
      return R[N][L][M][0]
    end
  end

  local H12, H34 = tetrahedral_number(L1+L2+1+1), tetrahedral_number(L3+L4+1)

  --statements:insert(rquote c.printf("bra_idx, ket.jshell_index: %d %d\n",bra_idx,ket.jshell_index) end)

  local patternL1 = generateJFockSpinPatternRestricted(L1)
  local patternL2 = generateJFockSpinPatternRestricted(L2)
  local patternL3 = generateJFockSpinPatternRestricted(L3)

  local pattern12 = generateJFockSpinPatternSorted(L1+L2+1)
  local pattern34 = generateJFockSpinPatternSorted(L3+L4)

  local bra_count = 0

  local gradAB = {}
  for A = 0,1 do
    gradAB[A] = {}
    for x = 0,2 do
      gradAB[A][x] = regentlib.newsymbol(double, "gradAB"..A..x)
      statements:insert(rquote
        var [gradAB[A][x]] = 0.0
      end)
    end
  end
  --local AB = {"A", "B"}
  --local xyz = {"x","y","z"}

  --statements:insert(rquote c.printf("setup gradient output\n") end)

  local k_min, k_max
  if (L1 > 0 and L2 > 0 and L3 > 0 and L4 >0) and (L1 + L2 + L3 + L4 >= 6) then
    k_min = k_idx
    k_max = k_idx
  else
    k_min = 0
    k_max = triangle_number(L3+1)-1
  end

  local Parr = {}
  for i = 0, triangle_number(L1+1)-1 do
    Parr[i] = {}
    for j = 0, triangle_number(L2+1)-1 do
      Parr[i][j] = {}
      local ket_count = -1 --get_ket_count(L3,L4,k_idx)-1
      
      local Ni, Li, Mi = unpack(patternL1[i+1])
      local Nj, Lj, Mj = unpack(patternL2[j+1])

      local x,y,z = { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }

      local Lfilt = { {Ni+Nj+x[1], Ni+Nj+x[2], Ni+Nj+x[3]}, {Ni+Nj+y[1], Ni+Nj+y[2], Ni+Nj+y[3]}, {Ni+Nj+z[1], Ni+Nj+z[2], Ni+Nj+z[3]} } 

      local Larr = {} -- check kgrad L setup. move outside of k loop
      for x = 0, L1+L2+1 do
        Larr[x] = {}
        for y = 0, L1+L2+1 do
          Larr[x][y] = {}
          for z = 0, L1+L2+1 do
            Larr[x][y][z] = regentlib.newsymbol(double, "Larr"..x..y..z)
            if (x + y +z) <= L1+L2+1 then
              statements:insert(rquote
                var [Larr[x][y][z]] = 0.0
              end)
            end
          end
        end
      end

      --statements:insert(rquote c.printf("setup Lfilt and Larr\n") end)

      --for k = 0, triangle_number(L3+1)-1 do
      for k = k_min, k_max do
        Parr[i][j][k] = {}
        ket_count = ket_count + 1

        -- setup P array
        for l=0, triangle_number(L4+1)-1 do
          Parr[i][j][k][l] = regentlib.newsymbol(double, "Parr"..i..j..k..l)
          statements:insert(rquote 
            var [Parr[i][j][k][l]] = [denik][i][k] * [denjl][j][l]
          end)
        end -- close l

        --statements:insert(rquote c.printf("populate Parr\n") end)

        local Nk, Lk, Mk = unpack(patternL3[k+1])
        local Dpattern = generateJFockSpinPatternRestricted(L4)
        for l = 0, triangle_number(L4+1)-1 do
          Dpattern[l+1][1] = Dpattern[l+1][1] + Nk
          Dpattern[l+1][2] = Dpattern[l+1][2] + Lk
          Dpattern[l+1][3] = Dpattern[l+1][3] + Mk
        end --close l

        local coeff = regentlib.newsymbol(double, "Coeff")
        statements:insert(rquote
          var [coeff] = 0.0
        end)

        for u = 0, H34-1 do
          statements:insert(rquote
            [coeff] = 0.0
          end)

          for t = 0, H12-1 do
            local Nt, Lt, Mt = unpack(pattern12[t+1])
            local Nu, Lu, Mu = unpack(pattern34[u+1])
            local N, L, M = Nt+Nu, Lt+Lu, Mt+Mu
            
            local den = {}
            if t == 0 then
              for l =0, triangle_number(L4+1)-1 do
                local X = Dpattern[l+1][1] - N
                local Y = Dpattern[l+1][2] - L
                local Z = Dpattern[l+1][3] - M
                if X >= 0 and Y >= 0 and Z >= 0 then
                  table.insert(den, l)
                end
              end -- close l
             
              --statements:insert(rquote c.printf("setup den\n") end)

              if table.getn(den) == 0 then
                break
              end

              for l = 0, table.getn(den)-1 do
                local idx = den[l+1]
                if L3 + L4 == 0 then
                  statements:insert(rquote 
                    [coeff] += [Parr[i][j][k][idx]] 
                  end)
                else
                  statements:insert(rquote
                    [coeff] += [Parr[i][j][k][idx]] * ket_prevals[{ket_idx,ket_count}]
                  end)
                end

                ket_count = ket_count + 1 
              end -- close l
               
              --statements:insert(rquote c.printf("add Parr contribs\n") end)

              if N+L+M == L3+L4 then ket_count = ket_count -1 end
            end -- close if t == 0

            --statements:insert(rquote c.printf("close t=0 loop\n") end)

            if (Nt <= Lfilt[1][1] and Lt <= Lfilt[1][2] and Mt <= Lfilt[1][3]) or
               (Nt <= Lfilt[2][1] and Lt <= Lfilt[2][2] and Mt <= Lfilt[2][3]) or
               (Nt <= Lfilt[3][1] and Lt <= Lfilt[3][2] and Mt <= Lfilt[3][3]) then
              if (Nu+Lu+Mu)%2 == 0 then
                    -- L[][][] += R[][][]0*coeff(Nt,Lt,Mt,N,L,M)
                statements:insert(rquote
                  [Larr[Nt][Lt][Mt]] += [getR(N,L,M)] * [coeff]
                end)
              else
                    -- L[][][] -= R[][][]0*coeff(Nt,Lt,Mt,N,L,M)
                statements:insert(rquote
                  [Larr[Nt][Lt][Mt]] -= [getR(N,L,M)] * [coeff]
                end)
              end -- close inner if
            end -- close outer if
          end -- close t
        end -- close u
      end -- close k

      --statements:insert(rquote c.printf("finsih Larr\n") end)

      for xi = 0, 2 do
        for cent = 0, 1 do
          for t = 0, H12-1 do
            local Nt, Lt, Mt = unpack(pattern12[t+1])
            if ( Nt <= Lfilt[xi+1][1] and Lt <= Lfilt[xi+1][2] and Mt <= Lfilt[xi+1][3]) then
              --statements:insert(rquote c.printf("enter EGP_idx loop\n") end)
              local EGP_idx = regentlib.newsymbol(int, "EGP_idx")
              statements:insert(rquote
                var [EGP_idx] = EGPmap[bra_count].stride 
              end)
              --statements:insert(rquote c.printf("delcare EGP_idx \n") end)

              local EGP_sign = regentlib.newsymbol(int, "EGP_sign")
              statements:insert(rquote
                var [EGP_sign] = EGPmap[bra_count].sign
              end)

              --statements:insert(rquote c.printf("EGPmap[%d].stride: %d EGPmap[%d].sign: %d\n",bra_count,[EGP_idx],bra_count,[EGP_sign]) end)

              statements:insert(rquote
                [output][xi*2+cent] += [EGP_sign] * [Larr[Nt][Lt][Mt]] * bra_EGP[{bra_idx,[EGP_idx]}]
              end)

              bra_count = bra_count + 1
            end
          end
        end
      end
    end
  end
  
  --for xi = 0,2 do
  --  for cent = 0,1 do
  --    statements:insert(rquote
  --      [output][cent*2+xi] += [gradAB[cent][xi]]
  --    end)
  --  end
  --end

  return statements
end
