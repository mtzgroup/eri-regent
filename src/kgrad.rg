import "regent"

require "helper"
require "mcmurchie.kgrad.generate_kgrad_integral"

local c = regentlib.c

function kgrad(r_bras_list, r_braEGP_list, r_kets_list, r_ketprevals_list, r_ketlabels_list, r_denik_list, r_denjl_list, r_output_list, r_braEGPmap_list, r_gamma_table, threshold, parallelism, largest_momentum)
  local statements = terralib.newlist()
  local L_start, L_end, L_stride = 0, getCompiledMaxMomentum(), 1

  for L1 = L_start, L_end, L_stride do
    for L2 = L_start, L_end, L_stride do
      for L3 = L_start, L_end, L_stride do
        for L4 = L_start, L_end, L_stride do
          if L2 >= L1 then
            
            c.printf("L1234 : %1.f %1.f %1.f %1.f\n", L1, L2, L3, L4)
            local r_bras = r_bras_list[L1][L2]
            local r_kets = r_kets_list[L3][L4]
            local r_bra_EGP = r_braEGP_list[L1][L2] --[1]
            local r_ket_prevals = r_ketprevals_list[L3][L4] 
            local r_denik, r_denjl
            local r_output = r_output_list[L1][L2]
            local r_EGPmap = r_braEGPmap_list[L1][L2]

            if L1<=L3 then
              r_denik = r_denik_list[L1][L3]
            else
              r_denik = r_denik_list[L3][L1]
            end

            if L2<=L4 then
              r_denjl = r_denjl_list[L2][L4]
            else
              r_denjl = r_denjl_list[L4][L2]
            end

            local p = parallelism

            local k_max = 0
            if (L1>0 and L2>0 and L3>0 and L4>0) and (L1+L2+L3+L4 >= 6) then
              k_max = triangle_number(L3+1)-1
            end
            for k = 0, k_max do
              local kgrad_integral = generateTaskMcMurchieKGradIntegral(L1,L2,L3,L4,k)
              if r_bras ~= nil and r_kets ~= nil then
                statements:insert(rquote

                  for i =0, p do
                    kgrad_integral(r_bras, r_kets, r_bra_EGP, r_ket_prevals, r_denik, r_denjl, r_output, r_gamma_table, threshold, largest_momentum, r_EGPmap)
                  end
                end)
              end
            end
          
          end
        end
      end
    end
  end
  return statements
end
                

