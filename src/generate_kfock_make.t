-- TODO: read in from args or env variable
local MAX_KFOCK=1

local kfock_sos = terralib.newlist()
local kfock_hs = terralib.newlist()
local lines = terralib.newlist()

local START = MAX_KFOCK
local END = 0
local stride = -1
for L1 = START, END, stride do
  for L2 = START, END, stride do
    for L3 = START, END, stride do
      for L4 = START, END, stride do
        if L1 < L3 or (L1 == L3 and L2 <= L4) then 
          local k_max = 0
          if (L1 + L2 > 1) and (L1 + L2 + L3 + L4 >= 5) then
            k_max = (L3+1)*(L3+2)/2 - 1  -- triangle number minus one
          end
          for k = 0, k_max do
            local kfock_so = "mcmurchie/kfock/libkfock"..L1..L2..L3..L4..k..".so"
            local kfock_h = "mcmurchie/kfock/kfock"..L1..L2..L3..L4..k..".h"
            kfock_sos:insert(kfock_so)
            kfock_hs:insert(kfock_h)
            lines:insert(kfock_so..": mcmurchie/kfock/generate_kfock_integral_def.rg")
            lines:insert("\t$(REGENT) $< $(REGENT_FLAGS) "..L1.." "..L2.." "..L3.." "..L4.." "..k)
          end
        end
      end
    end
  end
end

print("KFOCK_SO := " .. kfock_sos:concat(" "))
print("KFOCK_H := " .. kfock_hs:concat(" "))
for _, line in ipairs(lines) do
  print(line)
end
