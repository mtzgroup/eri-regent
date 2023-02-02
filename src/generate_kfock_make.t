-- TODO: read in from args or env variable
local MAX_KFOCK=0

local kfock_sos = terralib.newlist()
local kfock_hs = terralib.newlist()
local lines = terralib.newlist()

for L1 = 0, MAX_KFOCK do
  for L2 = 0, MAX_KFOCK do
    for L3 = 0, MAX_KFOCK do
      for L4 = 0, MAX_KFOCK do
        for k = 0, 0 do
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

print("KFOCK_SO := " .. kfock_sos:concat(" "))
print("KFOCK_H := " .. kfock_hs:concat(" "))
for _, line in ipairs(lines) do
  print(line)
end
