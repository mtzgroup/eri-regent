local MAX_KFOCK=tonumber(os.getenv('MAX_KFOCK'))

local registration_thunks = terralib.newlist()
local link_flags = terralib.newlist()
link_flags:insert("-L"..os.getenv("LEGION_INSTALL_PATH").."/lib")
link_flags:insert("-L"..os.getenv("LEGION_INSTALL_PATH").."/lib64")
link_flags:insert("-L.")
link_flags:insert("-L./mcmurchie/kfock")
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
            local header = terralib.includec("kfock"..L1..L2..L3..L4..k..".h")
            local thunk = header["kfock"..L1..L2..L3..L4..k.."_h_register"]
            registration_thunks:insert(quote thunk() end)
            link_flags:insert("-lkfock"..L1..L2..L3..L4..k)
          end
        end
      end
    end
  end
end
local toplevel_thunk
do
  local header = terralib.includec("topkfock.h")
  toplevel_thunk = header["topkfock_h_register"]
  link_flags:insert("-ltopkfock")
end
link_flags:insert("-lregent")
link_flags:insert("-llegion")
link_flags:insert("-lrealm")
link_flags:insert("-lm")

local legion = terralib.includec("legion.h")

terra main(argc : int, argv: &rawstring)
  [registration_thunks];
  toplevel_thunk(argc, argv)
end
terralib.saveobj("main", "executable", {main=main}, link_flags)
