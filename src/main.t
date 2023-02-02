-- TODO: read in from args or env variable
local MAX_KFOCK=0

local registration_thunks = terralib.newlist()
local link_flags = terralib.newlist()
link_flags:insert("-L"..os.getenv("LEGION_INSTALL_PATH").."/lib")
link_flags:insert("-L"..os.getenv("LEGION_INSTALL_PATH").."/lib64")
link_flags:insert("-L.")
link_flags:insert("-L./mcmurchie/kfock")
do
  local header = terralib.includec("topkfock.h")
  local thunk = header["topkfock_h_register"]
  registration_thunks:insert(quote thunk() end)
  link_flags:insert("-ltopkfock")
end
for L1 = 0, MAX_KFOCK do
  for L2 = 0, MAX_KFOCK do
    for L3 = 0, MAX_KFOCK do
      for L4 = 0, MAX_KFOCK do
        for k = 0, 0 do
          local header = terralib.includec("kfock"..L1..L2..L3..L4..k..".h")
          local thunk = header["kfock"..L1..L2..L3..L4..k.."_h_register"]
          registration_thunks:insert(quote thunk() end)
          link_flags:insert("-lkfock"..L1..L2..L3..L4..k)
        end
      end
    end
  end
end
link_flags:insert("-lregent")
link_flags:insert("-llegion")
link_flags:insert("-lrealm")
link_flags:insert("-lm")

local legion = terralib.includec("legion.h")

terra main(argc : int, argv: &rawstring)
  [registration_thunks];
  legion.legion_runtime_start(argc, argv, false)
end
terralib.saveobj("main", "executable", {main=main}, link_flags)
