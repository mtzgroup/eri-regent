local dir = arg[1]
if dir == nil then
  print("Usage: regent " .. arg[0] .. " [library directory]")
  return
end

import "regent"
require "coulomb"
local header = dir .. "/coulomb_tasks.h"
local lib = dir .. "/libcoulomb_tasks.so"
regentlib.save_tasks(header, lib)
print("Generated header at "..header.." and library at "..lib)
