import "regent"

require "fields"

function jfock_post_process(L12)
  local H12 = computeH(L12)
  local task post_process(r_kernel_output : region(ispace(int1d), double[H12]),
                          r_output        : region(ispace(int1d), double))
  where
    reads(r_kernel_output),
    reduces +(r_output)
  do

    -- TODO

  end
  post_process:set_name("JFockPostProcess"..L12)
  return post_process
end
