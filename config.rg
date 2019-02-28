import "regent"

local c = regentlib.c
local cstring = terralib.includec("string.h")

struct Config {
  num_gausses        : int;       -- Number of Hermite Gaussians
  num_bra_kets       : int;       -- Number of brakets
  num_data_values    : int;       -- Number of J values and size of density matrix
  highest_L          : int;       -- Highest angular momentum over all Gaussians
  parallelism        : int;
  verbose            : bool;
  input_filename     : int8[512];
  output_filename    : int8[512];
}

terra print_usage_and_abort()
  c.printf("Usage: regent coulomb.rg [OPTIONS]\n")
  c.printf("OPTIONS\n")
  c.printf("  -h              : Print the usage and exit.\n")
  c.printf("  -i {input.dat}  : Use {input.dat} as input data.\n")
  c.printf("  -o {output.dat} : Use {output.dat} as output data.\n")
  c.printf("  -p {value}      : Set the number of parallel tasks to {value}.\n")
  c.printf("  -v              : Verbose printing.\n")
  c.exit(0)
end

terra Config:initialize_from_command()
  self.parallelism = 1
  self.verbose = false
  self.input_filename[0] = 0
  self.output_filename[0] = 0

  var args = c.legion_runtime_get_input_args()
  var i = 1
  while i < args.argc do
    if cstring.strcmp(args.argv[i], "-h") == 0 then
      print_usage_and_abort()
    elseif cstring.strcmp(args.argv[i], "-i") == 0 then
      if self.input_filename[0] ~= 0 then
        c.printf("Error: Only accepts one input file!\n")
        c.abort()
      end
      i = i + 1
      var file = c.fopen(args.argv[i], "r")
      if file == nil then
        c.printf("File '%s' does not exist!\n", args.argv[i])
        c.abort()
      end
      cstring.strncpy(self.input_filename, args.argv[i], 512)
      -- The first line of the file is the number of HermiteGaussians
      c.fscanf(file, "%d", &self.num_gausses)
      var line : int8[512]
      var L : int
      self.num_data_values = 0
      self.highest_L = 0
      c.fgets(line, 512, file) -- Read blank line
      for i = 0, self.num_gausses do
        c.fgets(line, 512, file)
        c.sscanf(line, "%d", &L)
        if L > self.highest_L then
          self.highest_L = L
        end
        var H : int = (L + 1) * (L + 2) * (L + 3) / 6
        self.num_data_values = self.num_data_values + H
      end
      c.fclose(file)
    elseif cstring.strcmp(args.argv[i], "-o") == 0 then
      if self.output_filename[0] ~= 0 then
        c.printf("Error: Only accepts one output file!\n")
        c.abort()
      end
      i = i + 1
      var file = c.fopen(args.argv[i], "r")
      if file ~= nil then
        c.printf("File '%s' already exists!\n", args.argv[i])
        c.abort()
      end
      cstring.strncpy(self.output_filename, args.argv[i], 512)
    elseif cstring.strcmp(args.argv[i], "-p") == 0 then
      i = i + 1
      self.parallelism = c.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-v") == 0 then
      self.verbose = true
    else
      print_usage_and_abort()
    end
    i = i + 1
  end

  if self.input_filename[0] == 0 then
    c.printf("Error: Input file not given!\n")
    c.abort()
  end

  -- self.num_bra_kets = self.num_gausses * (self.num_gausses - 1) / 2
  self.num_bra_kets = self.num_gausses * self.num_gausses
end

return Config
