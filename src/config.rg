import "regent"

local c = regentlib.c
local assert = regentlib.assert
local cstring = terralib.includec("string.h")

struct Config {
  num_gausses          : int;       -- Number of Hermite Gaussians
  num_data_values      : int;       -- Number of J values and size of density matrix
  highest_L            : int;       -- Highest angular momentum over all Gaussians
  input_filename       : int8[512];
  output_filename      : int8[512];
  true_values_filename : int8[512];
  num_trials           : int;
  verbose              : bool;
}

terra print_usage_and_abort()
  c.printf("Usage: regent coulomb.rg [OPTIONS]\n")
  c.printf("OPTIONS\n")
  c.printf("  -h                   : Print the usage and exit.\n")
  c.printf("  -i {input.dat}       : Use {input.dat} as input data.\n")
  c.printf("  -o {output.dat}      : Use {output.dat} as output data.\n")
  c.printf("  -v {true_output.dat} : Use {true_output.dat} to check results.\n")
  c.printf("  --trials {value}     : Run {value} times.\n")
  c.printf("  --verbose            : Verbose printing.\n")
  c.exit(0)
end

terra Config:initialize_from_command()
  self.input_filename[0] = 0
  self.output_filename[0] = 0
  self.true_values_filename[0] = 0
  self.num_trials = 1
  self.verbose = false

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
      c.fscanf(file, "%d\n", &self.num_gausses)
      var line : int8[512]
      var L : int
      self.num_data_values = 0
      self.highest_L = 0
      for j = 0, self.num_gausses do
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
    elseif cstring.strcmp(args.argv[i], "-v") == 0 then
      if self.true_values_filename[0] ~= 0 then
        c.printf("Error: Only accepts one true values file!\n")
        c.abort()
      end
      i = i + 1
      var file = c.fopen(args.argv[i], "r")
      if file == nil then
        c.printf("File '%s' does not exist!\n", args.argv[i])
        c.abort()
      end
      cstring.strncpy(self.true_values_filename, args.argv[i], 512)
    elseif cstring.strcmp(args.argv[i], "--trials") == 0 then
      i = i + 1
      self.num_trials = c.atoi(args.argv[i])
      assert(self.num_trials > 0, "Number of trials must be positive.")
    elseif cstring.strcmp(args.argv[i], "--verbose") == 0 then
      self.verbose = true
    else
      c.printf("Warning: Unknown option %s\n", args.argv[i])
    end
    i = i + 1
  end

  if self.input_filename[0] == 0 then
    c.printf("Error: Input file not given!\n")
    c.abort()
  end
end

return Config
