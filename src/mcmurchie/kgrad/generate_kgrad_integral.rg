import "regent"

require "fields"
require "helper"
require "mcmurchie.kgrad.generate_kernel"
require "mcmurchie.generate_R_table"

local rsqrt = regentlib.rsqrt(double)

local c = regentlib.c

local _kgrad_integral_cache = {}
function generateTaskMcMurchieKGradIntegral(L1,L2,L3,L4,k_idx)
  local NL1 = triangle_number(L1 + 1)
  local NL3 = triangle_number(L3 + 1)
  local Nout = NL1 * NL3
  local L_string = LToStr[L1]..LToStr[L2]..LToStr[L3]..LToStr[L4]..k_idx
  if _kgrad_integral_cache[L_string] ~= nil then
    return _kgrad_integral_cache[L_string]
  end

  c.printf("starting kgrad fxn\n")

  local L12, L34 = L1 + L2, L3 + L4
  -- Create a table of Regent variables to hold Hermite polynomials.
  local R = {}
  for N = 0, L12+L34+1 do -- inclusive
    R[N] = {}
    for L = 0, L12+L34-N+1 do -- inclusive
      R[N][L] = {}
      for M = 0, L12+L34-N-L+1 do -- inclusive
        R[N][L][M] = {}
        for j = 0, L12+L34-N-L-M+1 do -- inclusive
          R[N][L][M][j] = regentlib.newsymbol(double, "R"..N..L..M..j)
        end
      end
    end
  end

  c.printf("done with R\n")

  local
  __demand(__leaf) 
  --__demand(__cuda) -- NOTE: comment out if printing from kernels (debugging) 
  task kgrad_integral(r_bras           : region(ispace(int1d), getKGradBra(L1+L2)), -- using kgrad parser 
                      r_kets           : region(ispace(int1d), getKFockPair(L3, L4)),
                      r_bra_EGP    : region(ispace(int2d), double),
                      r_ket_prevals    : region(ispace(int2d), double),
                      r_denik       : region(ispace(int2d), getKFockDensity(L1, L3)),
                      r_denjl       : region(ispace(int2d), getKFockDensity(L2,L4)),
                      r_output         : region(ispace(int1d), getKGradOutput(L1, L2)),
                      r_gamma_table    : region(ispace(int2d), double[5]),
                      threshold        : float, 
                      largest_momentum : int,
                      r_EGPmap         : region(ispace(int1d), getKGradBraEGPMap(L1,L2)) )
  where
    reads(r_bras, r_kets, r_bra_EGP, r_ket_prevals,r_denik, r_denjl, r_gamma_table,r_EGPmap),
    reads writes(r_output.values)
  do
    var ket_idx_bounds_lo : int = r_kets.ispace.bounds.lo
    var ket_idx_bounds_hi : int = r_kets.ispace.bounds.hi
    for bra_idx in r_bras.ispace do
      for ket_idx = ket_idx_bounds_lo, ket_idx_bounds_hi+1 do
        var bra = r_bras[bra_idx]
        var ket = r_kets[ket_idx]
        var denik : getKFockDensity(L1,L3)
        var denjl : getKFockDensity(L2,L4)
        if L2 <= L4 then
          denjl = r_denjl[{bra.jshell_index,ket.jshell_index}]
        else
          denjl = r_denjl[{ket.jshell_index, bra.jshell_index}]
        end
        
        if L1 <= L3 then
          denik = r_denik[{bra.ishell_index,ket.ishell_index}]
        else
          denik = r_denik[{ket.ishell_index,bra.ishell_index}]
        end

        --if bra.bound * ket.bound > threshold then end

        var a = bra.location.x - ket.location.x
        var b = bra.location.y - ket.location.y
        var d = bra.location.z - ket.location.z --change from c for debugging 

        var factor = -1.0
        if bra.ishell_index == bra.jshell_index then
          factor = -0.5
        end
        var lambda = factor * bra.C * ket.C *rsqrt(bra.eta+ket.eta)
        var alpha = bra.eta * ket.eta * (1.0/(bra.eta+ket.eta))
        var t = alpha * (a*a + b*b + d*d)
        --c.printf("alpha, lambda, t, a, b, c: %lf %lf %lf %lf %lf %lf\n",alpha,lambda,t,a,b,d)
        --c.printf("r_EGPmap[1].sign = %d\n",r_EGPmap[1].sign)
        ;[generateStatementsComputeRTable(R,L1+L2+L3+L4+1,t,alpha,lambda,a,b,d,r_gamma_table)]
        --c.printf("compute R table done\n")
        ;[generateKGradKernelStatements(R,L1,L2,L3,L4,k_idx,bra,ket,r_bra_EGP,r_ket_prevals,r_EGPmap,bra_idx,ket_idx, rexpr denik.values end, rexpr denjl.values end, rexpr r_output[bra_idx].values end)]
        --c.printf("somehow clear kernel statmenets\n")
        --;[generateKGradKernelStatements(R,L1,L2,L3,L4,k_idx,bra,ket,r_bra_EGP,r_ket_prevals,r_EGPmap,bra_idx,ket_idx, rexpr denik.values end, rexpr denjl.values end, rexpr r_output[{N24, bra_idx, ket.ishell_index}].values end)] 
        --c.printf("somehow clear kernel statements\n")


      end
    end
  end
  kgrad_integral:set_name("KGradMcMurchie"..L_string)
  _kgrad_integral_cache[L_string] = kgrad_integral
  return kgrad_integral
end

