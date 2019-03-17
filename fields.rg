import "regent"

fspace HermiteGaussian {
  {x, y, z} : double;  -- Location of Gaussian
  eta       : double;  -- Exponent of Gaussian
  L         : int1d;   -- Angular momentum
  data_rect : rect1d;  -- Gives a range of indices where the number of values
                       -- is given by (L + 1) * (L + 2) * (L + 3) / 6
                       -- If `HermiteGaussian` is interpreted as a "bra", then
                       --`data_rect` refers to the J values. Otherwise, it
                       -- refers to the density matrix values.
  bound     : double;  -- TODO
}

fspace Double {
  value : double;
}
