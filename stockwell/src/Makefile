tc = np.arange(0.,60.,1./40.)
spec, scales = get_cwt(c, tc)
freq = (1.0/wave.fourier_from_scales(scales, 'morlet', 8))
plt.figure(); plt.pcolormesh(tc, freq, abs(spec))