"""
For a located large event, with R, T, Z component seismograms:

With the scalar azimuth:

1. Calculate and plot the NIP filter
1. Plot the Sr and filtered Sr
1. Plot the St and filtered St
1. Plot the original R, T, Z under the NIP filtered R and T

With the instantaneous azimuth estimates

1. Calculate and plot the NIP filter
1. Plot the instantaneous azimuths
1. Plot the Sr and filtered Sr
1. Plot the St and filtered St
1. Plot the original R, T, Z under the NIP filtered R and T


For timings, run:

```python
python -m cProfile test_filter.py
```


"""
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.basemap import Basemap
import numpy as np

from obspy import read
from obspy.signal import rotate_RT_NE
from obspy.taup import taup

from distaz import distaz
from stockwell import stransform, istransform
import stockwell.filter as filt
import stockwell.plotting as splt

LANDSCAPE = (11,8.5)
PORTRAIT = (8.5,11)
POWERPOINT = (10, 7.5)
SCREEN = (31, 19)
HALFSCREEN = (SCREEN[0]/2.0, SCREEN[1])

# save myself some RAM for plotting big arrays
plt.ioff()

######################### CALCULATIONS ##################################
# TODO: put the functions above into a stockwell.plotting module

#st = read("tests/data/SampleWaveforms/E2010-01-10-00-27-39/Dsp/aak-ii-00*")
#st = read("tests/data/SampleWaveforms/E2010-01-10-00-27-39/Dsp/anmo*")
#st = read("tests/data/SampleWaveforms/E2010-01-10-00-27-39/Dsp/mdj*")
#st = read('/wave/seismic2/user_dirs/hans/Mines/Kazakh_Net/Discrim_Study/Waves/Single_Charge_KTS/KURK/BH/1998/19980814074411.KURK.II.BH*', format='SAC')
st = read('/wave/seismic2/user_dirs/hans/Mines/Kazakh_Net/Discrim_Study/Waves/Single_Charge_KTS/KURK/BH/1998/19980815024059.KURK.KZ.BH*', format='SAC')



# filter band
#fmin = 1./80
fmin= 1.0/80
#fmin= None

fmax = 1./30
#fmax = 1.0/5
#fmax = None


tr = st[0]
fs = tr.stats.sampling_rate
sac = tr.stats.sac
deg, km, az, baz = distaz(sac.evla, sac.evlo, sac.stla, sac.stlo)
az_prop = baz + 180
if az < 0.0:
    az += 360.0
if baz < 0.0:
    baz += 360.0
if az_prop > 360:
    az_prop -= 360

# cut window and arrivals 
#vmax = 5.0 
vmax = 2.5
#vmin = 2.5
vmin = 1.5
swmin = km/vmax
swmax = km/vmin
tmin = tr.stats.starttime + swmin
tmax = tr.stats.starttime + swmax
tt = taup.getTravelTimes(deg, sac.evdp, model='ak135')

tarrivals = [(itt['phase_name'], itt['time']) for itt in tt]
swarrivals = [(str(swvel), km/swvel) for swvel in (5.0, 4.0, 3.0, 2.0)]
tarrivals.extend(swarrivals)

arrivals= []
for arr, itt in tarrivals:
    if arr in ('P', 'S') or (arr, itt) in swarrivals:
        arrivals.append((arr, itt))

t0 = 1000

st = st.trim(endtime=tmax)
if fmin and fmax:
    st.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True)
else:
    fmin, fmax = 0.0, fs/2.0

# surface wave filter specs
xpr = -int(np.sign(np.sin(np.radians(baz))))
polarization = 'retrograde'

# nomenclature:
#[nevrt][sd][f]
# n : north
# e : east
# v : vertical
# s : scalar rotation (great circle)
# d : dynamic rotation (NIP estimate)
# f : NIP filtered

rs = st.select(component='R')[0].data
ts = st.select(component='T')[0].data
v = st.select(component='Z')[0].data

Sv = stransform(v, fs)
Srs = stransform(rs, fs)
Sts = stransform(ts, fs)

n, e = rotate_RT_NE(rs, ts, baz)

Sn, T, F = stransform(n, fs, return_time_freq=True)
Se = stransform(e, fs)

# period
P = 1./F

# odd cases where shape doesn't match
nm, nn = Sv.shape
T = T[:nm, :nn]
F = F[:nm, :nn]
P = P[:nm, :nn]

# scalar NIP and filter
nips = filt.NIP(Srs, filt.shift_phase(Sv, polarization=polarization))
sfilt = filt.get_filter(nips, polarization=polarization, threshold=0.8)

rsf = istransform(Srs*sfilt, Fs=fs)
vsf = istransform(Sv*sfilt, Fs=fs)
#tsf = istransform(Sts*f, Fs=fs)


# instantaneous rotations
theta = filt.instantaneous_azimuth(Sv, Sn, Se, polarization, xpr)
Srd, Std = filt.rotate_NE_RT(Sn, Se, theta)

# dynamic NIP and filter
nipf = filt.NIP(Srd, filt.shift_phase(Sv, polarization=polarization))
dfilt = filt.get_filter(nipf, polarization=polarization, threshold=0.8)

Srd[np.isnan(Srd)] = 0.0
Std[np.isnan(Std)] = 0.0
rd = istransform(Srd, Fs=fs)
td = istransform(Std, Fs=fs)

dfilt[np.isnan(dfilt)] = 0.0
rdf = istransform(Srd*dfilt, Fs=fs)
vdf = istransform(Sv*dfilt, Fs=fs)
#tdf = istransform(Std*dfilt, Fs=fs)

# get universal min/max values
dmax = max([np.abs(data).max() for data in (n, e, v, rs, ts, rsf, vsf, rd, td, rdf, vdf)])
cmax = max([np.abs(data).max() for data in (Srs, Sts, Sv, Srd, Std)])

################################ map ###########################################
plt.figure()
m = Basemap()
m.fillcontinents(color='0.75',zorder=0)
m.drawcountries()
m.drawcoastlines(linewidth=0.3)
xevt, yevt = m(sac.evlo, sac.evla)
xsta, ysta = m(sac.stlo, sac.stla)
m.drawgreatcircle(sac.evlo, sac.evla, sac.stlo, sac.stla, color='k')
m.plot(xevt, yevt, 'r*', markersize=17)
m.plot(xsta, ysta, 'b^', markersize=17)
plt.title("$\Delta$ = {}, mag {}, az = {:.0f}, baz = {:.0f}".format(deg, sac.mag, baz-180, baz))

plt.savefig('map.png')
plt.close()

########## check dynamic vs inst. rotations #################
# plot estimated instantaneous azimuth
_ = splt.plot_instantaneous_azimuth(theta, fs, xlim=(t0, len(v)))
plt.title('Instantaneous azimuth. great circle azimuth = {:.1f}'.format(az_prop))
plt.savefig('instantaneous_azimuth.png', dpi=200)
plt.close()

# time-series rotation comparison plot (static vs dynamic), 6 panels
fig = plt.figure(figsize=SCREEN)
fig = splt.tile_comparison(T, F, Sv, Srs, Srd, Sts, Std, v, rs, rd, ts, td,
                           arrivals, flim=(fmax, fmax), clim=(0.0, cmax),
                           dlim=(-dmax, dmax), hatch=(theta-az_prop),
                           hatchlim=(-20, 20), fig=fig, xlim=(t0, len(v)))
plt.savefig('rotation_comparison.png', dpi=200)
plt.close()


######################## check scalar Rayleigh filters #########################
fig = plt.figure(figsize=HALFSCREEN)
fig = splt.check_filters(T, F, Sv, Srs, Sts, vsf, rsf, ts, arrivals,
                         flim=(fmin, fmax), clim=(0.0, cmax), dlim=(-dmax, dmax),
                         xlim=(t0, len(v), hatch=sfilt, hatchlim=(0.0, 0.8),
                         fig=fig)
plt.savefig('filter_rayleigh_scalar.png', dpi=200)
plt.close()



######################## check dynamic Rayleigh filters ########################
fig = plt.figure(figsize=HALFSCREEN)
fig = splt.check_filters(T, F, Sv, Srd, Std, vsf, rdf, td, arrivals,
                         flim=(fmin, fmax), clim=(0.0, cmax), dlim=(-dmax, dmax),
                         xlim=(t0, len(v), hatch=dfilt, hatchlim=(0.0, 0.8),
                         fig=fig)
plt.savefig('filter_rayleigh_dynamic.png', dpi=200)
plt.close()

# plot NIP and filter
print("Plotting NIP and filter, scalar az.")
splt.plot_NIP(T, F, nips, fs, flim=(0, fmax))
plt.title('NIP, scalar azimuth')
plt.savefig('nip_filter_scalar.png', dpi=200)
plt.close()

# plot Sr and filtered Sr
# plot_tile
# plt.savefig('stransforms_scalar.png', dpi=200)
# plt.close()

# plot original and filtered waves
print("Plotting filtered and raw waves.")
plt.figure(figsize=LANDSCAPE)
splt.compare_waveforms(v, vsf, rs, rsf, ts, arrivals)
plt.savefig('waves_scalar.png', dpi=200)
plt.close()

############################ gridspec plots

fig = plt.figure(figsize=SCREEN)
# nip? nipf?
fig = splt.NIP_filter_plots(T, F, nips, fs, Srs, Sts, Sv, rs, rsf, v, vsf, ts,
                            arrivals, flim=(0, fmax), hatch=f,
                            hatchlim=(0.0, 0.8), fig=fig)
plt.savefig('stransform_scalar.png', dpi=200)
plt.close()

del nip, f, Sr, St

############################### instantaneous azimuth ###################################
print("Estimating instantaneous azimuth.")
theta = filt.instantaneous_azimuth(Sv, Sn, Se, polarization, xpr)

fig = plt.figure(figsize=PORTRAIT)
gs0 = gridspec.GridSpec(3, 2)
gs0.update(hspace=0.10, wspace=0.10, left=0.05, right=0.95, top=0.95, bottom=0.05)
(ax11, ax12), (ax21, ax22), (ax31, ax32), (ax41, ax42), (ax51, ax52), (ax61, ax62) = make_tiles(fig, gs0)


# plot estimated instantaneous azimuth
plt.figure()
#az_prop = baz-180
print("Plotting instantaneous azimuth.")
plt.title('Instantaneous - great circle azimuth {:.1f}'.format(az_prop))
#plt.contourf(T, F, theta, cmap=plt.cm.hsv)
plt.imshow(theta - (az_prop), origin='lower', cmap=plt.cm.seismic, extent=[0, theta.shape[1], 0, fs/2], 
           aspect='auto',interpolation='nearest')
plt.colorbar()
plt.contour(T, F, theta - az_prop, [60, 40, 20, 0.0, -20, -40, -60], linewidth=1.5, colors='k')
#cs = plt.contour(T, F, theta - az_prop, [-20, -10, 0, 10, 20], colors='k')
#plt.clabel(cs)
#plt.contour(T, F, theta, [180+baz], linewidth=2.0, colors='k')
plt.axis('tight')
plt.ylim((0,fmax))
mx = np.nanmax(theta-az_prop)
plt.clim(-mx, mx)

plt.savefig('instantaneous_azimuth.png', dpi=200)
plt.close()

print("Computing NIP and filter.")
Sr, St = filt.rotate_NE_RT(Sn, Se, theta)
nip = filt.NIP(Sr, filt.shift_phase(Sv, polarization=polarization))
f = filt.get_filter(nip, polarization=polarization, threshold=0.8)

Sr[np.isnan(Sr)] = 0.0
St[np.isnan(St)] = 0.0
rdata = istransform(Sr, Fs=fs)
tdata = istransform(St, Fs=fs)

# plot NIP and filter
print("Plotting NIP and filter, inst. az.")
plt.figure()
plt.title('NIP, inst. azimuth')
plt.imshow(nip, cmap=plt.cm.seismic, origin='lower', extent=[0,nip.shape[1], 0, fs/2], 
           aspect='auto')
plt.colorbar()
plt.contour(T, F, nip, [0.8], linewidth=2.0, colors='k')
plt.axis('tight')
plt.ylim((0,fmax))
plt.ylabel('frequency [Hz]')
plt.xlabel('time [sec]')

plt.savefig('nip_filter.png', dpi=200)
plt.close()

# plot Sr and filtered Sr
print("Plotting Sr and filter, inst. az.")
plt.figure(figsize=LANDSCAPE)

plt.subplot(221)
plt.title('Sr(t,f) and filter, inst. azimuth')
plt.pcolormesh(T, F, np.abs(Sr))
plt.colorbar()
plt.contourf(T, F, f, [0, 0.8], colors='k', alpha=0.2)
plt.contour(T, F, f, [0.8], linewidth=1.0, colors='k')
plt.axis('tight')
plt.ylim((0,fmax))
plt.ylabel('frequency [Hz]')
plt.xlabel('time [sec]')

plt.subplot(222)
plt.title('St(t,f) and filter, inst. azimuth')
plt.pcolormesh(T, F, np.abs(St))
plt.colorbar()
plt.contourf(T, F, f, [0, 0.8], colors='k', alpha=0.2)
plt.contour(T, F, f, [0.8], linewidth=1.0, colors='k')
plt.axis('tight')
plt.ylim((0,fmax))
plt.ylabel('frequency [Hz]')
plt.xlabel('time [sec]')

plt.subplot(223)
plt.title('Sv(t,f) and filter, inst. azimuth')
plt.pcolormesh(T, F, np.abs(Sv))
plt.colorbar()
plt.contourf(T, F, f, [0, 0.8], colors='k', alpha=0.2)
plt.contour(T, F, f, [0.8], linewidth=1.0, colors='k')
plt.axis('tight')
plt.ylim((0,fmax))
plt.ylabel('frequency [Hz]')
plt.xlabel('time [sec]')

plt.savefig('stransforms_inst.png', dpi=200)
plt.close()

# get filtered time-domain waves
print("Getting time-domain waves.")
f[np.isnan(f)] = 0.0
Sr[np.isnan(Sr)] = 0.0
Sv[np.isnan(St)] = 0.0
rf = istransform(Sr*f, Fs=fs)
vf = istransform(Sv*f, Fs=fs)
tf = istransform(St*f, Fs=fs)

# plot original and filtered waves
print("Plotting filtered and raw waves.")
plt.figure(figsize=LANDSCAPE)
plt.subplot(311)
plt.title('vertical')
plt.plot(z.data, 'gray', label='original')
plt.plot(vf,'k', label='NIP filtered')
plt.legend(loc='lower left')

plt.subplot(312)
plt.title('radial')
plt.plot(r.data, 'gray', label='original')
plt.plot(rf, 'k', label='NIP filtered')
plt.legend(loc='lower left')

plt.subplot(313)
plt.title('transverse')
plt.plot(t.data, 'gray', label='original')
plt.legend(loc='lower left')

# plot arrivals
for arr, itt in arrivals:
    plt.vlines(itt, z.data.min(), z.data.max(), 'k', linestyle='dashed')
    plt.text(itt, z.data.max(), arr, fontsize=9, horizontalalignment='left', va='top')

plt.axis('tight')
plt.savefig('waves_inst.png', dpi=200)
plt.close()


if __name__ == '__main__':
    
