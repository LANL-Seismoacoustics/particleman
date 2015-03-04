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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import numpy as np

from obspy import read
from obspy.signal import rotate_RT_NE
from obspy.taup import taup

from distaz import distaz

from stockwell import stransform, istransform
import stockwell.filter as filt

LANDSCAPE = (11,8.5)
PORTRAIT = (8.5,11)
POWERPOINT = (10, 7.5)
SCREEN = (31, 19)

# save myself some plotting memory for big arrays
#plt.ioff()

# filter band
fmin = 1./80
fmax = 1./30

#st = read("tests/data/SampleWaveforms/E2010-01-10-00-27-39/Dsp/aak-ii-00*")
#st = read("tests/data/SampleWaveforms/E2010-01-10-00-27-39/Dsp/anmo*")
st = read("tests/data/SampleWaveforms/E2010-01-10-00-27-39/Dsp/mdj*")
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
vmax = 5.0 
vmin = 2.5
#vmin = 1.5
swmin = km/vmax
swmax = km/vmin
tmin = tr.stats.starttime + swmin
tmax = tr.stats.starttime + swmax
tt = taup.getTravelTimes(deg, sac.evdp, model='ak135')
tarrivals = [(itt['phase_name'], itt['time']) for itt in tt]
swarrivals = [(str(swvel), km/swvel) for swvel in (5.0, 4.0, 3.0)]
tarrivals.extend(swarrivals)
arrivals= []
for arr, itt in tarrivals:
    if arr in ('P', 'S') or (arr, itt) in swarrivals:
        arrivals.append((arr, itt))

st = st.trim(endtime=tmax)
st.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True)

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
tm = np.arange(len(v))*fs

# get original z and n components by unrotating
n, e = rotate_RT_NE(rs, ts, baz)

# raw N, E, V s-transforms."
print("Computing raw N, E, V s-transforms.")
Sn, T, F = stransform(n, fs, return_time_freq=True)
Se = stransform(e, fs)
Sv = stransform(v, fs)

Srs = stransform(r, fs)
Sts = stransform(t, fs)

nips = filt.NIP(Srs, filt.shift_phase(Sv, polarization=polarization))
sfilt = filt.get_filter(nips, polarization=polarization, threshold=0.8)

# remove nans
sfilt[np.isnan(sfilt)] = 0.0
Srs[np.isnan(Srs)] = 0.0
Sts[np.isnan(Sts)] = 0.0

#m, n = Sts.shape
#T = T[:m, :n]
#F = F[:m, :n]
rsf = istransform(Srs*f, Fs=fs)
vsf = istransform(Sv*f, Fs=fs)
#tsf = istransform(Sts*f, Fs=fs)

xpr = -int(np.sign(np.sin(np.radians(baz))))
polarization = 'retrograde'


################ map ################
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

######################## scalar azimuth ############################################
print("Computing raw R, T s-transforms.")

print("Computing NIP and filter.")

# plot NIP and filter
print("Plotting NIP and filter, scalar az.")

plt.figure()
plt.title('NIP, scalar azimuth')
plt.imshow(nips, cmap=plt.cm.seismic, origin='lower', extent=[0,nips.shape[1], 0, fs/2], 
           aspect='auto',interpolation='nearest')
plt.colorbar()
plt.contour(T, F, nips, [0.8], linewidth=2.0, colors='k')
plt.axis('tight')
plt.ylim((0,fmax))
plt.ylabel('frequency [Hz]')
plt.xlabel('time [sec]')

plt.savefig('nip_filter_scalar.png', dpi=200)
plt.close()

# plot Sr and filtered Sr
# plot_tile
plt.savefig('stransforms_scalar.png', dpi=200)
plt.close()

# plot original and filtered waves
print("Plotting filtered and raw waves.")
plt.figure(figsize=LANDSCAPE)
plt.subplot(311)
plt.title('vertical')
plt.plot(v, 'gray', label='original')
plt.plot(vsf,'k', label='NIP filtered')
plt.legend(loc='lower left')

plt.subplot(312)
plt.title('radial')
plt.plot(rs, 'gray', label='original')
plt.plot(rsf, 'k', label='NIP filtered')
plt.legend(loc='lower left')

plt.subplot(313)
plt.title('transverse')
plt.plot(ts, 'gray', label='original')
plt.legend(loc='lower left')

# plot arrivals
for arr, itt in arrivals:
    plt.vlines(itt, v.min(), v.max(), 'k', linestyle='dashed')
    plt.text(itt, v.max(), arr, fontsize=9, horizontalalignment='left', va='top')

plt.savefig('waves_scalar.png', dpi=200)
plt.close()

############################ gridspec plots
# from http://matplotlib.org/1.3.1/users/gridspec.html
def plot_tile(fig, ax1, T, F, S, ax2, d1, label1, d2=None, label2=None, arrivals=None,
              flim=None, clim=None, hatch=None, hatchlim=None, dlim=None):
    """
    Returns
    -------
    matplotlib.collections.QuadMesh
        The ax1 image from pcolormesh.

    Examples
    --------
    # filtered versus unfiltered radial, and set color limits
    >>> plot_tile(fig, ax21, T, F, Srs, ax22, rs, 'unfiltered', rsf, 'NIP filtered', 
        arrivals=arrivals, flim=(0.0, fmax), clim=(0.0, 5e-5), hatch=sfilt, hatchlim=(0.0, 0.8))
    # scalar versus dynamic rotated radial 
    >>> plot_tile(fig, ax21, T, F, Srs, ax22, rs, 'scalar', rd, 'dynamic', arrivals
        flim=(0.0, fmax), clim=(0.0, 5e-5), hatch=dfilt, hatchlim=(0.0, 0.8))

    """
    tm = T[0]
    # TODO: remove fig from signature?
    ax1.axes.get_xaxis().set_visible(False)
    im = ax1.pcolormesh(T, F, np.abs(S))
    if clim:
        im.set_clim(clim)
    if (hatch is not None) and hatchlim:
        ax1.contourf(T, F, hatch, hatchlim, colors='k', hatches=['x'], alpha=0.0)
        ax1.contour(T, F, hatch, [max(hatchlim)], linewidth=1.0, colors='k')
    if flim:
        ax1.set_ylim(flim)
    ax1.set_ylabel('frequency [Hz]')
    divider = make_axes_locatable(ax1)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #cbar = plt.colorbar(im, cax=cax, format='%.2e')
    fig.add_subplot(ax1)

    # waves and arrivals
    if d2 is not None:
        ax2.plot(tm, d2, 'gray', label=label2)
    ax2.plot(tm, d1, 'k', label=label1)
    ax2.set_ylabel('amplitude')
    leg = ax2.legend(loc='lower left', frameon=False, fontsize=14)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    ax2.set_xlim(tm[0], tm[-1])
    if dlim:
        ax2.set_ylim(dlim)
    if arrivals:
        for arr, itt in arrivals:
            ax2.vlines(itt, d1.min(), d1.max(), 'k', linestyle='dashed')
            ax2.text(itt, d1.max(), arr, fontsize=12, ha='left', va='top')

    cbar = plt.colorbar(im, fraction=0.05, pad=0.01, ax=[ax1, ax2], format='%.2e')
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    fig.add_subplot(ax2)

    return im



fig = plt.figure(figsize=SCREEN)
# course 2x2 grid
gs0 = gridspec.GridSpec(2, 2)
gs0.update(hspace=0.10, wspace=0.10, left=0.05, right=0.95, top=0.95, bottom=0.05)

# top left axes: NIP
gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[0])
ax11 = plt.Subplot(fig, gs1[:, :])
ax11.set_title('NIP, retrograde Rayleigh, scalar azimuth')
im = ax11.imshow(nip, cmap=plt.cm.seismic, origin='lower', extent=[0,nip.shape[1], 0, fs/2], 
                 aspect='auto',interpolation='nearest')
#plt.colorbar(im)
ax11.contour(T, F, nip, [0.8], linewidth=2.0, colors='k')
ax11.axis('tight')
ax11.set_ylim((0, fmax))
ax11.set_ylabel('frequency [Hz]')
ax11.set_xlabel('time [sec]')

divider = make_axes_locatable(ax11)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
#cbar = plt.colorbar(im, fraction=0.05, ax=ax11, format='%.2e')

fig.add_subplot(ax11)

# top right: Radial
# s transform and filter
gs2 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[1], hspace=0.0)
ax21 = plt.Subplot(fig, gs2[:-1, :])
ax22 = plt.Subplot(fig, gs2[-1, :], sharex=ax21)
ax21.axes.get_xaxis().set_visible(False)
ax21.set_title('Radial')
im = ax21.pcolormesh(T, F, np.abs(Sr))
print "images"
print ax21.get_images()
#plt.colorbar(pc)
#ax21.contourf(T, F, f, [0, 0.8], colors='k', alpha=0.2)
ax21.contourf(T, F, f, [0, 0.8], colors='k', hatches=['x'], alpha=0.0)
ax21.contour(T, F, f, [0.8], linewidth=1.0, colors='k')
ax21.set_ylim((0, fmax))
ax21.set_ylabel('frequency [Hz]')
divider = make_axes_locatable(ax21)
#cax = divider.append_axes("right", size="5%", pad=0.05)
#cbar = plt.colorbar(im, cax=cax, format='%.2e')
fig.add_subplot(ax21)
# waves and arrivals
ax22.plot(rdata, 'gray', label='original')
#ax22.plot(r.data, 'gray', label='original')
ax22.plot(rf, 'k', label='NIP filtered')
ax22.set_ylabel('amplitude')
leg = ax22.legend(loc='lower left', frameon=False, fontsize=14)
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)
ax22.axis('tight')
for arr, itt in arrivals:
    ax22.vlines(itt, r.data.min(), r.data.max(), 'k', linestyle='dashed')
    ax22.text(itt, r.data.max(), arr, fontsize=12, horizontalalignment='left', va='top')
cbar = plt.colorbar(im, fraction=0.05, pad=0.01, ax=[ax21, ax22], format='%.2e')
ax22.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.add_subplot(ax22)

# bottom left: Transverse
# s transform and filter
gs3 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[2], hspace=0.0)
ax31 = plt.Subplot(fig, gs3[:-1, :])
ax31.set_title('Transverse S(t,f), scalar azimuth')
ax31.axes.get_xaxis().set_visible(False)
im = ax31.pcolormesh(T, F, np.abs(St))
ax31.contourf(T, F, f, [0, 0.8], colors='k', hatches=['x'], alpha=0.0)
ax31.contour(T, F, f, [0.8], linewidth=1.0, colors='k')
ax31.axis('tight')
ax31.set_ylim((0, fmax))
ax31.set_ylabel('frequency [Hz]')
fig.add_subplot(ax31)
# waves and arrivals
ax32 = plt.Subplot(fig, gs3[-1, :], sharex=ax31)
ax32.plot(tdata, 'gray', label='original')
#ax32.plot(t.data, 'gray', label='original')
try:
    ax32.plot(tf, 'k', label='filtered')
except NameError:
    pass
ax32.set_ylabel('amplitude')
leg = ax32.legend(loc='lower left', frameon=False, fontsize=14)
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)
ax32.axis('tight')
for arr, itt in arrivals:
    ax32.vlines(itt, t.data.min(), t.data.max(), 'k', linestyle='dashed')
    ax32.text(itt, t.data.max(), arr, fontsize=12, horizontalalignment='left', va='top')
cbar = plt.colorbar(im, fraction=0.05, pad=0.01, ax=[ax31, ax32], format='%.2e')
ax32.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.add_subplot(ax32)

# bottom right: Vertical
# s transform and filter
gs4 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[3], hspace=0.0)
ax41 = plt.Subplot(fig, gs4[:-1, :])
ax41.set_title('Vertical')
ax41.axes.get_xaxis().set_visible(False)
im = ax41.pcolormesh(T, F, np.abs(Sv))
ax41.contourf(T, F, f, [0, 0.8], colors='k', hatches=['x'], alpha=0.0)
ax41.contour(T, F, f, [0.8], linewidth=1.0, colors='k')
ax41.set_ylim((0, fmax))
ax41.set_ylabel('frequency [Hz]')
fig.add_subplot(ax41)
# waves and arrivals
ax42 = plt.Subplot(fig, gs4[-1, :], sharex=ax41)
ax42.plot(z.data, 'gray', label='original')
ax42.plot(vf, 'k', label='NIP filtered')
ax42.set_ylabel('amplitude')
leg = ax42.legend(loc='lower left', frameon=False, fontsize=14)
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)
ax42.axis('tight')
for arr, itt in arrivals:
    ax42.vlines(itt, z.data.min(), z.data.max(), 'k', linestyle='dashed')
    ax42.text(itt, z.data.max(), arr, fontsize=12, horizontalalignment='left', va='top')
cbar = plt.colorbar(im, fraction=0.05, pad=0.01, ax=[ax41, ax42], format='%.2e')
ax42.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.add_subplot(ax42)

plt.savefig('stransform_scalar.png', dpi=200)

del nip, f, Sr, St

############################### instantaneous azimuth ###################################
print("Estimating instantaneous azimuth.")
theta = filt.estimate_azimuth(Sv, Sn, Se, polarization, xpr)

# plot estimated instantaneous azimuth
plt.figure()
#az_prop = baz-180
print("Plotting instantaneous azimuth.")
plt.title('Instantaneous - great circle azimuth {:.1f}'.format(az_prop))
#plt.contourf(T, F, theta, cmap=plt.cm.hsv)
plt.imshow(theta - (az_prop), origin='lower', cmap=plt.cm.seismic, extent=[0, theta.shape[1], 0, fs/2], 
           aspect='auto',interpolation='nearest')
plt.colorbar()
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
