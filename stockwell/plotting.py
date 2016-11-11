import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from matplotlib.ticker import FormatStrFormatter
import numpy as np

LANDSCAPE = (11,8.5)
PORTRAIT = (8.5,11)
POWERPOINT = (10, 7.5)
SCREEN = (31, 19)
HALFSCREEN = (SCREEN[0]/2.0, SCREEN[1])


# save myself some plotting memory for big arrays
#plt.ioff()


def plot_tile(fig, ax1, T, F, S, ax2, d1, label1, d2=None, label2=None,
              arrivals=None, flim=None, clim=None, hatch=None, hatchlim=None,
              dlim=None):
    """
    Plot time-frequency pcolormesh tiles above time-aligned aligned time-series.

    Parameters
    ----------
    fig : matplotlib.Figure
    ax1 : matplotlib.axis.Axis
        Axis for time-frequency pcolormesh tile and optional hatching.
    ax2 : matplotlib.axis.Axis
        Axis for time-series plot.
    T, F, S : numpy.ndarray (ndim 2)
        Time, frequency, S-transform tiles from stockwell.stransform
    d1, d2 : numpy.ndarray (ndim 1)
        Time-series, plotted black.  Optional d2 plotted gray. These need to
        be registered in time to T.
    arrivals : sequence of (str, float) 2-tuples
        Sequence of arrivals to plot, of the form (label, time_in_seconds)
    dlim : 2-tuple of floats
        Limits on the time-series amplitudes (y axis limits).
    hatch : numpy.ndarray (ndim 2)
        Optional tile used for hatch mask.
    hatchlim : tuple
        Hatch range used to display mask.  2-tuple of floats (hmin, hmax).

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
    sciformatter = FormatStrFormatter('%.2e')

    # grab a time vector
    tm = T[0]

    # TODO: remove fig from signature?
    ax1.axes.get_xaxis().set_visible(False)
    im = ax1.pcolormesh(T, F, np.abs(S))
    if clim:
        im.set_clim(clim)
    if (hatch is not None) and hatchlim:
        ax1.contourf(T, F, hatch, hatchlim, colors='w', alpha=0.2)
        #ax1.contourf(T, F, hatch, hatchlim, colors='w', hatches=['x'], alpha=0.0)
        #ax1.contour(T, F, hatch, [max(hatchlim)], linewidth=1.0, colors='w')
    if flim:
        ax1.set_ylim(flim)
    ax1.set_ylabel('frequency [Hz]')
    divider = make_axes_locatable(ax1)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #cbar = plt.colorbar(im, cax=cax, format='%.2e')
    fig.add_subplot(ax1)

    # waves and arrivals
    dmx = np.abs(d1).max()
    if d2 is not None:
        ax2.plot(tm, d2, 'gray', label=label2)
        dmx = max([dmx, np.abs(d2).max()])
    ax2.plot(tm, d1, 'k', label=label1)
    ax2.set_ylabel('amplitude')
    leg = ax2.legend(loc='lower left', frameon=False, fontsize=14)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    ax2.set_xlim(tm[0], tm[-1])
    if not dlim:
        dlim = (-dmx, dmx)
    ax2.set_ylim(dlim)
    if arrivals:
        for arr, itt in arrivals:
            ax2.vlines(itt, d1.min(), d1.max(), 'k', linestyle='dashed')
            ax2.text(itt, d1.max(), arr, fontsize=12, ha='left', va='top')

    cbar = plt.colorbar(im, fraction=0.05, pad=0.01, ax=[ax1, ax2], format='%.2e')
    #ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.yaxis.set_major_formatter(sciformatter)
    fig.add_subplot(ax2)

    return im


def make_tiles(fig, gs0, skip=[]):
    """
    Give a list of (ax1, ax2) tuples for each non-skipped SubPlotSpec in gs0.

    """
    axes = []
    for i, igs in enumerate(gs0):
        if i in skip:
            pass
        else:
            iigs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=igs, hspace=0.0)
            ax1 = plt.Subplot(fig, iigs[:-1, :])
            ax2 = plt.Subplot(fig, iigs[-1, :], sharex=ax1)
            axes.append((ax1, ax2))

    return axes


def plot_instantaneous_azimuth(theta, fs=1.0, ylim=None, xlim=None, fig=None,
                               outfile=None):
    """
    Plot the instantanous azimuth TF tile using imshow.

    Parameters
    ----------
    theta : numpy.ndarray (ndim 2)
        Instantaneous azimuth calculated by stockwell.filter.instantanous_azimuth
    fs : float
        Sampling rate of data used.
    ylim : tuple (ymin, ymax)
        Optional frequency min/max for plot y limits in Hz.
    xlim : tuple (xmin, xmax)
        Optional time min/max for plot x limits in sec.
    fig : matplotlib.Figure instance

    Returns
    -------
    matplotlib.Figure

    """
    if not fig:
        f = plt.figure()

    plt.imshow(theta, origin='lower', cmap=plt.cm.hsv, aspect='auto',
               extent=[0, theta.shape[1], 0, fs/2.0], interpolation='nearest')
    plt.colorbar()
    plt.axis('tight')

    if ylim:
        plt.ylim(ylim)

    if xlim:
        plt.xlim((t0, len(v)))

    mx = np.nanmax(theta)
    plt.clim(-mx, mx)

    return f


def rotation_comparison(T, F, Sv, Srs, Srd, Sts, Std, v, rs, rd, ts, td,
                        arrivals, flim, clim, dlim, hatch=None, hatchlim=None,
                        fig=None, xlim=None):
    """
    Make a 6-panel side-by-side comparison of scalar versus dynamic rotations.

    Sv, Srs, Srd, Sts, Std : numpy.ndarray (ndim 2)
        The vertical, radial-scalar, radial-dynamic, transverse-scalar, and
        transverse-dynamic stockwell transforms.
    v, rs, rd, ts, td : numpy.ndarray (ndim 1)
        The corresponding vertical, radial-scalar, radial-dynamic,
        transverse-scalar, and transverse-dynamic time-series vectors.
    arrivals : sequence of (str, float) 2-tuples
        Sequence of arrivals to plot, of the form (label, time_in_seconds)
    flim, clim, dlim, xlim : tuple
        Frequency, stockwell amplitude, time-series amplitude, and time-series
        time limits of display.  2-tuples of (min, max) floats.
    hatch : numpy.ndarray (ndim 2)
        Optional tile used for hatch mask.
    hatchlim : tuple
        Hatch range used to display mask.  2-tuple of floats (hmin, hmax).

    Returns
    -------
    matplotlib.Figure

    """
    # from http://matplotlib.org/1.3.1/users/gridspec.html
    if not fig:
        fig = plt.figure()

    gs0 = gridspec.GridSpec(3, 2)
    gs0.update(hspace=0.15, wspace=0.15, left=0.05, right=0.95, top=0.95,
               bottom=0.05)
    tile1, tile2, tile3, tile4, tile5, tile6 = make_tiles(fig, gs0)
    ax11, ax12 = tile1
    ax21, ax22 = tile2
    ax31, ax32 = tile3
    ax41, ax42 = tile4
    ax51, ax52 = tile5
    ax61, ax62 = tile6

    ax11.set_title('Vertical')
    plot_tile(fig, ax11, T, F, Sv, ax12, v, 'vertical', arrivals=arrivals, 
              flim=flim, clim=clim, dlim=dlim)

    ax21.set_title('Vertical')
    plot_tile(fig, ax21, T, F, Sv, ax22, v, 'vertical', arrivals=arrivals, 
            flim=flim, clim=clim, dlim=dlim)

    ax31.set_title('Radial, scalar')
    plot_tile(fig, ax31, T, F, Srs, ax32, rs, 'great circle', rd, 'dynamic', 
            arrivals=arrivals, flim=flim, clim=clim, dlim=dlim, hatch=hatch,
            hatchlim=hatchlim)
    #ax31.contour(T, F, theta - az_prop, [20, 0.0, -20], linewidth=1.5, 
    #             colors=['r','w','b'])
    #ax31.contour(T, F, theta - az_prop, [-40, 0, 40], cmap=plt.cm.seismic)

    ax41.set_title('Radial, dynamic')
    plot_tile(fig, ax41, T, F, Srd, ax42, rd, 'dynamic', rs, 'great circle',  
            arrivals=arrivals, flim=flim, clim=clim, dlim=dlim, hatch=hatch,
            hatchlim=hatchlim)

    ax51.set_title('Transverse, scalar')
    plot_tile(fig, ax51, T, F, Sts, ax52, ts, 'great circle', td, 'dynamic', 
            arrivals=arrivals, flim=flim, clim=clim, dlim=dlim, hatch=hatch,
            hatchlim=hatchlim)
    #ax51.contour(T, F, theta - az_prop, [40, 0.0, -40], linewidth=1.5, 
    #             colors=['r','w','b'])
    #ax51.contour(T, F, theta - az_prop, [-40, 0, 40], cmap=plt.cm.seismic)

    ax61.set_title('Transverse, dynamic')
    plot_tile(fig, ax61, T, F, Std, ax62, td, 'dynamic', ts, 'great circle',
            arrivals=arrivals, flim=flim, clim=clim, dlim=dlim, hatch=hatch,
            hatchlim=hatchlim)

    if xlim:
        ax11.set_xlim(*xlim)
        ax21.set_xlim(*xlim)
        ax31.set_xlim(*xlim)
        ax41.set_xlim(*xlim)
        ax51.set_xlim(*xlim)
        ax61.set_xlim(*xlim)

    return fig


def check_filters(T, F, Sv, Srs, Sts, vsf, rsf, ts, arrivals, flim, clim
                  dlim, xlim, hatch=None, hatchlim=None, fig=None):
    """
    Parameters
    ----------
    T, F : numpy.ndarray (ndim 2)
        The time and freqency domain tiles/grids.
    Sv, Srs, Sts : numpy.ndarray (ndim 2)
        The vertical, scalar-rotated radial, and scalar-rotated transverse
        Stockwell transform tiles.
    vsf, rsf, ts : numpy.ndarray (ndim 1)
        The Stockwell NIP-filtered vertical and radial, and transverse
        time-series vectors.
    arrivals : sequence of (str, float) 2-tuples
        Sequence of arrivals to plot, of the form (label, time_in_seconds)
    dlim : 2-tuple of floats
        Limits on the time-series amplitudes (y axis limits).
    hatch : numpy.ndarray (ndim 2)
        Optional tile used for hatch mask.
    hatchlim : tuple
        Hatch range used to display mask.  2-tuple of floats (hmin, hmax).
    fig : matplotlib.Figure

    Returns
    -------
    matplotlib.Figure

    """
    if not fig:
        fig = plt.figure()

    gs0 = gridspec.GridSpec(3, 1)
    gs0.update(hspace=0.15, wspace=0.15, left=0.05, right=0.95, top=0.95,
               bottom=0.05)

    tile1, tile2, tile3 = make_tiles(fig, gs0)
    ax11, ax12 = tile1
    ax21, ax22 = tile2
    ax31, ax32 = tile3

    ax11.set_title('Vertical, scalar rotation')
    plot_tile(fig, ax11, T, F, Sv, ax12, vsf, 'filtered', v, 'vertical',
              arrivals=arrivals, flim=flim, clim=clim, dlim=dlim, hatch=hatch,
              hatchlim=hatchlim)

    ax21.set_title('Radial, scalar rotation')
    plot_tile(fig, ax21, T, F, Srs, ax22, rsf, 'filtered', rs, 'radial',
              arrivals=arrivals, flim=flim, clim=clim, dlim=dlim, hatch=hatch,
              hatchlim=hatchlim)

    ax31.set_title('Transverse, scalar rotation')
    plot_tile(fig, ax31, T, F, Sts, ax32, ts, 'transverse', arrivals=arrivals,
              flim=flim, clim=clim, dlim=dlim, hatch=hatch, hatchlim=hatchlim)

    if xlim:
        ax11.set_xlim(*xlim)
        ax21.set_xlim(*xlim)
        ax31.set_xlim(*xlim)

    return fig
