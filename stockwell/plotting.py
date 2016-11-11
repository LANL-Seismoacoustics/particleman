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


def plot_tile(fig, ax1, T, F, S, ax2, d1, label1, d2=None, label2=None, arrivals=None,
              flim=None, clim=None, hatch=None, hatchlim=None, dlim=None):
    """
    Parameters
    ----------
    fig : matplotlib.Figure
    ax1 : matplotlib.axis.Axis
    T, F, S : numpy.ndarray (rank 2)
        Time, frequency, S-transform tiles from stockwell.stransform
    ax2

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



def instantaneous_azimuth():

