#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module serves to implement necessary code to simulate
diffusion.

This module has functions for displaying simulations.
"""

import numpy as np
from matplotlib import cm
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import (MultipleLocator,
                               IndexLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

graph_lim = 10


def display_atom(atom_history, show_displacement=False):
    """Shows the position of an atom in a square lattice. Optionally
    show the atom position 'crumb trail' as well as display the
    displacment of the atom versus number of jumps.

    Args:
        atom (list): x,y coordinate of atom in the lattice.
    """

    # Create main figure and axis for atom plot
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot('121')

    # Check whether to plot atom jump history
    if isinstance(atom_history[0], int):
        # Place atom on the square lattice. Adjust zorder to enusure atom
        # is on top of grid lines
        x_final, y_final = atom_history
        ax1.scatter(x=x_final, y=y_final, zorder=3.5, c='blue')
    else:
        # x_final, y_final = atom_history[-1]
        # x_hist, y_hist = zip(*atom_history)
        # ax1.plot(x_hist, y_hist, c='g', alpha=0.6, zorder=2.5)
        # ax1.scatter(x=x_final, y=y_final, zorder=3.5, c='g')
        x_final, y_final = atom_history[-1]
        draw_atom_history(ax1, atom_history)

    # Add in arrow to highlight atom position
    ax1.arrow(x=0, y=0, dx=x_final, dy=y_final, width=0.2,
              length_includes_head=True, color='purple',
              zorder=3.0)

    # Set the limits of the x- and y-axes
    ax1.set_xlim(-graph_lim, graph_lim)
    ax1.set_ylim(-graph_lim, graph_lim)

    # Ensure that the axes look square
    # ax1.set_aspect('equal', adjustable='box')
    set_equal_aspect(ax1)

    # Add in square lattice atoms
    draw_lattice_atoms(ax1)

    # Add gridlines
    ax1.grid()

    # Format axis ticks
    set_ticks(ax1)

    # Check whether to plot atom displacement to add in additional
    # plot
    if show_displacement:
        ax2 = fig.add_subplot('122')
        disp_history = displacement_history(atom_history)
        ax2.plot(disp_history, c='purple', linewidth=4)
        set_equal_aspect(ax2)
        disp = disp_history[-1]
    else:
        disp = displacement([x_final, y_final])

    # Display final displacement of atom as text.
    ax1.text(x=0.95, y=0.92,
             s="Displacement = {0:.2f}".format(disp),
             horizontalalignment='right', verticalalignment='center',
             transform=ax1.transAxes,
             zorder=4,
             bbox=dict(facecolor='white', alpha=1))

    # Display the atom on the square lattice
    plt.show()


def display_atoms(atom_histories):
    """Shows the position of an atom in a square lattice.

    Args:
        atom (list): x,y coordinate of atom in the lattice.
    """

    mean_disp = 0
    n_atoms = len(atom_histories)
    m_steps = len(atom_histories[0])
    mean_disp_history = [0]*m_steps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    for atom_history in atom_histories:
        # Place atom on the square lattice. Adjust zorder to enusure atom
        # is on top of grid lines
        draw_atom_history(ax1, atom_history, n_atoms)
        disp_history = displacement_history(atom_history)
        ax2.plot(disp_history, c='purple', alpha=set_alpha(n_atoms))
        # ax1.scatter(x_final, y_final, zorder=3.0, c='g', alpha=0.25)
        # disp_final += (x_final**2 + y_final**2)**0.5 / n_atoms
        # hist_x, hist_y = zip(*atom_history)
        # ax1.plot(hist_x, hist_y, c='g', alpha=0.05, zorder=2.5)
        for i, disp in enumerate(disp_history):
            mean_disp_history[i] += disp/n_atoms

    ax2.plot(mean_disp_history, zorder=3, c='purple', linewidth=4)
    set_equal_aspect(ax2)

    # Set the limits of the x- and y-axes
    ax1.set_xlim(-graph_lim, graph_lim)
    ax1.set_ylim(-graph_lim, graph_lim)

    draw_lattice_atoms(ax1)

    # Add gridlines
    ax1.grid()

    # Ensure that the axes look square
    # ax1.set_aspect('equal', adjustable='box')
    set_equal_aspect(ax1)

    ax1.text(0.95, 0.92, "Mean Displacement = {0:.2f}".format(mean_disp_history[-1]), horizontalalignment='right', verticalalignment='center', transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=1))

    # Make x-axis with major ticks that
    # are multiples of 11 and Label major
    # ticks with '% 1.2f' formatting
    ax1.yaxis.set_major_locator(MultipleLocator(5))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('% d'))
    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('% d'))

    # make x-axis with minor ticks that
    # are multiples of 1 and label minor
    # ticks with '% 1.2f' formatting
    # ax1.minorticks_off()
    set_ticks(ax1)

    # Display the atom on the square lattice
    plt.show()


def display_probability(atom_histories, compare_gaussian=True):

    atoms_final = [atom_history[-1] for atom_history in atom_histories]
    x, y = zip(*atoms_final)
    k = gaussian_kde(np.vstack([x, y]))
    # xi, yi = np.mgrid[min(x):max(x):len(x)**0.5*1j,min(y):max(y):len(y)**0.5*1j]
    xi, yi = np.mgrid[-10:10:20j, -10:10:20j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))*100

    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121)

    # alpha=0.5 will make the plots semitransparent
    cmesh = ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5)

    # ax1.set_xlim([-10, 10])
    # ax1.set_ylim([-10, 10])

    set_equal_aspect(ax1)

    # ax1.yaxis.set_major_locator(MultipleLocator(5))
    # ax1.yaxis.set_major_formatter(FormatStrFormatter('% d'))
    # ax1.xaxis.set_major_locator(MultipleLocator(5))
    # ax1.xaxis.set_major_formatter(FormatStrFormatter('% d'))
    set_ticks(ax1)

    # fig.colorbar(cmesh, ax=ax1)
    # add_colorbar(cmesh)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cmesh,cax=cax)

    if compare_gaussian:
        ax2 = fig.add_subplot(122)
        mu = 0
        num_jumps = len(atom_histories[0])
        sigma = (num_jumps/3)**0.5 * 1
        # sigma = 1

        # Initializing value of x-axis and y-axis
        # in the range -1 to 1
        x, y = np.meshgrid(np.linspace(-10,10,100), np.linspace(-10,10,100))
        dst = np.sqrt(x*x+y*y)

        # Calculating Gaussian array
        gauss = np.exp(-( (dst-mu)**2 / ( 2.0 * sigma**2 ) ) ) * 100

        # alpha=0.5 will make the plots semitransparent
        ax2.contourf(x, y, gauss)

        # ax1.set_xlim([-10, 10])
        # ax1.set_ylim([-10, 10])

        # Set the limits of the x- and y-axes
        ax2.set_xlim(-10, 10)
        ax2.set_ylim(-10, 10)

        x0,x1 = ax2.get_xlim()
        y0,y1 = ax2.get_ylim()
        ax2.set_aspect((x1-x0)/(y1-y0))

        ax2.yaxis.set_major_locator(MultipleLocator(5))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('% d'))
        ax2.xaxis.set_major_locator(MultipleLocator(5))
        ax2.xaxis.set_major_formatter(FormatStrFormatter('% d'))

        ax2.grid()

        # fig.colorbar(cmesh, ax=ax1)
        # add_colorbar(cont_mesh)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cmesh,cax=cax)


    plt.show()


def draw_lattice_atoms(ax):
    x_lower, x_upper = [int(lim) for lim in ax.get_xlim()]
    y_lower, y_upper = [int(lim) for lim in ax.get_ylim()]
    for x in range(x_lower+1, x_upper+1):
        for y in range(y_lower+1, y_upper+1):
            lattice_circle = plt.Circle((x-0.5, y-0.5), 0.45, fill=False, edgecolor='gray')
            ax.add_artist(lattice_circle)


def draw_atom_history(ax, atom_history, n_atoms=1):
    x_final, y_final = atom_history[-1]
    x_hist, y_hist = zip(*atom_history)
    if n_atoms != 1:
        alpha = 0.2
    else:
        alpha = 1
    ax.plot(x_hist, y_hist, c='blue', alpha=alpha, zorder=2.5)
    ax.scatter(x=x_final, y=y_final, alpha=set_alpha(n_atoms), zorder=3.5, c='blue')


def set_equal_aspect(ax):
    x_lower, x_upper = ax.get_xlim()
    y_lower, y_upper = ax.get_ylim()
    ax.set_aspect((x_upper-x_lower)/(y_upper-y_lower))


def set_ticks(ax):

    # Make x and y axis major ticks multiples of 5
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_locator(MultipleLocator(5))

    # Make x and y axis major tiock labels integers
    ax.yaxis.set_major_formatter(FormatStrFormatter('% d'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('% d'))

    # Turn off minor ticks
    ax.minorticks_off()


def displacement_history(atom_history):
    disp_history = []
    for xy in atom_history:
        disp_history.append(displacement(xy))

    return disp_history


def displacement(xy):
    x = xy[0]
    y = xy[1]

    return (x**2 + y**2)**0.5


def set_alpha(x):
    n = 5
    return n / (n+x)

