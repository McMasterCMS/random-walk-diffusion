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

# Hard coded value for simulation limits and figure size
graph_lim = 10
figsize = (14, 7)


def display_atom(atom_history, show_displacement=False):
    """Shows the position of an atom in a square lattice.

    Optionally show the atom position 'crumb trail' as well as
    display the displacment of the atom versus number of jumps.

    Parameters
    ----------
    atom : list
           x,y coordinate of atom in the lattice.

    show_displacement : boolean
        Include the plot show displacement of atom. Defaults to False.
    """

    # Create main figure and axis for atom plot
    fig = plt.figure(figsize=figsize)
    ax_atom = fig.add_subplot('121')

    # Check whether atom is a single coordinate or a list including
    # previous positions
    if isinstance(atom_history[0], int):
        # Place atom on the square lattice
        x_final, y_final = atom_history
        ax_atom.scatter(x=x_final, y=y_final, zorder=3.5, c='blue')
    else:
        # Draw atom previous position, incuding final atom position
        x_final, y_final = atom_history[-1]
        draw_atom_history(ax_atom, atom_history)

    # Add in arrow to highlight atom position
    ax_atom.arrow(x=0, y=0, dx=x_final, dy=y_final, width=0.2,
                  length_includes_head=True, color='purple',
                  zorder=3.0)

    # Check whether to plot atom displacement
    if show_displacement:
        ax_disp = fig.add_subplot('122')

        # Draw and format
        draw_disp_history(ax_disp, atom_history)
        ax_disp.set_ylabel("Displacement")
        ax_disp.set_xlabel("Number of Simulaton Steps")

        # Get displacement for atom arrow, otherwise calculate it
        disp = disp_history[-1]
    else:
        disp = displacement([x_final, y_final])

    # Display final displacement of atom as text.
    draw_box(ax_atom, "Displacement", disp)

    # Add in square lattice atoms
    draw_lattice_atoms(ax_atom)
    set_ticks(ax_atom)

    # Ensure that the axes look square
    set_equal_aspect(ax_atom)
    set_equal_aspect(ax_disp)

    # Display the atom on the square lattice
    plt.show()


def display_atoms(atom_histories):
    """Shows the position of various atoms in a square lattice.

    Overlays the resulting simulation of various atoms and their
    positions. The displacement vs. simulation time graph is included
    displaying each atom displacement history as well at the average.

    Parameters
    ----------
    atom_histories
           The position history of various atoms as a list of lists.
    """

    n_atoms = len(atom_histories)
    m_steps = len(atom_histories[0])
    mean_disp_history = [0]*m_steps

    fig, (ax_atoms, ax_disps) = plt.subplots(1, 2, figsize=figsize)

    # Iterate through all atoms simulations
    for atom_history in atom_histories:
        # Draw atom trajectory and displacement
        draw_atom_history(ax_atoms, atom_history, n_atoms)
        draw_disp_history(ax_disps, n_atoms)

        # Calculate the average displacement for the simulation
        # Append to list for plotting
        disp_history = calc_disp_history(atom_history)
        for i, disp in enumerate(disp_history):
            mean_disp_history[i] += disp/n_atoms

    # Plot average of all trajectory displacements
    ax_disps.plot(mean_disp_history, zorder=3, c='purple', linewidth=4)

    # Include average final displacement of atoms
    draw_box(ax_atoms, "Mean Displacement", mean_disp_history[-1])

    # Draw square lattice atoms
    draw_lattice_atoms(ax_atoms)
    set_ticks(ax_atoms)

    # Make plots square
    set_equal_aspect(ax_atoms)
    set_equal_aspect(ax_disps)

    # Display the atom on the square lattice
    plt.show()


def display_probability(atom_histories, show_gaussian=True):

    atoms_final = [atom_history[-1] for atom_history in atom_histories]
    x, y = zip(*atoms_final)
    k = gaussian_kde(np.vstack([x, y]))
    # xi, yi = np.mgrid[min(x):max(x):len(x)**0.5*1j,min(y):max(y):len(y)**0.5*1j]
    xi, yi = np.mgrid[-10:10:20j, -10:10:20j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))*100

    fig = plt.figure(figsize=(14, 7))
    ax_atoms = fig.add_subplot(121)

    # alpha=0.5 will make the plots semitransparent
    cmesh = ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=1, cmap=plt.get_cmap('Blues'))


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

    if show_gaussian:
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
        # ax2.contourf(x, y, gauss, cmap=plt.get_cmap('Blues'))
        ax2.pcolormesh(x, y, gauss, cmap=plt.get_cmap('Blues'))

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
        alpha_hist = 0.1
        alpha_final = 0.4
    else:
        alpha_hist = 1
        alpha_final = 1
    ax.plot(x_hist, y_hist, c='blue', alpha=alpha_hist, zorder=2.5)
    ax.scatter(x=x_final, y=y_final, alpha=alpha_final, zorder=3.5, c='blue')


def draw_disp_history(ax, atom_history, n_atoms=1):

    # Calculate displacements based of previous positions
    disp_history = calc_disp_history(atom_history)

    if n_atoms != 1:
        alpha = set_alpha(n_atoms)
    else:
        alpha = 1

    # Plot and format
    ax.plot(disp_history, alpha=alpha, c='purple', linewidth=4)


def set_equal_aspect(ax):
    x_lower, x_upper = ax.get_xlim()
    y_lower, y_upper = ax.get_ylim()
    ax.set_aspect((x_upper-x_lower)/(y_upper-y_lower))


def draw_box(ax, text, float_value):
    value_str = "{0:.2f}".format(float_value)
    ax.text(x=0.95, y=0.92,
            s="text" + "=" + value_str,
            horizontalalignment='right', verticalalignment='center',
            transform=ax_atoms.transAxes,
            bbox=dict(facecolor='white', alpha=1))


def set_ticks(ax):

    ax.grid()

    # Make x and y axis major ticks multiples of 5
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_locator(MultipleLocator(5))

    # Make x and y axis major tiock labels integers
    ax.yaxis.set_major_formatter(FormatStrFormatter('% d'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('% d'))

    # Turn off minor ticks
    ax.minorticks_off()


def calc_disp_history(atom_history):
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

