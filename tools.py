#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module serves to implement necessary code to simulate
diffusion.

This module has functions for displaying simulations.
"""

import numpy as np
import random
from matplotlib import cm
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import (MultipleLocator,
                               MaxNLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


sim_bounds = 10


def display_atom(atom, atom_history=None, disp_history=None):
    """Shows the position of an atom in a square lattice.

    Args:
        atom (list): x,y coordinate of atom in the lattice.
    """

    if disp_history is None:
        fig = plt.figure(figsize=(14, 7))
        ax1 = fig.add_subplot('121')
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        ax2.plot(disp_history)
        x0,x1 = ax2.get_xlim()
        y0,y1 = ax2.get_ylim()
        ax2.set_aspect((x1-x0)/(y1-y0))

    # Place atom on the square lattice. Adjust zorder to enusure atom
    # is on top of grid lines
    x, y = atom
    ax1.scatter(x, y, zorder=3.0)
    ax1.arrow(0, 0, x, y, width=0.2, length_includes_head=True, color=(0,0,0), edgecolor=(0,0,0), zorder=3.5)

    disp = (x**2 + y**2)**0.5
    ax1.text(0.95, 0.92, "Displacement = {0:.2f}".format(disp), horizontalalignment='right', verticalalignment='center', transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=1))

    if atom_history is not None:
        hist_x, hist_y = zip(*atom_history)
        ax1.plot(hist_x, hist_y, alpha=0.5, zorder=2.5)

    # Set the limits of the x- and y-axes
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)

    # Add gridlines
    ax1.grid()

    # Ensure that the axes look square
    ax1.set_aspect('equal', adjustable='box')

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
    ax1.minorticks_off()

    # Display the atom on the square lattice
    plt.show()


def display_atoms(atoms, atom_histories, disp_histories):
    """Shows the position of an atom in a square lattice.

    Args:
        atom (list): x,y coordinate of atom in the lattice.
    """

    mean_disp = 0
    n_atoms = len(atoms)
    m_steps = len(disp_histories[0])
    mean_disp_sq_history = [0]*m_steps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    for atom, atom_history, disp_history in zip(atoms, atom_histories, disp_histories):
        # Place atom on the square lattice. Adjust zorder to enusure atom
        # is on top of grid lines
        x, y = atom
        ax1.scatter(x, y, zorder=3.0)
        mean_disp += (x**2 + y**2)**0.5 / n_atoms
        hist_x, hist_y = zip(*atom_history)
        ax1.plot(hist_x, hist_y, alpha=0.5, zorder=2.5)
        for i, disp in enumerate(disp_history):
            mean_disp_sq_history[i] += disp**2/n_atoms


    ax2.plot(mean_disp_sq_history)
    x0,x1 = ax2.get_xlim()
    y0,y1 = ax2.get_ylim()
    ax2.set_aspect((x1-x0)/(y1-y0))

    # Set the limits of the x- and y-axes
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)

    # Add gridlines
    ax1.grid()

    # Ensure that the axes look square
    ax1.set_aspect('equal', adjustable='box')

    ax1.text(0.95, 0.92, "Mean Displacement = {0:.2f}".format(mean_disp), horizontalalignment='right', verticalalignment='center', transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=1))

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
    ax1.minorticks_off()

    # Display the atom on the square lattice
    plt.show()


def display_probability(atoms_final, compare_gaussian=True, num_jumps=None):

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

    x0,x1 = ax1.get_xlim()
    y0,y1 = ax1.get_ylim()
    ax1.set_aspect((x1-x0)/(y1-y0))

    ax1.yaxis.set_major_locator(MultipleLocator(5))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('% d'))
    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('% d'))

    # fig.colorbar(cmesh, ax=ax1)
    # add_colorbar(cmesh)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cmesh,cax=cax)

    if compare_gaussian:
        ax2 = fig.add_subplot(122)
        mu = 0
        sigma = (num_jumps/3)**0.5 * 1
        # sigma = 1

        # Initializing value of x-axis and y-axis
        # in the range -1 to 1
        x, y = np.meshgrid(np.linspace(-10,10,100), np.linspace(-10,10,100))
        dst = np.sqrt(x*x+y*y)

        # Calculating Gaussian array
        gauss = np.exp(-( (dst-mu)**2 / ( 2.0 * sigma**2 ) ) ) * 100

        # alpha=0.5 will make the plots semitransparent
        cont_mesh = ax2.contourf(x, y, gauss)

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