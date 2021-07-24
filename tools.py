#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module serves to implement necessary code to simulate
diffusion.

This module has functions for displaying simulations.
"""

import matplotlib.pyplot as plt


def display_atom(atom):
    """Shows the position of an atom in a square lattice.

    Args:
        atom (list): x,y coordinate of atom in the lattice.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Place atom on the square lattice
    plt.scatter(atom[0], atom[1])

    # Set the limits of the x- and y-axes
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)

    # Add gridlines
    plt.grid()

    # Ensure that the axes look square
    ax.set_aspect('equal', adjustable='box')

    # Display the atom on the square lattice
    plt.show()
