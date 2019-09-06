# Ensembler - v0.2

[![CircleCI](https://circleci.com/gh/ccgcsms/Ensembler.svg?style=svg)](https://circleci.com/gh/ccgcsms/Ensembler)

## Description
This project tries to give users very easy to use and simple functionality to develop code for physical ensembles.

## Contents
*Potential functions
 Contains simple functions, that can be stocked together. 
 Also implementation of new potentials is very easy, as there only few functions that need to be overwritten.
 Examples: Harmonic Oscillator, Wave function, etc.. 
 Also different dimensionalities can be used.
** OneD
** TwoD
** ND

*Systems
This module is used to setup a simulation. It gets a potential, integrator and other parameters.

*Integrators
This module provides integrators for integrating potential functions. E.g. Monte Carlo, Velocity Verlet,...

*Visualization
This module contains predefined visualization a and animation functions.

*Ensembles
This module contains the replica exchange and conveyorbelt approaches.



## Current development
* rebuild Potentials more performant
* write test for all potentials
