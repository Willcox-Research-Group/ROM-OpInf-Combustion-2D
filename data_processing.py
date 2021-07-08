# data_processing.py
"""Tools for lifting and scaling data.

* lift(): transform GEMS variables to learning variables.
* unlift(): transform learning variables to GEMS variables.
* scale(): scale lifted data to the bounds set by config.SCALE_TO.
* unscale(): unscale scaled data to their original bounds.
"""
import logging
import numpy as np

import config
import chemistry_conversions as chem
import time
time.perf_counter()


# Lifting Transformation =====================================================

def lift(data):
    """Transform GEMS data to the lifted variables,

        [p, v_x, v_y, T, Y_CH4, Y_O2, Y_H2O, Y_CO2]
        -->
        [p, v_x, v_y, T, xi, c_CH4, c_O2, c_H2O, c_CO2].

    Parameters
    ----------
    data : (NUM_GEMSVARS*dof, num_snapshots) ndarray
        Unscaled, untransformed GEMS data.

    Returns
    -------
    lifted_data : (NUM_ROMVARS*dof, num_snapshots) ndarray
        Nonscaled, lifted data.
    """
    # Unpack the GEMS data.
    p, vx, vy, T, Y_CH4, Y_O2, Y_H2O, Y_CO2 = np.split(data,
                                                       config.NUM_GEMSVARS)
    masses = [Y_CH4, Y_O2, Y_H2O, Y_CO2]

    # Compute specific volume.
    xi = chem.specific_volume(p, T, masses)

    # Compute molar concentrations.
    molars = chem.mass2molar(masses, xi)

    # Put the lifted data together.
    return np.concatenate([p, vx, vy, T, xi] + molars)


def unlift(data):
    """Transform the learning variables back to the GEMS variables,

        [p, v_x, v_y, T, xi, c_CH4, c_O2, c_H2O, c_CO2]
        -->
        [p, v_x, v_y, T, Y_CH4, Y_O2, Y_H2O, Y_CO2]

    Parameters
    ----------
    data : (NUM_ROMVARS*dof, num_snapshots) ndarray
        Nonscaled, lifted data.

    Returns
    -------
    unlifed_data : (NUM_GEMSVARS*dof, num_snapshots) ndarray
        Unscaled, untransformed GEMS data.
    """
    # Unpack the lifted data.
    p, vx, vy, T, xi, c_CH4, c_O2, c_H2O, c_CO2 = np.split(data,
                                                           config.NUM_ROMVARS)
    molars = [c_CH4, c_O2, c_H2O, c_CO2]

    # Compute mass fractions.
    masses = chem.molar2mass(molars, xi)

    # Put the unlifted data together.
    return np.concatenate([p, vx, vy, T] + masses)


# Variable getting / setting ==================================================

def _varslice(varname, datasize):
    """Get the slice where a specified variable is found in the given data.

    Parameters
    ----------
    datasize : int
        Number of rows (2D) or entries (1D) of data, e.g., data.shape[0].
        Must be a multiple of config.NUM_ROMVARS.

    varname : str
        An entry of config.ROM_VARIABLES indicating the variable to get/set.

    Returns
    -------
    s : slice
        A slice object for accessing the specified variable
    """
    varindex = config.ROM_VARIABLES.index(varname)
    chunksize, remainder = divmod(datasize, config.NUM_ROMVARS)
    if remainder != 0:
        raise ValueError("data cannot be split evenly"
                         f" into {config.NUM_ROMVARS} chunks")
    return slice(varindex*chunksize, (varindex+1)*chunksize)


def getvar(varname, data):
    """Extract the specified variable from the given data."""
    return data[_varslice(varname, data.shape[0])]


# MinMax scaling / unscaling ==================================================

def scale(data, scales=None, variables=None):
    """Scale data *IN-PLACE* by variable, meaning every chunk of DOF
    consecutive rows is scaled separately. Thus, DOF / data.shape[0] must be
    an integer.

    If `scales` is provided, variable i is scaled as
    new_variable[i] = (raw_variable[i] - scales[i,0]) / scales[i,1].
    Otherwise, the scaling is learned from the data.

    Parameters
    ----------
    data : (num_variables*DOF, num_snapshots) ndarray
        Dataset to be scaled.

    scales : (NUM_ROMVARS, 2) ndarray or None
        Shifting and scaling factors. If None, learn the factors from the data:
            scales[i] = [shift, max(abs(raw_variable[i] - shift))],
        where shift = mean(raw_variable[i]) for pressure, temperature, and
        specific volume, and shift = 0 for the other variables.

    variables : list(str)
        List of variables to scale, a subset of config.ROM_VARIABLES.
        This argument can only be given when `scales` is provided as well.
        This also requires `data.shape[0]` to be divisible by `len(variables)`.

    Returns
    -------
    scaled_data : (num_variables*DOF, num_snapshots)
        Scaled data.

    scales : (NUM_ROMVARS, 2) ndarray
        Shifting and dilation factors used to scale the data.
    """
    # Determine whether learning the scaling transformation is needed.
    learning = (scales is None)
    if learning:
        if variables is not None:
            raise ValueError("scale=None only valid for variables=None")
        scales = np.empty((config.NUM_ROMVARS, 2), dtype=np.float)
    else:
        # Validate the scales.
        _shape = (config.NUM_ROMVARS, 2)
        if scales.shape != _shape:
            raise ValueError(f"`scales` must have shape {_shape}")

    # Parse the variables.
    if variables is None:
        variables = config.ROM_VARIABLES
    elif isinstance(variables, str):
        variables = [variables]
    varindices = [config.ROM_VARIABLES.index(v) for v in variables]

    # Make sure the data can be split correctly by variable.
    nchunks = len(variables)
    chunksize, remainder = divmod(data.shape[0], nchunks)
    if remainder != 0:
        raise ValueError("data to scale cannot be split"
                         f" evenly into {nchunks} chunks")

    # Do the scaling by variable.
    for i,vidx in enumerate(varindices):
        s = slice(i*chunksize,(i+1)*chunksize)
        if learning:
            assert i == vidx
            if variables[i] in ["p", "T", "xi"]:
                scales[vidx,0] = np.mean(data[s])
                shifted = data[s] - scales[vidx,0]
            else:
                scales[vidx,0] = 0
                shifted = data[s]
            scales[vidx,1] = np.abs(shifted).max()
            data[s] = shifted / scales[vidx,1]
        else:
            data[s] = (data[s] - scales[vidx,0]) / scales[vidx,1]

    # Report info on the learned scaling.
    if learning:
        sep = '|'.join(['-'*12]*2)
        report = f"""Learned new scaling
                       Shift    |    Denom
                    {sep}
    Pressure        {scales[0,0]:<12.3e}|{scales[0,1]:>12.3e}
                    {sep}
    x-velocity      {scales[1,0]:<12.3f}|{scales[1,1]:>12.3f}
                    {sep}
    y-velocity      {scales[2,0]:<12.3f}|{scales[2,1]:>12.3f}
                    {sep}
    Temperature     {scales[3,0]:<12.3e}|{scales[3,1]:>12.3e}
                    {sep}
    Specific Volume {scales[4,0]:<12.3f}|{scales[4,1]:>12.3f}
                    {sep}
    CH4 molar       {scales[5,0]:<12.3f}|{scales[5,1]:>12.3f}
                    {sep}
    O2  molar       {scales[6,0]:<12.3f}|{scales[6,1]:>12.3f}
                    {sep}
    H2O molar       {scales[8,0]:<12.3f}|{scales[8,1]:>12.3f}
                    {sep}
    CO2 molar       {scales[7,0]:<12.3f}|{scales[7,1]:>12.3f}
                    {sep}"""
        logging.info(report)

    return data, scales


def unscale(data, scales, variables=None):
    """Unscale data *IN-PLACE* by variable, meaning every chunk of DOF
    consecutive rows is unscaled separately. Thus, DOF / data.shape[0] must be
    an integer. Variable i is assumed to have been previously scaled by
    variable[i] = (old_variable[i] - scales[i,0]) / scales[i,1].

    Parameters
    ----------
    data : (num_variables*dof, num_snapshots) ndarray
        Dataset to be unscaled.

    scales : (NUM_ROMVARS, 2) ndarray
        Shifting and scaling factors. UNscaling is given by
        new_variable[i] = (variable[i] * scales[i,1]) + scales[i,0].

    variables : list(str)
        List of variables to scale, a subset of config.ROM_VARIABLES.
        This requires `data.shape[0]` to be divisible by `len(variables)`.

    Returns
    -------
    unscaled_data : (num_variables*dof, num_snapshots)
        Unscaled data.
    """
    # Validate the scales.
    _shape = (config.NUM_ROMVARS, 2)
    if scales.shape != _shape:
        raise ValueError(f"`scales` must have shape {_shape}")
    scale_from, scale_to = np.split(scales, 2, axis=1)

    # Parse the variables.
    if variables is None:
        variables = config.ROM_VARIABLES
    elif isinstance(variables, str):
        variables = [variables]
    varindices = [config.ROM_VARIABLES.index(v) for v in variables]

    # Make sure the data can be split correctly by variable.
    nchunks = len(variables)
    chunksize, remainder = divmod(data.shape[0], nchunks)
    if remainder != 0:
        raise ValueError("data to unscale cannot be split"
                         f" evenly into {nchunks} chunks")

    # Do the unscaling by variable.
    for i,vidx in enumerate(varindices):
        s = slice(i*chunksize,(i+1)*chunksize)
        data[s] = (data[s] * scales[vidx,1]) + scales[vidx,0]

    return data
