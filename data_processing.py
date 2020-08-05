# data_processing.py
"""Tools for lifting and scaling data.

* lift(): transform GEMS variables to learning variables.
* unlift(): transform learning variables to GEMS variables.
* scale(): scale lifted data to the bounds set by config.SCALE_TO.
* unscale(): unscale scaled data to their original bounds.
"""
import logging
import numpy as np

import rom_operator_inference as roi

import chemistry_conversions as chem
from config import NUM_GEMSVARS, ROM_VARIABLES, NUM_ROMVARS, DOF, SCALE_TO


# Lifting Transformation =====================================================

def lift(data):
    """Transform GEMS data to the lifted variables,

        [P, v_x, v_y, T, Y_CH4, Y_O2, Y_H2O, Y_CO2]
        -->
        [P, v_x, v_y, T, xi, c_CH4, c_O2, c_H2O, c_CO2].

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
    P, vx, vy, T, Y_CH4, Y_O2, Y_H2O, Y_CO2 = np.split(data, NUM_GEMSVARS)
    masses = [Y_CH4, Y_O2, Y_H2O, Y_CO2]

    # Compute specific volume.
    xi = chem.specific_volume(P, T, masses)

    # Compute molar concentrations.
    molars = chem.mass2molar(masses, xi)

    # Put the lifted data together.
    return np.concatenate([P, vx, vy, T, xi] + molars)


def unlift(data):
    """Transform the learning variables back to the GEMS variables,

        [P, v_x, v_y, T, xi, c_CH4, c_O2, c_H2O, c_CO2]
        -->
        [P, v_x, v_y, T, Y_CH4, Y_O2, Y_H2O, Y_CO2]

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
    P, vx, vy, T, xi, c_CH4, c_O2, c_H2O, c_CO2 = np.split(data, NUM_ROMVARS)
    molars = [c_CH4, c_O2, c_H2O, c_CO2]

    # Compute mass fractions.
    masses = chem.molar2mass(molars, xi)

    # Put the unlifted data together.
    return np.concatenate([P, vx, vy, T] + masses)


# Variable getting / setting ==================================================

def _varslice(varname, datasize):
    """Get the slice where a specified variable is found in the given data.

    Parameters
    ----------
    datasize : int
        The number of rows (2D) or entries (1D) of data, e.g., data.shape[0].
        Must be a multiple of config.NUM_ROMVARS.
    
    varname : str
        An entry of config.ROM_VARIABLES indicating the variable to get/set.
    
    Returns
    -------
    s : slice
        A slice object for accessing the specified variable
    """
    varindex = ROM_VARIABLES.index(varname)
    chunksize, remainder = divmod(datasize, NUM_ROMVARS)
    if remainder != 0:
        raise ValueError("data cannot be split evenly"
                         f" into {NUM_ROMVARS} chunks")
    return slice(varindex*chunksize, (varindex+1)*chunksize)
    

def getvar(varname, data):
    """Extract the specified variable from the given data."""
    return data[_varslice(varname, data.shape[0])]


# MinMax scaling / unscaling ==================================================

def scale(data, scales=None, variables=None):
    """Scale data *IN-PLACE* by variable, meaning every chunk of DOF
    consecutive rows is scaled separately. Thus, DOF / data.shape[0] must be
    an integer.

    If `scales` is provided, variable i is scaled from the interval
    [scales[i,0], scales[i,1]] to [scales[i,2], scales[i,3]].
    Otherwise, the scaling is learned from the data.

    Scaling algorithm follows sklearn.preprocessing.MinMaxScaler.

    Parameters
    ----------
    data : (num_variables*DOF, num_snapshots) ndarray
        The dataset to be scaled.

    scales : (NUM_ROMVARS, 4) ndarray
        The before-and-after minimum and maximum of each variable. That is,
        scales[i] = [min(raw_variable i),    max(raw_variable i),       # from
                     min(scaled_variable i), max(scaled_variable i)].   # to
        Scaling sends [scales[i,0],scales[i,1]] -> [scales[i,2],scales[i,3]].
        If None, learn the scaling from the data and config.SCALE_TO as
        scales[i] = [min(raw_variable i),  max(raw_variable i),         # from
                     config.SCALE_TO[i,0], config.SCALE_TO[i,1]]        # to

    variables : list(str)
        List of variables to scale, a subset of config.ROM_VARIABLES.
        This argument can only be given when `scales` is provided as well.
        This also requires `data.shape[0]` to be divisible by `len(variables)`.

    Returns
    -------
    scaled_data : (num_variables*DOF, num_snapshots)
        The scaled data.

    scales : (NUM_ROMVARS, 2) ndarray
        The minimum and maximum of each variable.
    """
    # Determine whether learning the scaling transformation is needed.
    learning = (scales is None)
    if learning:
        if variables is not None:
            raise ValueError("scale=None only valid for variables=None")
        scale_from = np.empty((NUM_ROMVARS, 2), dtype=np.float)
        scale_to = SCALE_TO
        means = np.empty(NUM_ROMVARS)
    else:
        # Validate the scales.
        _shape = (NUM_ROMVARS, 4)
        if scales.shape != _shape:
            raise ValueError(f"`scales` must have shape {_shape}")
        scale_from, scale_to = np.split(scales, 2, axis=1)

    # Parse the variables.
    if variables is None:
        variables = ROM_VARIABLES
    elif isinstance(variables, str):
        variables = [variables]
    varindices = [ROM_VARIABLES.index(v) for v in variables]

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
            means[i] = data[s].mean()
            if variables[i] in ["vx", "vy"]:
                maxv = np.abs(data[s]).max()
                scale_from[i] = (-maxv, maxv)
                data[s] /= maxv
            else:
                data[s], _, scale_from[i] = roi.pre.scale(data[s], scale_to[i])
        else:
            if variables[i] in ["vx", "vy"]:
                data[s] /= scales[vidx,1]
            else:
                data[s] = roi.pre.scale(data[s],
                                        scale_to[vidx], scale_from[vidx])

    # Report info on the learned scaling.
    if learning:
        scales = np.concatenate((scale_from, scale_to), axis=1)
        sep = '|'.join(['-'*12]*3)
        report = f"""\nLearned new Min-Max scaling
                        Min     |    Mean    |     Max
                    {sep}
    Pressure        {scales[0,0]:<12.3e}|{means[0]:^12.3e}|{scales[0,1]:>12.3e}
                    {sep}
    x-velocity      {scales[1,0]:<12.3f}|{means[1]:^12.3f}|{scales[1,1]:>12.3f}
                    {sep}
    y-velocity      {scales[2,0]:<12.3f}|{means[2]:^12.3f}|{scales[2,1]:>12.3f}
                    {sep}
    Temperature     {scales[3,0]:<12.3e}|{means[3]:^12.3e}|{scales[3,1]:>12.3e}
                    {sep}
    Specific Volume {scales[4,0]:<12.3f}|{means[4]:^12.3f}|{scales[4,1]:>12.3f}
                    {sep}
    CH4 molar       {scales[5,0]:<12.3f}|{means[5]:^12.3f}|{scales[5,1]:>12.3f}
                    {sep}
    O2  molar       {scales[6,0]:<12.3f}|{means[6]:^12.3f}|{scales[6,1]:>12.3f}
                    {sep}
    H2O molar       {scales[8,0]:<12.3f}|{means[8]:^12.3f}|{scales[8,1]:>12.3f}
                    {sep}
    CO2 molar       {scales[7,0]:<12.3f}|{means[7]:^12.3f}|{scales[7,1]:>12.3f}
                    {sep}"""
        logging.info(report)

    return data, scales


def unscale(data, scales, variables=None):
    """Unscale data *IN-PLACE* by variable, meaning every chunk of DOF
    consecutive rows is unscaled separately. Thus, DOF / data.shape[0] must be
    an integer. Variable i is assumed to have been previously scaled from
    [scales[i,0], scales[i,1]] to [scales[i,2], scales[i,3]].

    Parameters
    ----------
    data : (num_variables*dof, num_snapshots) ndarray
        The dataset to be unscaled.

    scales : (NUM_ROMVARS, 4) ndarray
        The before-and-after minimum and maximum of each variable. That is,
        scales[i] = [min(raw_variable i),    max(raw_variable i),       # to
                     min(scaled_variable i), max(scaled_variable i)].   # from
        UNscaling sends [scales[i,0],scales[i,1]] <- [scales[i,2],scales[i,3]].

    variables : list(str)
        List of variables to scale, a subset of config.ROM_VARIABLES.
        This argument can only be given when `scales` is provided as well.
        This also requires `data.shape[0]` to be divisible by `len(variables)`.

    Returns
    -------
    unscaled_data : (num_variables*dof, num_snapshots)
        The unscaled data.
    """
    # Validate the scales.
    _shape = (NUM_ROMVARS, 4)
    if scales.shape != _shape:
        raise ValueError(f"`scales` must have shape {_shape}")
    scale_from, scale_to = np.split(scales, 2, axis=1)

    # Parse the variables.
    if variables is None:
        variables = ROM_VARIABLES
    elif isinstance(variables, str):
        variables = [variables]
    varindices = [ROM_VARIABLES.index(v) for v in variables]

    # Make sure the data can be split correctly by variable.
    nchunks = len(variables)
    chunksize, remainder = divmod(data.shape[0], nchunks)
    if remainder != 0:
        raise ValueError("data to unscale cannot be split"
                         f" evenly into {nchunks} chunks")

    # Do the unscaling by variable.
    for i,vidx in enumerate(varindices):
        s = slice(i*chunksize,(i+1)*chunksize)
        if variables[i] in ["vx", "vy"]:
            data[s] *= scale_from[vidx,1]
        else:
            data[s] = roi.pre.scale(data[s], scale_from[vidx], scale_to[vidx])

    return data
