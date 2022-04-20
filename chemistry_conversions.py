# chemistry_conversions.py
"""Tools for chemistry-related data conversions.
See data_processing.lift() and data_processing.unlift().

* mass2molar(): convert mass fractions to molar concentrations.
* molar2mass(): convert molar concentrations to mass fractions.
* specific_volume(): compute specific volume from pressure, temperature,
    and species mass fractions.
* temperature(): compute temperature from pressure, specific volume,
    and species molar concentrations.
"""
import config


# Input Validation ===========================================================

def _check_shapes(species, *args):
    """Ensure NUM_SPECIES species are passed in (mass fractions or molar
    concentrations) and that each argument has the same shape.
    """
    if len(species) != config.NUM_SPECIES:
        raise ValueError(f"{config.NUM_SPECIES} species required, "
                         f"got {len(species)}")
    _shape = species[0].shape
    for spc in species:
        if spc.shape != _shape:
            raise ValueError("species must all have same shape")
    for other in args:
        if other.shape != _shape:
            raise ValueError("inputs not aligned with species")


# Mass fractions to/from molar concentrations =================================

def mass2molar(masses, xi):
    """Convert mass fractions to molar concentrations following
    https://en.wikipedia.org/wiki/Molar_concentration.
    In the paper, mass fractions are denoted Y_l and molar concentrations
    are denoted c_l, so this is Y_l -> c_l.

    Parameters
    ----------
    masses : list of NUM_SPECIES (domain_size, num_snapshots) ndarrays
        Mass fractions for each of the chemical species.
    xi : (domain_size, num_snapshots) ndarray
        Specific volume of the mixture [m^3/kg]

    Returns
    -------
    molars : list of NUM_SPECIES (domain_size, num_snapshots) ndarrays
        Molar concentrations for each chemical species [mol/m^3].
    """
    # Check shapes.
    _check_shapes(masses, xi)

    # Do the conversion from mass fractions to molar concentrations.
    return [mass_fraction / (xi * molar_mass)
            for mass_fraction, molar_mass in zip(masses, config.MOLAR_MASSES)]


def molar2mass(molars, xi):
    """Convert molar concentrations to mass fractions following
    https://en.wikipedia.org/wiki/Molar_concentration.
    In the paper, molar concentrations are denoted c_l and mass fractions
    are  denoted Y_l, so this is c_l -> Y_l.

    Parameters
    ----------
    molars : list of NUM_SPECIES (domain_size, num_snapshots) ndarrays
        Molar concentrations for each chemical species [mol/m^3].
    xi : (domain_size, num_snapshots) ndarray
        Secific volume (1/density) [m^3/kg].

    Returns
    -------
    masses : list of NUM_SPECIES (domain_size, num_snapshots) ndarrays
        Mass fractions for each of the chemical species.
    """
    _check_shapes(molars, xi)

    # Do the conversion from molar concentrations to mass fractions.
    return [molar_conc * molar_mass * xi
            for molar_conc, molar_mass in zip(molars, config.MOLAR_MASSES)]


# Ideal gas law conversions ===================================================

def specific_gas_constant(masses):
    """Compute the specific gas constant for a given state of mass fractions.

    R_specific = R/M_avg    --- R : universal gas constant (8.314 J/mol K)

    1/M_avg = sum Y_i/M_i   --- Y_i: species mass fractions
                            --- M_i: corresponding molar mass (g/mol)

    Parameters
    ----------
    masses : list of NUM_SPECIES (domain_size, num_snapshots) ndarrays
        Mass fractions for each of the chemical species.

    Returns
    -------
    R_specific : (domain_size, num_snapshots) ndarray (WAS float)
        Specific gas constants J/(kg K) at each point in the space-time domain.
    """
    Mavg_inv = sum(massfrac / molarmass
                   for massfrac, molarmass in zip(masses, config.MOLAR_MASSES))
    return 1000 * config.R_UNIVERSAL * Mavg_inv


def specific_volume(p, T, masses):
    """Compute the specific volume.

    Parameters
    ----------
    p : (domain_size, num_snapshots) ndarray
        Pressure [Pa = kg/ms^2].
    T : (domain_size, num_snapshots) ndarray
        Temperature [K].
    masses : list of NUM_SPECIES (domain_size, num_snapshots) ndarrays
        Mass fractions for each of the chemical species.

    Returns
    -------
    xi : (domain_size, num_snapshots) ndarray
        Specific volume [m^3/kg], according to the ideal gas law.
    """
    # Check shapes
    _check_shapes(masses, p, T)

    # Compute the specific gass constant at each space-time point.
    R_specific = specific_gas_constant(masses)

    # Compute the density of the mixture (ideal gas law).
    return R_specific * T / p


def temperature(p, xi, molars):
    """Compute temperature from (predicted) states using the ideal gas law.

    Parameters
    ----------
    p : (domain_size, num_snapshots) ndarray
        Pressure [Pa = kg/ms^2].
    xi : (domain_size, num_snapshots) ndarray
        Specific volume (1/density). Called xi in the paper.
    molars : list of NUM_SPECIES (domain_size, num_snapshots) ndarrays
        Molar concentrations for each chemical species [mol/m^3].

    Returns
    -------
    T : (domain_size, num_snapshots) ndarray
        Temperature [K], according to the ideal gas law.
    """
    # Compute specific gas constant (shapes checked in molar2mass()).
    R_specific = specific_gas_constant(molar2mass(molars, xi))

    # Compute temperature (ideal gas law).
    return p * xi / R_specific
