import numpy as np

NA = 6.0221408e23
"""
The unit system is the following
length: angstrom
density: g/cm^3
"""


def density_gcm3_to_molcm3(density_gcm3, molar_mass_gmol):
    """

    :param density_gcm3: density of the sample in g/cm3
    :param molar_mass_gmol: The sample atomic mass measured in g/mol
    :return:
    """
    return density_gcm3 / molar_mass_gmol


def get_molecule_number(density_gcm3, molar_mass_gmol, box_size_A):
    """
    Get the molecular number given the box size in A and density in g/cm3 and molar mass

    :param density_gcm3:
    :param molar_mass_gmol:
    :param box_size_A:
    :return:
    """
    # molar num
    molar_density = density_gcm3_to_molcm3(density_gcm3=density_gcm3,
                                           molar_mass_gmol=molar_mass_gmol)
    molecule_number = int(((box_size_A / 1e8) ** 3) * molar_density * NA)

    return molecule_number


def get_molecule_number_1D(density_gcm3, molar_mass_gmol, box_size_A):
    """
    Get the molecule number and spacing per dimension

    :param density_gcm3:
    :param molar_mass_gmol:
    :param box_size_A:
    :return:
    """
    # molar num
    molar_density = density_gcm3_to_molcm3(density_gcm3=density_gcm3,
                                           molar_mass_gmol=molar_mass_gmol)
    molecule_number_1D = int(np.cbrt(((box_size_A / 1e8) ** 3) * molar_density * NA))
    molecule_number_3D = molecule_number_1D ** 3

    # Calculate the ratio to shrink the volume
    ratio = molecule_number_3D / NA / molar_density / ((box_size_A / 1e8) ** 3)
    # Get the new simulation  size and edge spacing
    box_size_new_A = np.cbrt(ratio) * box_size_A
    spacing = box_size_new_A / molecule_number_1D

    return molecule_number_3D, molecule_number_1D, box_size_new_A, spacing


def get_box_size_A(density_gcm3, molar_mass_gmol, mol_num):
    """
    Derive a box size that is close to the molar mass and density in g/cm3
    for a give molecule number
    :param density_gcm3:
    :param molar_mass_gmol:
    :param mol_num:
    :return:
    """
    volume_cm3 = mol_num / (density_gcm3 / molar_mass_gmol * NA)

    return np.cbrt(volume_cm3) * 1e8  # Convert the cm to A


# Parse the fluid info
def parseFluidInfo(fileName):
    fluidInfo = {
        "Temperature": np.loadtxt(fileName, usecols=0),
        "Pressure": np.loadtxt(fileName, usecols=1),
        "Density": np.loadtxt(fileName, usecols=2),
        "Volume": np.loadtxt(fileName, usecols=3),
        "Internal Energy": np.loadtxt(fileName, usecols=4),
        "Enthalpy": np.loadtxt(fileName, usecols=5),
        "Entropy": np.loadtxt(fileName, usecols=6),
        "Cv": np.loadtxt(fileName, usecols=7),
        "Cp": np.loadtxt(fileName, usecols=8),
        "Sound Spd": np.loadtxt(fileName, usecols=9),
        "Joule - Thomson": np.loadtxt(fileName, usecols=10),
        "Viscosity": np.loadtxt(fileName, usecols=11),
        "Therm.Cond.": np.loadtxt(fileName, usecols=12),
    }

    return fluidInfo