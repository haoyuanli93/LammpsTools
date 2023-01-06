import numpy as np

NA = 6.0221408 * 0.1


def get_molecule_positions(box_size_A, molecule_num, random_number_seed):
    """
    Create a random array representing
    the positions of water molecules in size a box

    :param box_size_A:
    :param molecule_num:
    :param random_number_seed: Force one to select a random number seed so that one won't forget
                                to use a different seed for a different simulation
    :return:
    """
    molecule_num = int(molecule_num)
    box_size_A = float(box_size_A)
    np.random.seed(random_number_seed)

    # First divides the whole space into several cubes according to the molecule number
    axisPartNum = int(np.cbrt(float(molecule_num)) + 1)

    # Create a numpy array to represent this partition
    gridCoordinate = np.zeros((axisPartNum, axisPartNum, axisPartNum, 3), dtype=np.float64)
    gridCoordinate[:, :, :, 0] = np.arange(axisPartNum)[:, np.newaxis, np.newaxis]
    gridCoordinate[:, :, :, 1] = np.arange(axisPartNum)[np.newaxis, :, np.newaxis]
    gridCoordinate[:, :, :, 2] = np.arange(axisPartNum)[np.newaxis, np.newaxis, :]

    # Convert the 3D coordinate to 1D to randomly choose from it
    gridCoordinate = np.reshape(a=gridCoordinate, newshape=(axisPartNum ** 3, 3))

    # Shuffle the array and choose from it
    np.random.shuffle(gridCoordinate)

    # Choose the first several samples as the initial position of the molecules
    gridCoordinate = gridCoordinate[:molecule_num, :]

    # Convert the grid coordinate to the molecule positions in A
    gridCoordinate *= (box_size_A / float(axisPartNum))

    # Move the center to 0
    gridCoordinate -= (box_size_A / 2.)

    # Purturb the water molecules
    gridCoordinate += np.random.rand(molecule_num, 3) * 0.2

    return gridCoordinate


def get_molecule_number(density_g_cm3, molar_mass, box_size_A):
    """
    Get the molecular number given the box size in A and density in g/cm3 and molar mass

    :param density_g_cm3:
    :param molar_mass:
    :param box_size_A:
    :return:
    """
    # molar num
    NA = 6.0221408 * 0.1
    molecule_number = int((box_size_A ** 3) * density_g_cm3 / molar_mass * NA)

    return molecule_number


def get_box_size_A(density_g_cm3, molar_mass, mol_num):
    """
    Derive a box size that is close to the molar mass and density in g/cm3
    for a give molecule number
    :param density_g_cm3:
    :param molar_mass:
    :param mol_num:
    :return:
    """
    volume = mol_num / (density_g_cm3 / molar_mass * NA)

    return np.cbrt(volume)


def create_system_info(file_name, density_g_cm3, box_size_A, molecule_file, molecule_name, molar_mass, random_seed):
    # Get the number of molecules to create
    molecule_number = get_molecule_number(density_g_cm3=density_g_cm3,
                                          molar_mass=molar_mass,
                                          box_size_A=box_size_A)

    # Get the coordinate of the molecules
    mol_coordinate = get_molecule_positions(box_size_A=box_size_A,
                                            molecule_num=molecule_number,
                                            random_number_seed=random_seed)

    with open(file_name, 'w') as data_file:
        data_file.write("write_once(\"Data Boundary\") { \n")
        data_file.write("{} {} xlo xhi \n".format(-box_size_A / 2., box_size_A / 2.))
        data_file.write("{} {} ylo yhi \n".format(-box_size_A / 2., box_size_A / 2.))
        data_file.write("{} {} zlo zhi \n".format(-box_size_A / 2., box_size_A / 2.))
        data_file.write("} \n")
        data_file.write("\n")

        data_file.write("write_once(\"In Init\") {\n")
        data_file.write("\n")
        data_file.write("units           real \n")
        data_file.write("boundary p p p\n")
        data_file.write("atom_style      full \n")
        data_file.write("}\n")
        data_file.write("\n")

        # data_file.write("# import the forcefield file\n")
        data_file.write("# import molecule building block file\n")
        data_file.write("import \"{}\" \n".format(molecule_file))
        data_file.write("\n")
        data_file.write("# create a single copy of this molecule at position 0,0,0\n")

        for mol_idx in range(molecule_number):
            data_file.write("mol{} = new {}.move({}, {}, {})\n".format(mol_idx,
                                                                       molecule_name,
                                                                       mol_coordinate[mol_idx, 0],
                                                                       mol_coordinate[mol_idx, 1],
                                                                       mol_coordinate[mol_idx, 2],
                                                                       ))


def get_sbatch_file_cori(file_name, calculation_hour, account_name):
    with open(file_name, 'w') as data_file:
        data_file.write("#!/bin/bash \n")
        data_file.write("#SBATCH --qos=regular \n")
        data_file.write("#SBATCH --time={}:00:00 \n".format(int(calculation_hour)))
        data_file.write("#SBATCH --nodes=2 \n")
        data_file.write("#SBATCH --constraint=knl \n")
        data_file.write("#SBATCH --job-name=md # Job name for allocation \n")
        data_file.write("#SBATCH --output=logFiles/%j.log # File to which STDOUT will be written, %j inserts jobid \n")
        data_file.write("#SBATCH --error=logFiles/%j.error # File to which STDERR will be written, %j inserts jobid \n")

        if not (account_name is None):
            data_file.write("#SBATCH --account={} \n".format(account_name))

        data_file.write("module load lammps \n")
        data_file.write(
            "srun -n 136 -c 2 --cpu-bind=cores lmp_cori -in miniRun.lmp -log logFiles/mylog_$SLURM_JOB_ID.lammps \n")
