import numpy as np

NA = 6.0221408e23
"""
The unit system is the following
length: angstrom
density: g/cm^3
"""


###############################################
#   Basic utilities
###############################################
def density_gcm3_to_molcm3(density_gcm3, atomic_mass_gmol):
    """

    :param density_gcm3: density of the sample in g/cm3
    :param atomic_mass_gmol: The sample atomic mass measured in g/mol
    :return:
    """
    return density_gcm3 / atomic_mass_gmol


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
    axis_part_num = int(np.cbrt(float(molecule_num)) + 1)

    # Create a numpy array to represent this partition
    grid_coor = np.zeros((axis_part_num, axis_part_num, axis_part_num, 3), dtype=np.float64)
    grid_coor[:, :, :, 0] = np.arange(axis_part_num)[:, np.newaxis, np.newaxis]
    grid_coor[:, :, :, 1] = np.arange(axis_part_num)[np.newaxis, :, np.newaxis]
    grid_coor[:, :, :, 2] = np.arange(axis_part_num)[np.newaxis, np.newaxis, :]

    # Convert the 3D coordinate to 1D to randomly choose from it
    grid_coor = np.reshape(a=grid_coor, newshape=(axis_part_num ** 3, 3))

    # Shuffle the array and choose from it
    np.random.shuffle(grid_coor)

    # Choose the first several samples as the initial position of the molecules
    grid_coor = grid_coor[:molecule_num, :]

    # Convert the grid coordinate to the molecule positions in A
    grid_coor *= (box_size_A / float(axis_part_num))

    # Move the center to 0
    grid_coor -= (box_size_A / 2.)

    # Purturb the water molecules
    grid_coor += np.random.rand(molecule_num, 3) * 0.2

    return grid_coor


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
    volume_cm3 = mol_num / (density_g_cm3 / molar_mass * NA)

    return np.cbrt(volume_cm3) * 1e8  # Convert the cm to A


###############################################
#   Create a Bash script to submit the job
###############################################
def get_sbatch_file_cori(file_name, calculation_hour, account_name):
    with open(file_name, 'w') as data_file:
        data_file.write(
            "#!/bin/bash \n" +
            "#SBATCH --qos=regular \n" +
            "#SBATCH --time={}:00:00 \n".format(int(calculation_hour)) +
            "#SBATCH --nodes=2 \n" +
            "#SBATCH --constraint=knl \n" +
            "#SBATCH --job-name=md # Job name for allocation \n" +
            "#SBATCH --output=logFiles/%j.log # File to which STDOUT will be written, %j inserts jobid \n" +
            "#SBATCH --error=logFiles/%j.error # File to which STDERR will be written, %j inserts jobid \n"
        )

        if not (account_name is None):
            data_file.write("#SBATCH --account={} \n".format(account_name))

        data_file.write("module load lammps \n" +
                        "srun -n 136 -c 2 --cpu-bind=cores lmp_cori" +
                        " -in miniRun.lmp -log logFiles/mylog_$SLURM_JOB_ID.lammps \n")


###############################################
#   Create file for moltemplate
###############################################
def create_system_info(file_name,
                       density_g_cm3,
                       box_size_A,
                       molecule_file,
                       molecule_name,
                       molar_mass,
                       random_seed):
    # Get the number of molecules to create
    molecule_number = get_molecule_number(density_g_cm3=density_g_cm3,
                                          molar_mass=molar_mass,
                                          box_size_A=box_size_A)

    # Get the coordinate of the molecules
    mol_coordinate = get_molecule_positions(box_size_A=box_size_A,
                                            molecule_num=molecule_number,
                                            random_number_seed=random_seed)

    with open(file_name, 'w') as data_file:
        data_file.write(
            "write_once(\"Data Boundary\") { \n" +
            "{} {} xlo xhi \n".format(-box_size_A / 2., box_size_A / 2.) +
            "{} {} ylo yhi \n".format(-box_size_A / 2., box_size_A / 2.) +
            "{} {} zlo zhi \n".format(-box_size_A / 2., box_size_A / 2.) +
            "} \n" +
            "\n" +

            "write_once(\"In Init\") {\n" +
            "\n" +
            "units           real \n" +
            "boundary p p p\n" +
            "atom_style      full \n" +
            "}\n" +
            "\n" +

            # data_file.write("# import the forcefield file\n")
            "# import molecule building block file\n" +
            "import \"{}\" \n".format(molecule_file) +
            "\n" +
            "# create a single copy of this molecule at position 0,0,0\n"

        )

        for mol_idx in range(molecule_number):
            data_file.write("mol{} = new {}.move({}, {}, {})\n".format(mol_idx,
                                                                       molecule_name,
                                                                       mol_coordinate[mol_idx, 0],
                                                                       mol_coordinate[mol_idx, 1],
                                                                       mol_coordinate[mol_idx, 2],
                                                                       ))


def initializeLammpsScriptNVT(fileName, temperature, randomSeed,
                              saveAtomPosition=False, dump_num=100,
                              restart_num=5000, run_num=10000, getRDF=False):
    with open(fileName, 'w') as lammpsScript:
        lammpsScript.write("#   Load system information \n")
        lammpsScript.write("include \"system.in.init\" \n")
        lammpsScript.write("read_data \"system.data\" \n")
        lammpsScript.write("include \"system.in.settings\" \n")
        lammpsScript.write("\n")
        lammpsScript.write("# Define time\n")
        lammpsScript.write("timestep 0.5 \n")
        lammpsScript.write("\n")
        lammpsScript.write("# Define variables\n")
        lammpsScript.write("variable P equal press \n")
        lammpsScript.write("variable T equal temp \n")
        lammpsScript.write("variable rho equal density \n")
        lammpsScript.write("\n")
        lammpsScript.write("# Minimize the energy\n")
        lammpsScript.write("minimize 0.10 0.10 100000 100000 \n")
        lammpsScript.write("\n")
        lammpsScript.write("# Initialize the velocity\n")
        lammpsScript.write("velocity all create {}  {} \n".format(temperature, randomSeed))
        lammpsScript.write("run 0 \n")
        lammpsScript.write("velocity all scale {} \n".format(temperature))
        lammpsScript.write("\n")
        lammpsScript.write("#Define group\n")
        lammpsScript.write("group tip4p type  1  2 \n")
        lammpsScript.write("\n")
        lammpsScript.write("#Define the npt ensemble of the run\n")
        lammpsScript.write("fix therm all ave/time 1 10 10 v_P v_T v_rho file myThermo.txt \n")
        lammpsScript.write("fix fxnvt all nvt temp {} {} 10.0\n".format(temperature, temperature))
        lammpsScript.write("fix fRattleTIP4p tip4p rattle 0.0001 10 100 b 1 a 1 \n")

        # Get the radial distribution function
        if getRDF:
            lammpsScript.write("#Get rdf\n")
            lammpsScript.write("compute myRDF tip4p rdf 50 1 1 \n")
            lammpsScript.write("fix getRDF all ave/time 5 20 100 c_myRDF[*] file ./output/myRDF.rdf mode vector \n")
            lammpsScript.write("\n")

        # Get the output data
        if saveAtomPosition:
            lammpsScript.write("#Save atom positions\n")
            lammpsScript.write("dump 1 tip4p custom {} ./output/atomPos.* id type x y z\n".format(dump_num))
            lammpsScript.write("\n")

        # Specify the output thermo info
        lammpsScript.write("neigh_modify \n")
        lammpsScript.write("\n")
        lammpsScript.write("# Define the restart info\n")
        lammpsScript.write("restart {} ./output/myRestart.* \n".format(restart_num))
        lammpsScript.write("run  {} \n".format(run_num))
