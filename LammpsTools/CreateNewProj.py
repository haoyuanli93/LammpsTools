import os
import shutil

import numpy as np


####################################################
#    Universal functions
####################################################
def getMoleculePositions(boxSizeA, moleculeNum, randomNumberSeed):
    """
    Create a random array representing
    the positions of water molecules in size a box

    :param boxSizeA:
    :param moleculeNum:
    :param randomNumberSeed: Force one to select a random number seed so that one won't forget
                                to use a different seed for a different simulation
    :return:
    """
    moleculeNum = int(moleculeNum)
    boxSizeA = float(boxSizeA)
    np.random.seed(randomNumberSeed)

    # First divides the whole space into several cubes according to the molecule number
    axisPartNum = int(np.cbrt(float(moleculeNum)) + 1)

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
    gridCoordinate = gridCoordinate[:moleculeNum, :]

    # Convert the grid coordinate to the molecule positions in A
    gridCoordinate *= (boxSizeA / float(axisPartNum))

    # Move the center to 0
    gridCoordinate -= (boxSizeA / 2.)

    # Purturb the water molecules
    gridCoordinate += np.random.rand(moleculeNum, 3) * 0.2

    return gridCoordinate


def getMoleculeNumber(densityGCm3, molarMass, boxSizeA):
    """
    IntermediateScatteringFunction the molecular number given the box size in A and density in g/cm3 and molar mass

    :param densityGCm3:
    :param molarMass:
    :param boxSizeA:
    :return:
    """
    # molar num
    NA = 6.0221408 * 0.1
    molecule_number = int((boxSizeA ** 3) * densityGCm3 / molarMass * NA)

    return molecule_number


def getBoxSizeA(densityGCm3, molarMass, molNum):
    """
    Derive a box size that is close to the molar mass and density in g/cm3
    for a give molecule number
    :param densityGCm3:
    :param molarMass:
    :param molNum:
    :return:
    """
    # molar num
    NA = 6.0221408 * 0.1
    volume = molNum / (densityGCm3 / molarMass * NA)

    return np.cbrt(volume)


#############################################################################
#      Sample specific functions
#############################################################################
def createSystemInfo(fileName, densityGCm3, boxSizeA, molFile, molName, molarMass, randomSeed):
    # IntermediateScatteringFunction the number of molecules to create
    molecule_number = getMoleculeNumber(densityGCm3=densityGCm3, molarMass=molarMass, boxSizeA=boxSizeA)

    # IntermediateScatteringFunction the coordinate of the molecules
    mol_coordinate = getMoleculePositions(boxSizeA=boxSizeA, moleculeNum=molecule_number,
                                          randomNumberSeed=randomSeed)

    with open(fileName, 'w') as data_file:
        data_file.write("write_once(\"Data Boundary\") { \n")
        data_file.write("{} {} xlo xhi \n".format(-boxSizeA / 2., boxSizeA / 2.))
        data_file.write("{} {} ylo yhi \n".format(-boxSizeA / 2., boxSizeA / 2.))
        data_file.write("{} {} zlo zhi \n".format(-boxSizeA / 2., boxSizeA / 2.))
        data_file.write("} \n")
        data_file.write("\n")

        data_file.write("write_once(\"In Init\") {\n")
        data_file.write("\n")
        data_file.write("units           real \n")
        data_file.write("boundary p p p\n")
        data_file.write("atom_style      full \n")
        data_file.write("pair_style      lj/cut/tip4p/long 1 2 1 1 0.1546485 8.5 \n")
        data_file.write("bond_style      harmonic \n")
        data_file.write("angle_style     harmonic \n")
        data_file.write("dihedral_style  none \n")
        data_file.write("improper_style  none \n")
        data_file.write("\n")
        data_file.write("kspace_style    pppm 1e-5 \n")
        data_file.write("}\n")
        data_file.write("\n")

        # data_file.write("# import the forcefield file\n")
        data_file.write("# import molecule building block file\n")
        data_file.write("import \"{}\" \n".format(molFile))
        data_file.write("\n")
        data_file.write("# create a single copy of this molecule at position 0,0,0\n")

        for mol_idx in range(molecule_number):
            data_file.write("mol{} = new {}.move({}, {}, {})\n".format(mol_idx,
                                                                       molName,
                                                                       mol_coordinate[mol_idx, 0],
                                                                       mol_coordinate[mol_idx, 1],
                                                                       mol_coordinate[mol_idx, 2],
                                                                       ))


def initializeLammpsScriptNPT(fileName, temperature, pressure, randomSeed):
    with open(fileName, 'w') as lammpsScript:
        lammpsScript.write("#   Load system information \n")
        lammpsScript.write("include \"system.in.init\" \n")
        lammpsScript.write("read_data \"system.data\" \n")
        lammpsScript.write("include \"system.in.settings\" \n")
        lammpsScript.write("\n")
        lammpsScript.write("# Define time\n")
        lammpsScript.write("timestep 0.25 \n")
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
        lammpsScript.write("#Define the thermo output\n")
        lammpsScript.write("thermo          10 \n")
        lammpsScript.write("thermo_style    custom step temp pe etotal press density \n")
        lammpsScript.write("\n")
        lammpsScript.write("#Define group\n")
        lammpsScript.write("group tip4p type  1  2 \n")
        lammpsScript.write("\n")
        lammpsScript.write("#Define the npt ensemble of the run\n")
        lammpsScript.write("fix therm all ave/time 1 10 10 v_P v_T v_rho file myThermo.txt \n")
        lammpsScript.write("fix fxnvt all nvt temp {} {} 4.0\n".format(temperature, temperature))
        lammpsScript.write("fix fRattleTIP4P tip4p rattle 0.0001 10 100 b 1 a 1 \n")
        # Specify the output thermo info
        lammpsScript.write("neigh_modify \n")
        lammpsScript.write("\n")
        lammpsScript.write("# Define the restart info\n")
        lammpsScript.write("restart 20000 water.restart.* \n")
        lammpsScript.write("run   20000 \n")


def createNewProj(projID, temperature=293.15, pressure=1.0, densityGCm3=0.789, molNum=1024, randomSeed=1000,
                  ensemble="NVT"):
    """
    This function aims to create a new project directly for simulation.

    """

    # Create new files contains the information of the system
    try:
        shutil.copytree("./systemInfo/", "./proj{}/".format(projID))
    except FileExistsError:
        print("This project exists. Pleasse use an new project ID.")
        return

    molarMass = 46.07
    # Create the system.lt file in the new project file
    createSystemInfo(fileName="./proj{}/system.lt".format(projID),
                     densityGCm3=densityGCm3,
                     boxSizeA=getBoxSizeA(densityGCm3=densityGCm3, molarMass=molarMass, molNum=int(molNum)),
                     molFile="tip4p2005.lt".format(projID),
                     molName="TIP4P2005",
                     molarMass=molarMass,
                     randomSeed=randomSeed)

    # Convert the moltemplate file into lammps file
    os.chdir("./proj{}/".format(projID))
    os.system("~/Software/moltemplate/moltemplate/scripts/moltemplate.sh system.lt")
    os.mkdir("./logFiles")
    os.mkdir("./output")
    os.chdir("../")

    # Create the initial lammps simulation script
    if ensemble == "NVT":
        initializeLammpsScriptNPT(fileName="./proj{}/miniRun.lmp".format(projID),
                                  temperature=temperature,
                                  pressure=pressure,
                                  randomSeed=randomSeed)
    else:
        print("The ensemble is not available yet.")
