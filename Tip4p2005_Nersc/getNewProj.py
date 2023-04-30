import argparse
import os
import shutil
import sys

import LammpsTools.util

sys.path.append(r"/mnt/c/Users/haoyuan/Documents/GitHub/LammpsTools")

parser = argparse.ArgumentParser()
parser.add_argument("temperature", help="temperature in K", type=float)
parser.add_argument("randomSeed", help="Reduced temperature", type=int)
parser.add_argument("runNum", help="number of time steps to run", type=int)
parser.add_argument("restartNum", help="number of runs before save a restart point", type=int)
parser.add_argument("dumpNum", help="number of runs before dump a position file", type=int)
parser.add_argument("density", help="The density of the sample", type=float)
parser.add_argument("getRDF", help="Flag for whether calculate the RDF.", type=bool)
parser.add_argument("moleculeNum", help="molecule number to simulate.", type=int)

args = parser.parse_args()


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


def createNewProj(proj_ID, temperature=293.15, density_g_cm3=1.0, molecule_num=1024, random_seed=1000, ):
    # Create new files contains the information of the system
    try:
        shutil.copytree(r"/mnt/c/Users/haoyuan/Documents/GitHub/LammpsTools/LammpsTools/auxNotebooks",
                        "./proj{}/".format(proj_ID))
    except FileExistsError:
        print("This project exists. Pleasse use an new project ID.")
        return

    molar_mass = 18.01528
    # Create the system.lt file in the new project file
    util.create_system_info(file_name="./proj{}/system.lt".format(proj_ID),
                            density_g_cm3=density_g_cm3,
                            box_size_A=LammpsTools.util.get_box_size_A(density_gcm3=density_g_cm3,
                                                                       molar_mass_gmol=molar_mass,
                                                                       mol_num=int(molecule_num)),
                            molecule_file=r"/mnt/c/Users/haoyuan/Documents/GitHub/LammpsTools/moleculeZoo/tip4p2005.lt",
                            molecule_name="TIP4P2005",
                            molar_mass=molar_mass,
                            random_seed=random_seed,
                            )

    # Convert the moltemplate file into lammps file
    os.chdir("./proj{}/".format(proj_ID))
    os.system("~/github/moltemplate/moltemplate/scripts/moltemplate.sh system.lt")
    os.mkdir("./logFiles")
    os.mkdir("./output")
    os.chdir("../")

    # Create the initial lammps simulation script
    initializeLammpsScriptNVT(fileName="./proj{}/miniRun.lmp".format(proj_ID),
                              temperature=temperature,
                              randomSeed=random_seed,
                              getRDF=True,
                              saveAtomPosition=True,
                              dump_num=args.dumpNum,
                              restart_num=args.restartNum,
                              run_num=args.runNum
                              )

    # Create the sbatch file to submit the job
    util.get_sbatch_file_cori(file_name="./proj{}/submit.mini".format(proj_ID),
                              calculation_hour=24,
                              account_name='m4070')


# Actually create the new system
proj_id = "_{:.2f}C_{:.2f}gcm3".format(args.temperature - 273.15, args.density)
proj_id = proj_id.replace('.', 'p')

createNewProj(proj_ID=proj_id,
              temperature=args.temperature,
              density_g_cm3=args.density,
              molecule_num=args.moleculeNum,
              random_seed=args.randomSeed,
              )

os.system("tar -zvcf proj{}.gz proj{}".format(proj_id, proj_id))
