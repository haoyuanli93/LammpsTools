from LammpsTools import util
import shutil
import os

molar_mass = 18.01528


# Create new submission script
def get_sbatch_file_cori(file_name,
                         nodeNum=1,
                         thread_per_cpu=2,
                         thread_num=68,
                         calculation_hour=1,
                         account_name='m4070'):
    with open(file_name, 'w') as data_file:
        data_file.write(
            "#!/bin/bash \n" +
            "#SBATCH --qos=regular \n" +
            "#SBATCH --time={}:00:00 \n".format(int(calculation_hour)) +
            "#SBATCH --nodes={} \n".format(nodeNum) +
            "#SBATCH --constraint=knl \n" +
            "#SBATCH --job-name=md # Job name for allocation \n" +
            "#SBATCH --output=logFiles/%j.log # File to which STDOUT will be written, %j inserts jobid \n" +
            "#SBATCH --error=logFiles/%j.error # File to which STDERR will be written, %j inserts jobid \n"
        )

        if not (account_name is None):
            data_file.write("#SBATCH --account={} \n".format(account_name))
        data_file.write("module load lammps \n")
        data_file.write("srun -n {} -c {} --cpu-bind=cores lmp_cori".format(thread_num, thread_per_cpu) +
                        " -in miniRun.lmp -log logFiles/mylog_$SLURM_JOB_ID.lammps \n")


# Create the moltemplate file
def get_moltemplate_system_file(file_name,
                                density_g_cm3,
                                box_size_A,
                                molecule_file,
                                molecule_name,
                                molar_mass, ):
    # Get the number of molecules to create
    (molecule_number_3D,
     molecule_number_1D,
     box_size_new_A,
     spacing) = util.get_molecule_number_1D(density_gcm3=density_g_cm3,
                                            molar_mass_gmol=molar_mass,
                                            box_size_A=box_size_A)

    with open(file_name, 'w') as data_file:
        data_file.write(
            "write_once(\"Data Boundary\") { \n" +
            "{} {} xlo xhi \n".format(0.0, box_size_new_A) +
            "{} {} ylo yhi \n".format(0.0, box_size_new_A) +
            "{} {} zlo zhi \n".format(0.0, box_size_new_A) +
            "} \n" +
            "\n" +
            "write_once(\"In Init\") {\n" +
            "\n" +
            "units           real \n" +
            "boundary p p p\n" +
            "atom_style      full \n" +
            "}\n" +
            "\n" +
            "# import molecule building block file\n" +
            "import \"{}\" \n".format(molecule_file) +
            "\n"
        )
        data_file.write("mol = new {} [{}].move({}, 0.0, 0.0)\n".format(molecule_name, molecule_number_1D, spacing) +
                        "             [{}].move(0.0, {}, 0.0)\n".format(molecule_number_1D, spacing) +
                        "             [{}].move(0.0, 0.0, {})\n".format(molecule_number_1D, spacing)
                        )


# Create the lammps file
def get_lammps_file(fileName,
                    temperature,
                    temperature1,
                    randomSeed,
                    # saveAtomPosition=False,
                    dump_num=100,
                    # restart_num=5000,
                    run_num=10000,
                    run_num1=1000,
                    run_num2=1000,
                    # getRDF=
                    ):
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
        lammpsScript.write("#Calculate the temperature pressure and avoid some strange issue\n")
        lammpsScript.write("fix therm all ave/time 1 10 10 v_P v_T v_rho file ./output/myThermo.txt \n")
        lammpsScript.write("fix fRattleTIP4p tip4p rattle 0.0001 10 100 b 1 a 1 \n")
        lammpsScript.write("#Get rdf\n")
        lammpsScript.write("compute myRDF tip4p rdf 50 1 1 \n")
        lammpsScript.write("fix getRDF all ave/time 5 20 100 c_myRDF[*] file ./output/myRDF.rdf mode vector \n")
        lammpsScript.write("dump 1 tip4p custom {} ./output/atomPos.* id type x y z\n".format(dump_num))
        lammpsScript.write("neigh_modify \n")

        # --------------------------------------------------------
        # Calculate the equilibrium state
        # --------------------------------------------------------
        lammpsScript.write("fix fxnvt all nvt temp {} {} 10.0\n".format(temperature, temperature))
        lammpsScript.write("run  {} \n".format(run_num))
        lammpsScript.write("unfix fxnvt\n")

        # --------------------------------------------------------
        # Calculate the heating process
        # --------------------------------------------------------
        lammpsScript.write("fix fxnvt1 all nvt temp {} {} 10.0\n".format(temperature, temperature1))
        lammpsScript.write("run  {} \n".format(run_num1))
        lammpsScript.write("unfix fxnvt1\n")

        # --------------------------------------------------------
        # Calculate the later process
        # --------------------------------------------------------
        lammpsScript.write("fix fxnvt2 all nvt temp {} {} 10.0\n".format(temperature1, temperature1))
        lammpsScript.write("run  {} \n".format(run_num2))
        lammpsScript.write("unfix fxnvt2\n")


# Create the simulation project
def createNewProj(proj_ID,
                  temperature=293.15,
                  temperature_change=50.0,
                  density_gcm3=1.0,
                  box_size_A=100,
                  random_seed=1000, ):

    workdir = "/mnt/c/Users/haoyuan/Desktop/PosDoc/LammpsProjInitialization/proj{}".format(proj_ID)
    # curdir = os.getcwd()

    # Create new files contains the information of the system
    try:
        shutil.copytree("/mnt/c/Users/haoyuan/Documents/GitHub/LammpsTools/LammpsTools/auxNotebooks",
                        workdir)
    except FileExistsError:
        print("This project exists. Pleasse use an new project ID.")
        return

    os.chdir(workdir)
    # Create the system.lt file in the new project file
    get_moltemplate_system_file(
        file_name="system.lt",
        density_g_cm3=density_gcm3,
        box_size_A=box_size_A,
        molecule_file="/mnt/c/Users/haoyuan/Documents/GitHub/LammpsTools/moleculeZoo/tip4p2005.lt",
        molecule_name="TIP4P2005",
        molar_mass=molar_mass,
    )

    # Convert the moltemplate file into lammps file
    os.system("/home/haoyuan/software/moltemplate/moltemplate/scripts/moltemplate.sh system.lt")
    os.mkdir("logFiles")
    os.mkdir("output")

    # Create the initial lammps simulation script
    get_lammps_file(fileName="miniRun.lmp".format(proj_ID),
                    temperature=temperature,
                    temperature1=temperature + temperature_change,
                    randomSeed=random_seed,
                    dump_num=50,
                    run_num=10000,
                    run_num1=200,
                    run_num2=10000,
                    )

    # Create the sbatch file to submit the job
    get_sbatch_file_cori(file_name="submit.mini".format(proj_ID),
                         calculation_hour=1,
                         account_name='m4070')
