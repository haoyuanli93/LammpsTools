import os
import shutil
import sys


#tempList = [300, 400, 500, 600]
tempList = [282, 296, 298, 308, 328]


for idx in range(len(tempList)):
    temp = tempList[idx]

    # Check if the project already exist:
    try:
        shutil.copytree("./sourceProj/", "./T_{}K/".format(temp))
    except FileExistsError:
        print("This project exists. Pleasse use an new project ID.")

    # Create necessary files for the simulation
    os.chdir("./T_{}K/".format(temp))
    os.system("/home/haoyuan/github/moltemplate/moltemplate/scripts/moltemplate.sh ./system.lt")

    # Modify the temperature info in the miniRun file
    with open("./miniRun.lmp", 'r') as sourceFile:

        # Load the text
        data = sourceFile.read()

        # Change the text
        data = data.replace("velocity all create 300  9837",
                            "velocity all create {}  9837".format(temp))
        data = data.replace("velocity all scale 300",
                            "velocity all scale {}".format(temp))
        data = data.replace("fix fxnpt all npt temp 300.0 300.0 25.0 iso 1.0 1.0 250.0 drag 1.0",
                            "fix fxnpt all npt temp {} {} 25.0 iso 1.0 1.0 250.0 drag 1.0".format(temp, temp))

    with open("./miniRun.lmp", 'w') as sourceFile:

        # Load the text
        sourceFile.write(data)

    # Go to the orignial folder
    os.chdir("..")