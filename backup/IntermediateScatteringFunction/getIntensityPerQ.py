import numpy as np
import time

import sys
sys.path.append("/cds/home/h/haoyuan/Documents/my_repos/SpeckleContrastEstimation/")
from ContrastEstimation import ScatteringInfoMD


###############################################################
#   Load a single data set to get simulation info
###############################################################
tic = time.time()
atomNum, spaceLimit, atomType, atomPosition  = ScatteringInfoMD.load_atom_info("./output/atomPos.20000")
toc = time.time()
print("It takes {:.2f} seconds to load the atom position".format(toc - tic))


# Sort according to the atom type
(atomTypeArray,
 atomTypeStartIdx,
 atomTypeCount,
 atomType,
 atomPosition) = ScatteringInfoMD.categorize_atoms(atom_types=atomType,
                                                   position_holder=atomPosition)


###########################################################################
#    Get wave-vectors to calculate the intermediate scattering function
###########################################################################
qArray = np.linspace(start=0.1, stop=1, num=20)
qRange = [-0.02, 0.02]

qVectorArray = []
qVectorNum = [0, ]

for qIdx in range(qArray.shape[0]):
    qVectorArray.append(ScatteringInfoMD.get_q_vector_list_in_range(box_size_xyz_A=spaceLimit[:, 1] - spaceLimit[:, 0],
                                                                    q_low_A=qArray[qIdx] + qRange[0],
                                                                    q_high_A=qArray[qIdx] + qRange[1], )
                        )
    qVectorNum.append(qVectorArray[-1].shape[0])

qVectorArray = np.ascontiguousarray(np.vstack(qVectorArray))


####################################################################
#   Get the intensity for each Q in each simulation
###################################################################
# Loop through all the atom configurations
for x in range(601):
    tic = time.time()

    # Load the atom configuration
    (atomNum,
     spaceLimit,
     atomType,
     atomPosition) = ScatteringInfoMD.load_atom_info("./output/atomPos.{}".format(x * 100 + 20000))

    # Sort according to the atom type
    (atomTypeArray,
     atomTypeStartIdx,
     atomTypeCount,
     atomType,
     atomPosition) = ScatteringInfoMD.categorize_atoms(atom_types=atomType,
                                                       position_holder=atomPosition)

    # Get the scattering intensity for the pre-selected q vectors
    intensity = ScatteringInfoMD.get_MD_formfactor_at_Q_list_parallel_at_Q(q_list_A=qVectorArray,
                                                                           atom_position_array=atomPosition,
                                                                           atom_type_array=atomType,
                                                                           atom_type_name_list=['O', 'H'])

    np.save("./output/intensity.{}.npy".format(x * 100 + 20000), intensity)

    toc = time.time()
    print("It takes {:.2f} seconds to process 1 atom configuration".format(toc - tic))