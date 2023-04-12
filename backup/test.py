import sys
import time

import numpy as np

sys.path.append("/global/cfs/cdirs/lcls/xpp/software/SpeckleContrastEstimation")
from ContrastEstimation import ScatteringInfoMD

# Define the q bines to calculate the scattering
qLim = [1.3, 2.5]

# ########################################3
#  Because this simulation is using NPT ensemble
#  the size of the volume changes from time to time
#  therefore, here, I need to calcualte the Q vector for each cases
# ########################################
for sIdx in range(200):  # There are totally 200 snap shots to calculate

    # Load the data to know the size of the volume
    tic = time.time()
    atomNum, spaceLimit, atomType, atomPosition = ScatteringInfoMD.load_atom_info("./outputAtom/atomPos.20000")
    toc = time.time()
    print("It takes {:.2f} seconds to load the atom position".format(toc - tic))

    # Sort the atoms
    (atomTypeArray,
     atomTypeStartIdx,
     atomTypeCount,
     atomType,
     atomPosition) = ScatteringInfoMD.categorize_atoms(atom_types=atomType,
                                                       position_holder=atomPosition)
    # Get the wave-vector mesh
    deltaKx = np.pi * 2 / (spaceLimit[0, 1] - spaceLimit[0, 0])
    deltaKy = np.pi * 2 / (spaceLimit[1, 1] - spaceLimit[1, 0])
    deltaKz = np.pi * 2 / (spaceLimit[2, 1] - spaceLimit[2, 0])

    # Get the k Grid
    nKx = qLim[1] / deltaKx
    nKy = qLim[1] / deltaKy
    nKz = qLim[1] / deltaKz

    kGrid = np.zeros((2 * nKx + 1, 2 * nKy + 1, 2 * nKz + 1, 3))
    kGrid[:, :, :, 0] = (np.arange(-nKx, nKx + 1, dtype=np.float64) * deltaKx)[:, np.newaxis, np.newaxis]
    kGrid[:, :, :, 1] = (np.arange(-nKy, nKy + 1, dtype=np.float64) * deltaKy)[np.newaxis, :, np.newaxis]
    kGrid[:, :, :, 2] = (np.arange(-nKz, nKz + 1, dtype=np.float64) * deltaKz)[np.newaxis, np.newaxis, :]

    kLenGrid = np.linalg.norm(kGrid, axis=-1)

    # Get the q vectors within the range of simulation
    qVector = kGrid[np.where((kLenGrid >= qLim[0]) & (kLenGrid < qLim[1]))]

    # Get the scattering intensity for the pre-selected q vectors
    intensity = ScatteringInfoMD.get_MD_formfactor_at_Q_list_parallel_at_Q(q_list_A=qVector,
                                                                           atom_position_array=atomPosition,
                                                                           atom_type_array=atomType,
                                                                           atom_type_name_list=['O', 'H'])

    np.save("./outputRDF/intensity.{}.npy".format(sIdx * 100 + 20000), intensity)
