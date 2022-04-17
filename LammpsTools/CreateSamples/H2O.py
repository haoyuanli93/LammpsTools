import numpy as np
from scipy.spatial.transform import Rotation

N_A = 6.02214076e23  # avogadro constant

h_mass = 1.00784  # atomic mass for Hydrogen
o_mass = 15.9994  # atomic mass for oxygen

h_charge = 0.5564  # hydrogen charge
o_charge = -1.1128  # oxygen charge

h2o_mass = 2 * h_mass + o_mass  # molecule mass of H2O

hoh_angle = np.deg2rad(104.52)  # The hoh angle of water
hoh_angle_eps = np.deg2rad(1e-5)  # uncertainty of the bond angle

ho_bond_length = 0.9572  # The H-O bond length in A

dist_min = 2.7  # minimal distance between two oxygen atoms


def get_oxygen_positions(box_size_A, molecule_num, random_number_seed):
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
    grid_coordinate = np.zeros((axis_part_num, axis_part_num, axis_part_num, 3), dtype=np.float64)
    grid_coordinate[:, :, :, 0] = np.arange(axis_part_num)[:, np.newaxis, np.newaxis]
    grid_coordinate[:, :, :, 1] = np.arange(axis_part_num)[np.newaxis, :, np.newaxis]
    grid_coordinate[:, :, :, 2] = np.arange(axis_part_num)[np.newaxis, np.newaxis, :]

    # Convert the 3D coordinate to 1D to randomly choose from it
    grid_coordinate = np.reshape(a=grid_coordinate, newshape=(axis_part_num ** 3, 3))

    # Shuffle the array and choose from it
    np.random.shuffle(grid_coordinate)

    # Choose the first several samples as the initial position of the molecules
    grid_coordinate = grid_coordinate[:molecule_num, :]

    # Convert the grid coordinate to the molecule positions in A
    grid_coordinate *= (box_size_A / float(axis_part_num))

    # Move the center to 0
    # grid_coordinate -= (box_size_A / 2.)

    # Purturb the water molecules
    max_move = (box_size_A / float(axis_part_num) - dist_min)
    # grid_coordinate += (np.random.rand(molecule_num, 3) - 0.5) * max_move
    grid_coordinate += np.random.rand(molecule_num, 3) * max_move

    return grid_coordinate


def get_hygen_positions_random_orientation(oxygen_positions, random_number_seed):
    """

    :param oxygen_positions:
    :param random_number_seed:
    :return:
    """
    # Get molecule numbers
    molecule_num = oxygen_positions.shape[0]

    # Create holders for the Hydrogen
    h_holder = np.zeros((molecule_num, 2, 3), dtype=np.float64)

    # Create the vector for the first hydrogen
    h_holder[:, 0, 0] = 1.

    # slightly change the bond angle
    np.random.seed(random_number_seed)
    angles = hoh_angle + (np.random.rand(molecule_num) - 0.5) * hoh_angle_eps
    h_holder[:, 1, 0] = np.cos(angles)
    h_holder[:, 1, 1] = np.sin(angles)

    # Add the bond length
    h_holder *= ho_bond_length

    ##############################################################################
    #    Rotate the Hydrogen bonds
    ##############################################################################
    # Split the array a little bit if the molecule number is very large
    batch_num = molecule_num // 1000
    for batch_idx in range(molecule_num // 1000):
        rotation_matrix = Rotation.random(num=1000).as_matrix()

        batch_holder = np.copy(h_holder[batch_idx * 1000: (batch_idx + 1) * 1000])
        # Rotate the vectors for the first Hydrogen
        h_holder[batch_idx * 1000: (batch_idx + 1) * 1000, 0, :] = rotation_matrix[:, :, 0]

        # Rotate the vector for the second Hydrogen
        h_holder[batch_idx * 1000: (batch_idx + 1) * 1000, 1, :] = (np.multiply(rotation_matrix[:, :, 0],
                                                                                batch_holder[:, 1, 0][:, np.newaxis]) +
                                                                    np.multiply(rotation_matrix[:, :, 1],
                                                                                batch_holder[:, 1, 1][:, np.newaxis])
                                                                    )

    # Process the last batch
    num = molecule_num - 1000 * batch_num
    rotation_matrix = Rotation.random(num=num).as_matrix()

    batch_holder = np.copy(h_holder[batch_num * 1000: molecule_num])
    # Rotate the vectors for the first Hydrogen
    h_holder[batch_num * 1000: molecule_num, 0, :] = rotation_matrix[:, :, 0]

    # Rotate the vector for the second Hydrogen
    h_holder[batch_num * 1000: molecule_num, 1, :] = (np.multiply(rotation_matrix[:, :, 0],
                                                                  batch_holder[:, 1, 0][:, np.newaxis]) +
                                                      np.multiply(rotation_matrix[:, :, 1],
                                                                  batch_holder[:, 1, 1][:, np.newaxis])
                                                      )

    ##############################################################################
    #     Attach the Hydrogen bond to the oxygen
    ##############################################################################
    return h_holder + oxygen_positions[:, np.newaxis, :]


def save_water_molecule_data(oxygen_positions, hydrogen_positions, box_size_A, file_name):
    """
    Save the oxygen and hydrogen position to the file

    :param oxygen_positions:
    :param hydrogen_positions:
    :param box_size_A
    :param file_name:
    :return:
    """

    # Get the atom number
    molecule_num = oxygen_positions.shape[0]

    with open(file_name, 'w') as data_file:
        # Comment line
        data_file.write("LAMMPS Atom File \n")
        data_file.write('\n')

        # head
        # Specify statistics of the atoms
        data_file.write("{} atoms\n".format(molecule_num * 3))
        data_file.write("{} bonds\n".format(molecule_num * 2))
        data_file.write("{} angles\n".format(molecule_num))
        data_file.write("{} dihedrals\n".format(0))
        data_file.write("{} impropers\n".format(0))

        data_file.write("\n")
        data_file.write("\n")

        data_file.write("{} atom types\n".format(2))
        data_file.write("{} bond types\n".format(1))
        data_file.write("{} angle types\n".format(1))

        data_file.write("\n")
        data_file.write("\n")

        # Specify the box info
        data_file.write("{} {} xlo xhi\n".format(0, box_size_A))
        data_file.write("{} {} ylo yhi\n".format(0, box_size_A))
        data_file.write("{} {} zlo zhi\n".format(0, box_size_A))

        data_file.write("\n")
        data_file.write("\n")

        # Specify the masses

        data_file.write("Masses\n")
        data_file.write("\n")

        data_file.write("1 {}\n".format(o_mass))  # type 1 is oxygen
        data_file.write("2 {}\n".format(h_mass))  # type 2 is hydrogen
        data_file.write("\n")
        data_file.write("\n")

        # Specify the atom positions
        data_file.write("Atoms\n")
        data_file.write("\n")
        for molecule_idx in range(molecule_num):
            # Write the first oxygen
            data_file.write("{}   {}   {}   {}   {}   {}   {}\n".format(
                molecule_idx * 3 + 1,  # atom idx
                molecule_idx + 1,  # molecule idx
                1,
                o_charge,  # Charge
                oxygen_positions[molecule_idx, 0],  # x
                oxygen_positions[molecule_idx, 1],  # x
                oxygen_positions[molecule_idx, 2],  # x
            ))

            # Write the first hydrogen
            data_file.write("{}   {}   {}   {}   {}   {}   {}\n".format(
                molecule_idx * 3 + 2,  # atom idx
                molecule_idx + 1,  # molecule idx
                2,
                h_charge,  # Charge
                hydrogen_positions[molecule_idx, 0, 0],  # x
                hydrogen_positions[molecule_idx, 0, 1],  # x
                hydrogen_positions[molecule_idx, 0, 2],  # x
            ))

            # Write the second hydrogen
            data_file.write("{}   {}   {}   {}   {}   {}   {}\n".format(
                molecule_idx * 3 + 3,  # atom idx
                molecule_idx + 1,  # molecule idx
                2,
                h_charge,  # Charge
                hydrogen_positions[molecule_idx, 1, 0],  # x
                hydrogen_positions[molecule_idx, 1, 1],  # x
                hydrogen_positions[molecule_idx, 1, 2],  # x
            ))

        data_file.write("\n")
        data_file.write("\n")

        # Specify the bond positions
        data_file.write("Bonds\n")
        data_file.write("\n")

        for molecule_idx in range(molecule_num):
            # Add the first bond
            data_file.write("{}   {}   {}   {}\n".format(
                molecule_idx * 2 + 1,  # bond idx
                1,  # bond type
                molecule_idx * 3 + 1,
                molecule_idx * 3 + 2,
            ))

            # Add the second bond
            data_file.write("{}   {}   {}   {}\n".format(
                molecule_idx * 2 + 1,  # bond idx
                1,  # bond type
                molecule_idx * 3 + 1,
                molecule_idx * 3 + 3,
            ))

        data_file.write("\n")
        data_file.write("\n")

        # Specify the angles
        data_file.write("Angles\n")
        data_file.write("\n")

        for molecule_idx in range(molecule_num):
            # Add the first bond
            data_file.write("{}   {}   {}   {}   {}\n".format(
                molecule_idx,  # angle idx
                1,  # angle type
                molecule_idx * 3 + 2,
                molecule_idx * 3 + 1,
                molecule_idx * 3 + 3,
            ))


def get_data_file(box_size_A, density_g_cm3, file_name, random_number_seed):
    """

    :param box_size_A:
    :param density_g_cm3:
    :param file_name:
    :param random_number_seed:
    :return:
    """
    # Get the corresponding particle number
    volume = (box_size_A * 1e-8) ** 3  # volume in cm^3
    mass = volume * density_g_cm3  # mass in g
    particle_num = int(N_A * (mass / h2o_mass))

    # Get the oxygen and hydrogen positions
    oxygen_positions = get_oxygen_positions(box_size_A=box_size_A,
                                            molecule_num=particle_num,
                                            random_number_seed=random_number_seed)
    hydrogen_positions = get_hygen_positions_random_orientation(oxygen_positions=oxygen_positions,
                                                                random_number_seed=random_number_seed)

    # write to the data file
    save_water_molecule_data(oxygen_positions=oxygen_positions,
                             hydrogen_positions=hydrogen_positions,
                             box_size_A=box_size_A,
                             file_name=file_name)

    # Show some basic information
    print("Create an initialization file for {:.2e} molecules".format(particle_num))
