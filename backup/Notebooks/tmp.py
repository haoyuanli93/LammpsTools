import numpy as np

dist_min = 2.
box_size = 100.
molecule_number = 10320


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
    grid_coordinate -= (box_size_A / 2.)

    # Purturb the water molecules
    max_move = (box_size_A / float(axis_part_num) - dist_min)
    # grid_coordinate += (np.random.rand(molecule_num, 3) - 0.5) * max_move
    grid_coordinate += np.random.rand(molecule_num, 3) * max_move

    return grid_coordinate


# Get the coordinate of the molecules
mol_coordinate = get_molecule_positions(box_size_A=box_size, molecule_num=molecule_number, random_number_seed=1934)

with open("./system.lt", 'w') as data_file:
    data_file.write("write_once(\"Data Boundary\") { \n")
    data_file.write("{} {} xlo xhi \n".format(-box_size / 2., box_size / 2.))
    data_file.write("{} {} ylo yhi \n".format(-box_size / 2., box_size / 2.))
    data_file.write("{} {} zlo zhi \n".format(-box_size / 2., box_size / 2.))
    data_file.write("} \n")
    data_file.write("\n")

    data_file.write("write_once(\"In Init\") {\n")
    data_file.write("# a variable named `cutoff` is required by GROMOS_54A7_ATB.lt\n")
    data_file.write("variable cutoff equal 14.0 # Angstroms\n")
    data_file.write("boundary p p p\n")
    data_file.write("}\n")
    data_file.write("\n")

    data_file.write("# import the forcefield file\n")
    data_file.write("import \"GROMOS_54A7_ATB.lt\" \n")
    data_file.write("# import molecule building block file\n")
    data_file.write("import \"_U1K_allatom_optimized_geometry.lt\" \n")
    data_file.write("\n")
    data_file.write("# create a single copy of this molecule at position 0,0,0\n")

    for mol_idx in range(molecule_number):
        data_file.write("mol{} = new _U1K.move({}, {}, {})\n".format(mol_idx,
                                                                     mol_coordinate[mol_idx, 0],
                                                                     mol_coordinate[mol_idx, 1],
                                                                     mol_coordinate[mol_idx, 2],
                                                                     ))
