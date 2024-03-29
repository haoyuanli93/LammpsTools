import numpy as np


def get_thermo_info(log_file_name):
    """
    Extract the thermodynamics information
    from the log file

    :param log_file_name:
    :return:
    """

    with open(log_file_name, 'r') as log_file:
        # Load lines
        lines = log_file.readlines()

        # IntermediateScatteringFunction line numbers
        line_nums = len(lines)

        # IntermediateScatteringFunction a holder for the info to save
        thermo_info = []

        # Whether to save the thermo info the to list
        flag = False

        for line_idx in range(line_nums):

            # IntermediateScatteringFunction a line
            line = lines[line_idx]
            words = line.split()

            # Check if this is an empty line
            if not words:
                continue

            # Check if the start word is step
            if words[0] == 'Step':
                thermo_info_type = [str(word) for word in words]
                flag = True
                continue

            if words[0] == 'Loop':
                flag = False
                continue

            if flag:
                thermo_info.append([float(word) for word in words])

    # Convert the list to numpy array
    thermo_info = np.array(thermo_info)

    # Convert the numpy array to dictionary
    thermo_info_dict = {}

    # Add entry to the dictionary
    entry_num = len(thermo_info_type)
    for entry_idx in range(entry_num):
        thermo_info_dict.update({thermo_info_type[entry_idx]: np.copy(thermo_info[:, entry_idx])})

    return thermo_info_dict


def get_thermo_info_log_list(log_file_name_list):
    """
    When we restart a simulation several times,
    we can save the thermo info to the same dictionary
    with the common entry.

    :param log_file_name_list:
    :return:
    """
    thermo_info_dict_list = []
    entry_lists = []
    log_num = len(log_file_name_list)

    for log_idx in range(log_num):
        thermo_info_dict_list.append(get_thermo_info(log_file_name=log_file_name_list[log_idx]))
        entry_lists.append(list(thermo_info_dict_list[-1].keys()))

    # Total thermo info
    thermo_info_dict_tot = {}

    entry_list = set(entry_lists[0])
    entry_list = entry_list.intersection(*entry_lists)

    entry_list = list(entry_list)

    for entry_idx in range(len(entry_list)):
        entry_name = entry_list[entry_idx]
        content = np.concatenate([thermo_info_dict_list[idx][entry_name] for idx in range(log_num)])
        thermo_info_dict_tot.update({entry_name: content})

    return thermo_info_dict_tot


def get_rdf_info(rdf_file):
    """

    :param rdf_file:
    :return:
    """
    with open(rdf_file, 'r') as rdf_file:

        # Create a holder to load the file
        rdf_holder = []
        step_holder = []

        # Skip the header
        for x in range(3):
            _ = rdf_file.readline()

        # Load a new line
        line = rdf_file.readline()
        while line:

            # Parse the new line
            words = line.split()

            # If there are only two words, then this is a description line
            if len(words) == 2:

                # Save the step number
                step_holder.append(int(words[0]))

                # Create a new entry in the rdf holder
                # Because a new description line
                # means the following data are from a new MD snapshot
                rdf_holder.append([])

            # otherwise, this is a data line
            else:
                rdf_holder[-1].append([float(words[y]) for y in range(1, 4)])

            # Load a new line
            line = rdf_file.readline()

    return np.array(step_holder, dtype=np.int64), np.array(rdf_holder, dtype=np.float64)
