import os

global num_engines
global working_dir
global permanent_dir
global num_sampled_random_graphs
global vertex_cut_off, edge_cut_off
global seed


# set_num_engines(n_engines) takes as an argument the amount of cores to be
# used. If -1 is passed as an argument, then all available cores will be used.
def set_num_engines(n_engines):
    global num_engines
    max = int(os.environ.get("SLURM_NTASKS", os.cpu_count()))
    if n_engines >= 1 and n_engines <= max:
        num_engines = n_engines
    else:
        num_engines = max
    print("Using: " + str(num_engines) + " cores")


def set_num_sampled_random_graphs(number_of_sampled_random_graphs):
    global num_sampled_random_graphs
    num_sampled_random_graphs = number_of_sampled_random_graphs


def set_cut_off(min_num_vertices, min_num_edges):
    global vertex_cut_off, edge_cut_off
    vertex_cut_off, edge_cut_off = min_num_vertices, min_num_edges


def set_seed(init_seed):
    global seed
    seed = init_seed


def set_working_dir(working_dir_path):
    global working_dir
    working_dir = working_dir_path

def set_permanent_dir(permanent_dir_path):
    global permanent_dir
    permanent_dir = permanent_dir_path


def get_num_engines():
    global num_engines
    return num_engines


def get_num_sampled_random_graphs():
    global num_sampled_random_graphs
    return num_sampled_random_graphs


def get_vertex_cut_off():
    global vertex_cut_off
    return vertex_cut_off


def get_edge_cut_off():
    global edge_cut_off
    return edge_cut_off


def get_seed():
    global seed
    return seed


def get_working_dir():
    global working_dir
    return working_dir


def get_data_dir():
    global working_dir
    return working_dir + "datasets/"


def get_log_dir():
    global working_dir
    return working_dir + "logs/"

def get_permanent_dir():
    global permanent_dir
    return permanent_dir