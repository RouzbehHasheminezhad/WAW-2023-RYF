from engine.utils.visualization import *
from engine.utils.io import *
from engine.config.config import *
from engine.utils.io import compute_z_score

if __name__ == '__main__':
    # The following sets the working directory where the analysis is performed,
    # and the corresponding results are saved.
    set_working_dir(os.getcwd() + "/")
    # If the directory to save the figures does not exist, create it.
    if not os.path.exists(get_working_dir() + "figures/"):
        os.mkdir(get_working_dir() + "/figures/")
    # Create the bar plot (Figure 1.) of the paper.
    bar_plot()
    # The following creates Figure 2. and Figure 3. of the paper where the
    # targeted attack considered is respectively static and dynamic.
    for adaptive in [True, False]:
        scatter_plot(scores=compute_z_score(1.0),
                     categories=["Technological", "Biological", "Auxiliary", "Transportation", "Social"],
                     is_adaptive=adaptive)
    # The following creates a visualization of the Collins yeast interactome
    # network as displayed in Figure 4. of the paper.
    draw_collins_yeast()
