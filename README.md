# The Myth of the Robust-Yet-Fragile Nature of Scale-Free Networks: An Empirical Analysis

**Authors**: Rouzbeh Hasheminezhad, August Bøgh Rønberg, Ulrik Brandes

## Setup (currently only available for GNU/Linux|MacOS)
Confirm that a [LaTeX ](https://www.latex-project.org/get/) distribution is
installed. It will be used in generating the figures of the paper. 

Clone this GitHub repository. If `conda` is not already installed, download and
install [Miniconda](https://docs.conda.io/en/latest/miniconda.html#).

The following command creates a `conda` environment that includes required
dependencies.

```
conda env create -f environment.yml
```

Activate the new `WAW` environment in `conda` before executing the following
steps in order. 

```
conda activate WAW
```

### Collecting, formatting, and preprocessing networks
The following script, collects networks from various online sources and formats
them in a `datasets` directory.  

The script uses all available CPU cores. To specify the number of cores, replace
-1 with the desired `number_of_cores`. 

```
python collect.py --cores -1
```

### Robustness analysis
After running the following script, the `datasets` directory is updated to
include the robustness scores for each network. 

The script uses all available CPU cores. To specify the number of cores, replace
-1 with the desired `number_of_cores`. 


The code runs in parallel on all cores, to specify the number of cores change
`-1` to the desired `number_of_cores`.

```
python analysis.py --cores -1
```
This script is resource-intensive for a personal computer.  To ease replication,
we provide all robustness scores [**here**](https://polybox.ethz.ch/index.php/s/qymJQoRMYMYPAvN).

### Visualizations
The following creates the directory `figures/` and generates the paper's figures
there. 
```
python figures.py
```

### Scale-freeness analysis
We do not provide the code to run the scale-freeness analysis here as our code
is a combination of licensed code from A. Broido et al. and Voitalov et al. all of which
we modified to fit our use case. Their original code can however, be found at:
[**Broido et al.**](https://github.com/adbroido/SFAnalysis) and 
[**Voitalov et al.**](https://github.com/ivanvoitalov/tail-estimation).


## Citation
If you use this script as a part of your research, we would be grateful if you
cite this code repository and/or the original paper.

```
@inproceedings{hasheminezhad2023myth,
  title={The Myth of the Robust-Yet-Fragile Nature of Scale-Free Networks: An Empirical Analysis},
  author={Hasheminezhad, Rouzbeh and R{\o}nberg, August B{\o}gh and Brandes, Ulrik},
  booktitle={Algorithms and Models for the Web Graph: 18th International Workshop, WAW 2023, Toronto, ON, Canada, May 23--26, 2023, Proceedings},
  pages={99--111},
  year={2023},
  organization={Springer}
}

```
## Contact
In case you have questions, please contact [Rouzbeh
Hasheminezhad](mailto:shashemi@ethz.ch) or [August Bøgh
Rønberg](mailto:ronberga@ethz.ch).