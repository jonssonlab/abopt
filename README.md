# abopt

Abopt is a toolkit for antibody optimization and antibody cocktail selection to target viral escape. This tool is built with anndata and has a Python based API; it includes antibody/virus fitness landscape generation and visualization, feature selection for antibody optimization, and an optmization algorithm that solves for antibody cocktails to address viral escape. This tool works with antibody sequence data and antibody and virus mutational scanning data/fitness lanscapes that can be generated experimentally or approximated using energy minimization tools like FoldX.  


NeurIPS 2020 LMRL workshop poster presentation is <a href="https://drive.google.com/file/d/1Zm_ei3fueVl2_HlRLixcX6dNxwPcy2FU/view">here.</a>

# Install
## Requirements:
- Python 3.6+

All code was tested on MacOSX computing environment with software FoldX 5.0 (download <a href="http://foldxsuite.crg.eu/">here<a>). There may exist some incompatibilities with FoldX configuration on other operating systems. 

## Dependencies:
- anndata
- biopandas
- colorcet
- cvxpy
- Levenshtein
- logomaker
- matplotlib
- numpy
- pandas
- PIL
- scipy
- seaborn
- sklearn
- statistics
- umap-learn
- wget

To install necessary packages, either install above packages directly via pip:
``` 
pip install scipy numpy
pip install pandas biopandas
pip install logomaker matplotlib PIL seaborn
pip install Levenshtein 
pip install cvxpy scikit-learn statistics umap-learn wget 
```

or alternatively, install via requirements.txt (package versions included):
` pip install -f requirements.txt `

# How to use this repo
## Source Modules
In order to appropriately use the source modules in **abopt**, clone the repository locally to the directory in which the code referencing **abopt** will run:
```
cd /path/to/dir
git clone git@github.com:jonssonlab/abopt.git
```

### Estimator
Classes and functions in the **estimator** module can be accessed by:
```
from abopt.src.estimator import *
```
The primary function in the **estimator** module is `fit_estimator`. `fit_estimator` creates a one-hot encoding on given amino acid sequences, fits a `SatLasso` object with the one-hot encoding and target values, and returns a dictionary containing: the fitted `SatLasso` object; the parameters (or *coefficients* ) of the `SatLasso` object; and, if specified, the parameters mapped back onto each of the amino acid sequences contained in the provided training data. Access the `fit_estimator` function via:
```
from abopt.src.estimator.estimator import fit_estimator
```

### Fitness
Classes and functions in the **fitness** module can be accessed by:

```
from abopt.src.fitness import *
```

The primary functions in the **fitness** module are: 

`repair` : repairs protein structure given a list of pdb names and saves the repaired structure in output/repair/ directory; current implementation uses FoldX for structure repair.  


`remove` : removes chains from a list of protein structures as pdb names and saves the resulting structure in output/remove/ directory.  


`construct_position_scan_string` : constructs a comma delimited string of mutations for a given a protein structure, and user defined sequence locations, individual mutations or protein chain; for example for sequence locations set filter_type=='location', and filter=\[501\],for mutations (filter_type=='mutation' and filter=\['501Y'\].  

`scan` : performs mutational scanning on a given protein structure given a constructed mutation list using construct_position_scan_string function; saves the results in output/scan/ directory.
    
`energy` : calculates difference in binding energies between two different protein structure mutation scans; saves the resutls in output/energy/ directory.
    

### Cocktail
Classes and functions in the **cocktail** module can be accessed by:
```
from abopt.src.cocktail import *
```

The primary functions in the **cocktail** module are: 

`compute_cocktail` : computes the optimal antibody cocktail on antibody and virus fitness landscapes that satisfies the provided minimum virus mutant coverage and returns a tuple including a multi-hot encoding of the optimal antibodies (ordered as given by the antibody fitness array) and the minimal found value; or if the problem does not contain a solution, throws an internal error from **cvxpy**
```
from abopt.src.cocktail.cocktail import compute_cocktail
```

`run_simulations`: computes the optimal antibody cocktail as in `compute_cocktail` over a range of varying hyperparameter values ($\lambda$_1, $\lambda$_2) and provided minimum virus mutant coverages
```
from abopt.src.cocktail.cocktail import run_simulations
```

## Generate paper results
In order to generate the results from the <a href="https://www.biorxiv.org/">paper<\a>, please ensure that the correctly versioned packages are installed as stated in the installation section above. To produce the output and associated plots for the SatLasso regressors and coefficients, run the following bash command in the directory containing **abopt**:
```
python abopt/src/estimator/manuscript.py
```
Note: this program may take several minutes to terminate.

For generation and exploration of antibody and virus fitness landscapes, and specific analysis of designed antibody C105<sup>TH28I-YH58F<\sup>, execute:
```
python abopt/src/fitness/manuscript.py
python abopt/src/fitness/manuscript_C105.py
```

To compute the antibody cocktail optimization algorithm on the generated fitness landscapes and run algorithm simulations, execute:
```
python abopt/src/cocktail/manuscript.py
```
Note: this program may take several hours to terminate.
