# abopt

AbOpt is a tool for antibody optimization. This computational method jointly constrains the antibody design space and solves for antibody cocktails to address virus escape. Our contributions are: a new statistical model, the saturated LASSO (satlasso) for feature selection and a new combinatorial optimization algorithm that inputs fitness landscapes and solves for antibody cocktails that can address escape through virus mutations. 

## Disclaimer: this code is a work in progress currently being cleaned up. ##


# NeurIPS LMRL talk 
https://drive.google.com/file/d/1Zm_ei3fueVl2_HlRLixcX6dNxwPcy2FU/view

# Install 

To install necessary packages: 
    >   ` pip install wget `
    >   ` pip install biopandas `
    >   ` pip install scikit-learn`
    >   ` pip install Levenshtein `
    >   ` pip install cvxpy `
    >   ` pip install logomaker `

To re-run fitness landscape calculations for the manuscript you will need to install FoldX. 

http://foldxsuite.crg.eu/


# Manual

## antibody pipeline 


    abopt v 0.1 
   
    distance      Calculate Levenshtein distance between antibody sequences
    estimator     Run estimator on antibody sequences
    map           Map estimator FASTA locations to PDB locations 
    constrain     Constrain estimator features
    scan          Mutational scanning on structure 
    mutate        Mutate a molecular structure 
    repair        Repair a molecular structure
    epitope       Finds molecular structure binding regions
    energy        Run energy minimization functions 
    merge         Merges fitness landscape data  
    cocktail      Solves for antibody combinations to target virus escape mutants

