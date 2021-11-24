import sys
import os
from typing import Union

from scipy import sparse
import numpy as np
import pandas as pd
from argparse import Namespace

from estimator_utils.SatLasso import SatLasso, SatLassoCV
from estimator_utils.seqparser import seqparser, map_coefs, create_coefs_dataframe

def check_dataframe(df: pd.DataFrame, y_colname: str, sequence_colname: str, id_colname: str, heavy_chain_colname: str, light_chain_colname: str, map_back: bool):
    error = False
    if y_colname not in df.columns:
        error = True
        error_msg = "y_colname {} must be in df. Columns currently include: {}".format(y_colname, df.columns)
    elif sequence_colname not in df.columns:
        error = True
        error_msg = "sequence_colname {} must be in df. Columns currently include: {}".format(sequence_colname, df.columns)
    elif id_colname not in df.columns:
        error = True
        error_msg = "id_colname {} must be in df. Columns currently include: {}".format(id_colname, df.columns)
    elif map_back and (heavy_chain_colname not in df.columns or light_chain_colname not in df.columns):
        error = True
        error_msg = "Both heavy and light chain colnames {}, {} must be in df. Columns currently include: {}".format(heavy_chain_colname, light_chain_colname, df.columns)
    if error:
        raise ValueError(error_msg)
    return
 
# Main function
def fit_estimator(df: pd.DataFrame, y_colname: str, sequence_colname: str, id_colname: str, lambda1: Union[float, list], lambda2: Union[float, list], lambda3: Union[float, list], heavy_chain_colname: str = None, light_chain_colname: str = None, saturation: Union[int, float] = 'max', map_back: bool = False, cv: int = 0) -> dict:
    """
        Runs estimator program for amino acid sequences in given dataframe: Parse amino acid sequence and run SatLasso for variable selection.
        Arguments:
         - df: Dataframe containing amino acid sequences and associated metadata
         - y_colname: Name of column for y values (e.g. IC50 values)
         - sequence_colname: Name of column for AA sequences
         - id_colname: Name of column for identifying name of each AA sequence
         - lambda1: Lambda 1 value in SatLasso objective; or if using CV, start value for lambda 1
         - lambda2: Lambda 2 value in SatLasso objective; or if using CV, start value for lambda 2
         - lambda3: Lambda 3 value in SatLasso objective; or if using CV, start value for lambda 3
         - heavy_chain_colname: Name of column for heavy chain AA sequence (only used in map_back = True)
         - light_chain_colname: Name of column for light chain AA sequence (only used in map_back = True)
         - saturation: Saturation value to use for SatLasso(CV): can be float or {"max", "mode"}
         - map_back: Boolean variable to determine whether to map coefficients back to individual amino acid sequences in training data
         - cv: Cross-validation value: use int > 0 for SatLassoCV with specified number of folds; otherwise SatLasso (no CV) used
        Returns:
         - return_dict: dictionary with values:
            - Satlasso: SatLasso estimator fitted
            - Coefficients: dataframe of coefficients with associated AA position
            - If map_back is True, Mapped Coefficients: dataframe of coefficients with associated AA position mapped back to each antibody in the training data
    """
    
    ## Check dataframe for format
    check_dataframe(df, y_colname, sequence_colname, id_colname, heavy_chain_colname, light_chain_colname, map_back)
    
    ## Convert AA sequences to binary encoded matrix, save as scipy sparse matrix
    sparse_matrix = seqparser(df, sequence_colname)
    scp_sparse_matrix = sparse.csr_matrix(sparse_matrix)
    
    ## Retrieve y values for training regression
    y = df[y_colname].values.astype(float)
    
    ## Run satlasso fitting algorithm with / without cross-validation
    if not cv:
        satlasso = SatLasso(lambda_1 = lambda1, lambda_2 = lambda2, lambda_3 = lambda3, saturation = saturation, normalize = (transform == 'norm'))
            
    else:
        assert isinstance(lambda1, list) and isinstance(lambda2, list) and isinstance(lambda3, list), "The provided lambdas for cross-validation must be lists."
        satlasso = SatLassoCV(lambda_1s = lambda1, lambda_2s = lambda2, lambda_3s = lambda3, saturation = saturation, cv = cv)
    
    satlasso.fit(sparse_matrix, y)
    
    ## Get coefficients from SatLasso object
    coefficients = satlasso.coef_
    coefficients = create_coefs_dataframe(coefficients)
    
    return_dict = {"SatLasso": satlasso, "Coefficients": coefficients}
    
    if map_back:
        mapped_coefficients = map_coefs(df, df_coefs, heavy_chain_colname, light_chain_colname, id_colname, sequence_colname)
        return_dict["Mapped Coefficients"] = mapped_coefficients
    
    return return_dict
