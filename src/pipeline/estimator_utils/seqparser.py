import numpy as np
import pandas as pd

aalist = ['A', 'R', 'N', 'D','C','Q','E','G','H', 'I','L','K','M', 'F','P','S', 'T', 'W', 'Y' ,'V']

def one_hot_encode(aa):
    """
    One-hot encode an amino acid according to amino acid list (aalist) specified above.
    
    Parameters
    ----------
    aa : str
        One-character representation of an amino acid
    
    Returns
    ----------
    encoding : ndarray of shape (len_aalist,)
        One-hot encoding of amino acid
    """
    
    if aa not in aalist:
        return [0]*len(aalist)
    else:
        encoding = [0]*len(aalist)
        encoding[aalist.index(aa)] = 1
        return encoding

def seqparser(df, seq_col):
    """
    Parse amino acid sequences in dataframe; create a one-hot encoded matrix of sequences.
    
    Note: amino acid sequences must have the same length in order to be parsed.
    
    Parameters
    ----------
    df : pandas DataFrame
        Dataframe containing amino acid sequences to parse
    seq_col : str, default = 'sequence'
        Column in dataframe with amino acid sequences
    
    Returns
    ----------
    aamatrix : numpy array of shape (n_sequences, (len_sequence * len_aalist))
        One-hot encoded matrix of amino acid sequences in dataframe
    """
    
    aamatrix = np.empty((0, len(df[seq_col][0])*len(aalist)), int)
    for seq in df[seq_col]:
        row = []
        for aa in seq:
            row = row + one_hot_encode(aa)
        aamatrix = np.vstack((aamatrix,row))
    return aamatrix
    
def create_coefs_dataframe(coefs):
    """
    Create dataframe from coefficients with index : amino acid positions.

    Note: Intended to be used with coefficient output from regression package /
        (ex. SatLasso, SatLassoCV).

    Parameters
    ----------
    coefs : ndarray of shape (n_coefficients,)
        Coefficient values
        
    Returns
    ----------
    df : pandas DataFrame
        Coefficient dataframe
    """
    data = {
        "Coefficient": coefs,
        "Position": list(range(len(coefs)))*len(aalist),
        "AA": aalist*(len(coefs) // len(aalist))
    }
    df = pd.DataFrame.from_dict(data=data)
    return df

def map_coefs(df, coefs, heavy_chain_name, light_chain_name, id_col, seq_col):
    """
    Maps aligned sequences of amino acids back to original non-aligned sequences
    
    Note: Intended to be used with coefficient output from regression package /
        (ex. SatLasso, SatLassoCV)
    
    Parameters
    ----------
    df : pandas DataFrame
        Metadata including amino acid sequences for heavy and light chains, identifying name /
            of each heavy/light chain pair
    coefs : pandas DataFrame
        Coefficients dataframe
    heavy_chain_name : str
        Name of column for heavy chain sequences in df.
    light_chain_name : str
        Name of column for light chain sequences in df.
    id_col : str
        Name of column for identifier for each heavy/light chain sequences /
            (ex. name of antibody)
    
    Returns
    ----------
    map_df : pandas DataFrame
        Coefficients mapped back to amino acid in each heavy/light chain sequence with /
            MultiIndex : identifying_name, location, chain, amino_acid and columns: /
            wild_type and coefficient
    """
    
    map_df = pd.DataFrame(columns = [id_col, "Position", "Chain", "AA", "WT", "Coefficient"])
    map_df.set_index([id_col, "Position", "Chain", "AA"], inplace = True)
    len_heavy_chain
    for antibody in df[id_col]:
        sequence = df.loc[df[id_col] == antibody][heavy_chain_name]
        pos = 0
        for i in range(0, len(sequence)):
            if sequence[i] in aalist:
                wt = [False] * len(aalist)
                wt[aalist.index(sequence[i])] = True
                map_df = map_df.append(pd.DataFrame.from_dict({id_col: [antibody]*len(aalist), "Position": [str(pos)]*len(aalist), "Chain": ['H']*len(aalist), "AA": aalist, "WT": wt, "Coefficient": coefs.iloc[i, :]["Coefficient"]}, orient = "columns").set_index([id_col, "Position", "Chain", "AA"]))
                pos = pos+1

    for antibody in df[id_col]:
        sequence = df.loc[df[id_col] == antibody][light_chain_name]
        pos = 0
        for i in range(0, len(sequence)):
            if sequence[i] in aalist:
                wt = [False] * len(aalist)
                wt[aalist.index(sequence[i])] = True
                map_df = map_df.append(pd.DataFrame.from_dict({id_col: [antibody]*len(aalist), "Position": [str(pos)]*len(aalist), "Chain": ['L']*len(aalist), "AA": aalist, "WT": wt, "Coefficient": coefs.iloc[i+len_heavy_chain, :]["Coefficient"]}, orient = "columns").set_index([id_col, "Position", "Chain", "AA"]))
                pos = pos+1
    
    return map_df
