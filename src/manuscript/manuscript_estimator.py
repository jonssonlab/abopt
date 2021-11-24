from pipeline import estimator


import numpy as np
import pandas as pd

## For Nussenzweig data
## Read in the data
datapath = "../../data/estimator/NeutSeqData_C002-215_cleaned_aligned.csv",
df = pd.read_csv(datapath, sep=",", header=0)

## Transform y values by natural log
df["sars_cov_2_ic50_igml"] = np.log(df["sars_cov_2_ic50_igml"])

## Fit estimator
output_dict = estimator.fit_estimator(df = df, y_colname = "sars_cov_2_ic50_igml", sequence_colname = "sequence", id_colname = "antibody_id",
                                      lambda1 = 5.,lambda2 = 10., lambda3 = 6.25, heavy_chain_colname = "heavy_chain_aligned", light_chain_colname = "light_chain_aligned",
                                      saturation = "max", map_back = True, cv = 0)

coefficients = output_dict["Coefficients"]
mapped_coefficients = output_dict["Mapped Coefficients"]

## For Oxford CovAb Database
## Read in the data
datapath = "../data/estimator/NeutSeqData_VH3-53_66_aligned.csv",
df = pd.read_csv(datapath, sep=",", header=0)

## Fit estimator
output_dict = estimator.fit_estimator(df = df, y_colname = "IC50_ngml", sequence_colname = "sequence", id_colname = "antibody_id",
                                      lambda1 = 1., lambda2 = 10., lambda3 = 10., heavy_chain_colname = "heavy_chain_aligned",
                                      light_chain_colname = "light_chain_aligned", saturation = "max", map_back = True, cv = 0)

coefficients = output_dict["Coefficients"]
mapped_coefficients = output_dict["Mapped Coefficients"]
