import estimator
from plot import *
from seqparser import seqparser

from scipy import sparse
import numpy as np
import pandas as pd

plot = False

## For Nussenzweig data
## Read in the data
datapath = "../../data/estimator/NeutSeqData_C002-215_cleaned_aligned.csv"
df = pd.read_csv(datapath, sep=",", header=0)

## Transform y values by natural log
df["sars_cov_2_ic50_ngml"] = np.log(df["sars_cov_2_ic50_ngml"])

## Fit estimator
output_dict = estimator.fit_estimator(df = df, y_colname = "sars_cov_2_ic50_ngml", sequence_colname = "sequence", id_colname = "antibody_id",
                                      lambda1 = 5.,lambda2 = 10., lambda3 = 6.25, heavy_chain_colname = "heavy_chain_aligned", light_chain_colname = "light_chain_aligned",
                                      saturation = "max", map_back = False, cv = 0)

coefficients = output_dict["Coefficients"]
mapped_coefficients = output_dict["Mapped Coefficients"]

coefficient_filename = "../../output/estimator/NeutSeqData_C002-215_cleaned_aligned_coefficients.csv"
coefficients.replace(mapper={"Coefficients": "coefficients"}, axis=1, inplace=True).to_csv(coefficient_filename, sep=",", header=True, index=False)

mapped_coefficient_filename = "../../output/estimator/NeutSeqData_C002-215_cleaned_aligned_mapped_coefficients.csv"
mapped_coefficients.replace(mapper={"Coefficient": "coefficient",
                                   "Position": "location",
                                   "Chain": "chain",
                                   "AA": "aa",
                                   "WT": "wt"},
                          axis=1, inplace=True).to_csv(mapped_coefficient_filename, sep=",", header=True, index=False)

predictions = output_dict["SatLasso"].predict(
seqparser(df, "sequence"))
df["sars_cov_2_ic50_ngml_predicted"] = np.exp(predictions)

df.to_csv("../../output/estimator/NeutSeqData_C002-215_cleaned_aligned_with_predictors.csv", sep=",", header=0)

# PLOTTING

if plot:
    location_filepath = '../../data/plotting/'
    filepath = '../../output/estimator/'
    output_filepath = '../../figs/'
    filename = 'NeutSeqData_C002-215_cleaned_aligned'
    id_col = 'antibody_id'
    heavy_col = 'heavy_chain'
    light_col = 'light_chain'
    y_col = 'sars_cov_2_ic50_ngml'
    patient_col = 'participant_id'
    gene_col = 'heavy_v_gene'

    metadata = pd.read_csv(filepath+filename+'_with_predictors.csv', sep=',', header=0)
    coefficients = pd.read_csv(filepath+filename+'_coefficients.csv', sep=',', header=0, index_col=0)
    mapped_coefficients = pd.read_csv(filepath+filename+'_mapped_coefficients.csv', sep=',', header=0, index_col=[0,1,2,3])

    plot_prediction(output_filepath, filename, metadata, y_col, '$IC_{50}(ng/ml)$', 'Antibody')
    plot_estimator(output_filepath, filename, coefficients, 'Pseudopositions', 'Coefficients', 'coefficients')
    kde_plot(output_filepath, filename, coefficients, 'Coefficients', 'coefficients')

    coef_posmap = create_coef_matrix(coefficients)
    plot_coef_heatmap(output_filepath, filename, coef_posmap)
    plot_mapped_coefficients(filepath, location_filepath, output_filepath, filename, mapped_coefficients, 'C105', id_col)
    specific_coefficients = mapped_coefficients.loc[np.logical_and(mapped_coefficients.index.get_level_values(id_col)=='C105', mapped_coefficients.index.get_level_values('chain') == 'H')]
    wt_seq = ''.join(list(specific_coefficients[specific_coefficients['wild_type']==True].index.get_level_values('aa')))
    specific_coefficients = specific_coefficients[['coefficient']]
    specific_coefficients.index = [aa+str(location) for name,location,chain,aa in specific_coefficients.index]
    specific_coef_map = create_coef_matrix(specific_coefficients)
    binding_sites = pd.read_csv(location_filepath+'C105_contacts.csv', sep=',', header=0).number_antibody.values
    plot_logoplot(output_filepath, filename, specific_coef_map, binding_sites, 'Positions', 'Coefficients', 'C105', 2, wt_seq)

    full_l_distances, l_distances = create_levenshtein_map(metadata, id_col, heavy_col, light_col)
    l_linkage = create_linkage(full_l_distances)
    metadata[gene_col][~np.logical_or(metadata[gene_col].str.contains('HV3-53') , metadata[gene_col].str.contains('HV3-66'))] = 'Other'
    plot_clustermap(output_filepath, filename, l_linkage, full_l_distances, metadata, id_col, y_col, patient_col, gene_col, '$IC_{50}$', 'patient', 'VH gene')
    plot_hierarchical_clust(output_filepath, filename, full_l_distances, l_linkage, 9)

## For Oxford CovAb Database
## Read in the data
datapath = "../../data/estimator/NeutSeqData_VH3-53_66_aligned.csv"
df = pd.read_csv(datapath, sep=",", header=0)

## Fit estimator
output_dict = estimator.fit_estimator(df = df, y_colname = "IC50_ngml", sequence_colname = "sequence", id_colname = "antibody_id",
                                      lambda1 = 1., lambda2 = 10., lambda3 = 10., heavy_chain_colname = "heavy_chain_aligned",
                                      light_chain_colname = "light_chain_aligned", saturation = "max", map_back = True, cv = 0)

coefficients = output_dict["Coefficients"]
mapped_coefficients = output_dict["Mapped Coefficients"]

coefficient_filename = "../../output/estimator/NeutSeqData_VH3-53_66_aligned_coefficients.csv"
coefficients.replace(mapper={"Coefficients": "coefficients"}, axis=1, inplace=True).to_csv(coefficient_filename, sep=",", header=True, index=False)

mapped_coefficient_filename = "../../output/estimator/NeutSeqData_VH3-53_66_aligned_mapped_coefficients.csv"
mapped_coefficients.replace(mapper={"Coefficient": "coefficient",
                                   "Position": "location",
                                   "Chain": "chain",
                                   "AA": "aa",
                                   "WT": "wt"},
                          axis=1, inplace=True).to_csv(mapped_coefficient_filename, sep=",", header=True, index=False)
                          
predictions = output_dict["SatLasso"].predict(
seqparser(df, "sequence"))
df["IC50_ngml_predicted"] = predictions

df.to_csv("../../output/estimator/NeutSeqData_VH3-53_66_aligned_with_predictors.csv", sep=",", header=0)

# Plotting
if plot:
    combined_filename = filename + '_NeutSeqData_VH3-53_66_aligned'

    location_filepath = '../../data/plotting/'
    filepath = '../../output/estimator/'
    output_filepath = '../../figs/'
    filename = 'NeutSeqData_VH3-53_66_aligned'

    vh3_metadata = pd.read_csv(filepath+filename+'_with_predictors.csv', sep=',', header=0)
    df = create_combined_df([metadata, vh3_metadata], [id_col, 'antibody_id'], [heavy_col, 'heavy_chain'], [light_col, 'light_chain'], [y_col, 'IC50_ngml'], [patient_col, None], [gene_col, 'Heavy V Gene'], 'antibody_id', 'heavy_chain', 'light_chain', 'IC50', 'patient', 'VH gene')
    full_l_distances, _ = create_levenshtein_map(df, 'antibody_id', 'heavy_chain', 'light_chain')
    l_linkage = create_linkage(full_l_distances)
    plot_clustermap(output_filepath, combined_filename, l_linkage, full_l_distances, df, 'antibody_id', 'IC50', 'patient', 'VH gene', '$IC_{50}$', 'patient', 'VH gene')

    heavy_col = 'heavy_chain'
    light_col = 'light_chain'
    id_col = 'antibody_id'
    y_col = 'IC50_ngml'
    patient_col = None
    gene_col = 'Heavy V Gene'

    metadata = pd.read_csv(filepath+filename+'_with_predictors.csv', sep=',', header=0)
    coefficients = pd.read_csv(filepath+filename+'_coefficients.csv', sep=',', header=0, index_col=0)
    mapped_coefficients = pd.read_csv(filepath+filename+'_mapped_coefficients.csv', sep=',', header=0, index_col=[0,1,2,3])

    plot_prediction(output_filepath, filename, metadata, y_col, '$IC_{50}(ng/ml)$', 'Antibody')
    plot_estimator(output_filepath, filename, coefficients, 'Pseudopositions', 'Coefficients', 'coefficients')
    kde_plot(output_filepath, filename, coefficients, 'Coefficients', 'coefficients')

    coef_posmap = create_coef_matrix(coefficients)
    plot_coef_heatmap(output_filepath, filename, coef_posmap)
    for antibody in ['C105', 'CB6', 'CV30', 'B38', 'CC12.1']:
        plot_mapped_coefficients(filepath, location_filepath, output_filepath, filename, mapped_coefficients, antibody, id_col)
        specific_coefficients = mapped_coefficients.loc[np.logical_and(mapped_coefficients.index.get_level_values(id_col)==antibody, mapped_coefficients.index.get_level_values('chain') == 'H')]
        wt_seq = ''.join(list(specific_coefficients[specific_coefficients['wild_type']==True].index.get_level_values('aa')))
        specific_coefficients = specific_coefficients[['coefficient']]
        specific_coefficients.index = [aa+str(location) for name,location,chain,aa in specific_coefficients.index]
        specific_coef_map = create_coef_matrix(specific_coefficients)
        try:
            binding_sites = pd.read_csv(location_filepath+antibody+'_contacts.csv', sep=',', header=0).number_antibody.values
        except OSError as ose:
            binding_sites = []
        plot_logoplot(output_filepath, filename, specific_coef_map, binding_sites, 'Positions', 'Coefficients', antibody, 2, wt_seq)

    full_l_distances, l_distances = create_levenshtein_map(metadata, id_col, heavy_col, light_col)
    l_linkage = create_linkage(full_l_distances)
    #plot_l_heatmap(output_filepath, filename, l_distances)
    plot_clustermap(output_filepath, filename, l_linkage, full_l_distances, metadata, id_col, y_col, patient_col, gene_col, '$IC_{50}$', 'patient', 'VH gene')
    plot_hierarchical_clust(output_filepath, filename, full_l_distances, l_linkage, 3)
