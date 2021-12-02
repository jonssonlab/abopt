import sys
sys.path.insert(1, '/Users/vjonsson/Google Drive/data/repository/abopt/src/pipeline')

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.stats as stats 
from sklearn import preprocessing
import fitness as fit


''' Reference to estimator file '''
estimator_filename = '../../output/estimator/NeutSeqData_VH3-53_66_aligned_mapped_coefficients.csv'
file_ab_locations = '../../data/location/C105_locations.csv'

antibody_metadata = '../../data/meta/antibody_list.tsv'
antibody_list = ['C105']

#antibody_list = ['B38', 'C105','CC121', 'CB6', 'COVA2-39','CV30']

fitdata = fit.Fitness(antibody_metadata, antibody_list)

pdb_name = fitdata.pdb('C105')


''' Repair antibody/viral receptor original structure, outputs to output/repair '''

fitdata.repair([pdb_name])

' Remove virus and antibody from WT structure and repair these structures '
fitdata.remove([pdb_name], chain_type= 'antibody', property='repair')
fitdata.remove([pdb_name], chain_type= 'virus', property='repair')

pdb_list = [pdb_name + '_Repair', pdb_name + '_Repair_less_ab']

' Repair these structures '
fitdata.repair(pdb_list,property='remove')


' Mutational scanning of C105 repaired pdbs '

' Construct list of locations to scan '

location_file = '../../data/location/SARS_CoV_2_RBD_locations.csv'

posscan = fitdata.construct_position_scan_string (pdb_name='C105', location_file=location_file, chain = None, filter= [472,501])
pdb_list = [pdb_name + '_Repair', pdb_name + '_Repair_less_ab']

fitdata.scan (pdb_list=pdb_list, property='repair', scan_type='location', scan_values=posscan, scan_molecule='virus')

' Constrain estimator to very large negative of positive coefficients '
cutoff = 1e-8
antibody = 'C105'
estimator = fitdata.constrain(data_type ='estimator', data_file=estimator_filename, antibody=antibody, cutoff = [-cutoff, cutoff], top=1000)

' Graph the estimator locations to mutate '

' Get pdb sequence locations on antibody that are less than 98 '
estimator = estimator.loc[estimator.pdb_location < '98']
