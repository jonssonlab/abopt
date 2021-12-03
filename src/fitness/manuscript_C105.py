import sys
import fitness as fit


### Reference to estimator file
estimator_filename = '../../output/estimator/NeutSeqData_VH3-53_66_aligned_mapped_coefficients.csv'
file_ab_locations = '../../data/location/C105_locations.csv'

antibody_metadata = '../../data/meta/antibody_list.tsv'
antibody_list = ['C105', 'C105_TH28I_YH58F']

energy_estimator_path = '/Applications/foldxMacC11.tar_/'

### Mutationally scan virus in bound and unbound forms of C105 and C105opt

### Instantiate fitness object


    
fitdata = fit.Fitness(antibody_metadata_file=antibody_metadata, antibodies=antibody_list, energy_estimator='foldx', energy_estimator_path= energy_estimator_path)


pdb_names = [fitdata.pdb(antibody) for antibody in antibody_list]


### Repair antibody/viral receptor original structure, outputs to output/repair

#fitdata.repair(pdb_names)

### Remove antibody from WT structure and repair these structures

#fitdata.remove(pdb_names, chain_type= 'antibody', property='repair')
#fitdata.remove(pdb_names, chain_type= 'virus', property='repair')

pdb_list = [pdb_name + '_Repair_less_ab' for pdb_name in pdb_names]

### Repair structures 
# fitdata.repair(pdb_list,property='remove')

pdb_list_repair_less_ab = [pdb_name + '_Repair_less_ab_Repair' for pdb_name in pdb_names]
pdb_list_repair = [pdb_name + '_Repair' for pdb_name in pdb_names]

pdb_list = pdb_list_repair + pdb_list_repair_less_ab


### Construct a string of locations to scan 

sars_cov2 = '../../data/location/SARS_CoV_2_RBD_locations.csv'



posscan = fitdata.construct_position_scan_string (antibody_name = 'C105', virus_sequence_file=sars_cov2,
                                                  filter_type='chain', filter=None,
                                                  scan_molecule='virus')




fitdata.scan (pdb_list=pdb_list, property='repair', scan_values=posscan, scan_molecule='virus')

### Calculate energies and output 

fitdata.energy(antibody='C105', pdb = '6xcm_Repair', pdb_less= '6xcm_Repair_less_ab_Repair')
fitdata.energy(antibody='C105_TH28I_YH58F', pdb = '6xcm_Repair_TH28I_YH58F_Repair',
               pdb_less= '6xcm_Repair_TH28I_YH58F_Repair_less_ab_Repair')



