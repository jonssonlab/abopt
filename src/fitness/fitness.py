import energy as e 
import foldx as foldx
import pandas as pd
import structure
import os as os
import anndata as ad
import umap 
import numpy as np
import scipy.stats as stats
import colors as fmt 


class Fitness:

    def __init__(self, antibody_metadata_file:str, antibody_fitness_file: str=None, virus_fitness_file: str=None, antibodies:list=None, energy_estimator:str=None, energy_estimator_path:str=None, dropna:bool=False):
        
        """ Construct a fitness object with precalculated fitness landscapes, or
            an object for fitness landscape construction. 
        """        
        
        self.outpath = '../../output/'
        self.make_dirs(self.outpath)
        
        self.abfile = antibody_metadata_file

        ### If fitness landscape is pre-calculated, import and set fitdata
        self.fitdata = ad.AnnData()
        self.antibodies = antibodies
        self.embedding = pd.DataFrame() #Antibody distance

        ### If fitness landscape needs to generated, set energy estimator tool and path 
        if energy_estimator == None:            
            if antibody_fitness_file:
                self.import_fitness_landscapes(antibody_fitness_file, virus_fitness_file, self.abfile, dropna=dropna)
            else:
                raise ValueError('Provide either fitness estimator or fitness file')
        elif energy_estimator == 'foldx':
            self.energy_estimator = foldx.FoldXInterface(energy_estimator_path)
        else:
            raise ValueError('Not implemented')
            
        ### Set abdata with antibody metadata properties
        self.abdata = pd.read_table(self.abfile, sep=',')
        self.abdata['pdb_file'] = [pdb +'.pdb' for pdb in self.abdata.pdb]    
        self.abdata  = self.abdata.loc[self.abdata.antibody.isin(self.antibodies)]
        self.abdata['pdb_dir'] = [ '../../data/pdb/' + ab +'/'  for ab in self.antibodies]
        self.abdata['repair_dir'] = [ self.outpath + 'repair/' + ab +'/' for ab in self.antibodies]
        self.abdata['remove_dir'] = [self.outpath + 'remove/' + ab +'/' for ab in self.antibodies]
        self.abdata['constrain_dir'] = [self.outpath + 'constrain/' + ab +'/' for ab in self.antibodies]
        self.abdata['scan_dir'] = [self.outpath + 'scan/' + ab +'/' for ab in self.antibodies]
        self.abdata['energy_dir'] = [self.outpath + 'energy/' + ab +'/' for ab in self.antibodies]
        self.abdata['design_dir'] = [self.outpath + 'design/' + ab +'/' for ab in self.antibodies]
        self.abdata['mutate_dir'] = [self.outpath + 'mutate/' + ab +'/' for ab in self.antibodies]

        ### Construct dictionnaries for convenience 
        self.abpdb = dict(zip(self.abdata.antibody, self.abdata.pdb))
        self.abrbd = dict(zip(self.abdata.antibody, self.abdata.rbdchain))
        self.pdbrbd = dict(zip(self.abdata.pdb, self.abdata.rbdchain))

    def remove_antibodies(self,antibodies:list):

        """ Remove antibodies from this fitness object from antibodies list and from fitdata object """

        self.antibodies = [ab for ab in self.antibodies if ab not in antibodies]
        self.fitdata = self.fitdata[:,self.antibodies]        


    def make_dirs(self, outpath):

        dirs = ['repair', 'remove', 'constrain', 'scan', 'energy', 'design', 'mutate']

        if os.path.isdir(outpath) == False:
            mkdir(outpath)

        for dir in dirs:
            path = outpath + dir 
            if os.path.isdir(path) == False:
                mkdir(path)
                
    def check_dir(self, outpath):
        
        if os.path.isdir(outpath) == False:
            mkdir(outpath)
                
    # Get pdb name 
    def pdb(self, antibody: str):
        return self.abpdb[antibody]

    # Get RBD chain
    def rbdchain(self, antibody: str=None, pdb: str=None):
        if antibody == None:
            return self.abrbd[pdb]
        elif pdb == None:
            return self.abrbd[antibody]

        
    # Get name of directory given antibody or pdb name and property e.g. repair, mutate
    def pdbdir(self, pdb: str=None, property:str ='pdb'):

        property_dir = 'pdb_dir' 
        if property != 'pdb':
            property_dir = property + '_dir'
            
        return self.abdata.loc[self.abdata.pdb == pdb][property_dir].values[0]

    def outdir(self, pdb: str=None, property:str ='repair'):
        return self.abdata.loc[self.abdata.pdb == pdb][property + '_dir'].values[0]



    '''' Repair structure, if pdblist None then repair all structures in fitness '''
    
    def repair (self, pdb_list: list=None, property: str = 'pdb'):

        if pdb_list == None: # Run through all 
            pdb_list = self.abdata.pdb.unique()
        
        for pdb in pdb_list:
            pdb_name = pdb[0:4]
            pdb_dir = self.pdbdir(pdb_name, property=property)
            out_dir = self.outdir(pdb_name, property ='repair')
            self.check_dir(out_dir)
            self.energy_estimator.run_repair_model(pdb, pdb_dir, out_dir)

            
    ''' Remove a chain from the structure ''' 
    def remove(self, pdb_list: list=None, chain_type: str= None, property:str = 'repair'):

        for pdb in pdb_list:            
        
            pdb_name = pdb + '_Repair.pdb'
            pdb_dir = self.pdbdir(pdb, property='repair')
            out_dir = self.outdir(pdb, property ='remove')
            self.check_dir(out_dir)

            chains = structure.label_chains(pdb)

            if chains == None: # most likely not in  PDB
                rbd_chain = self.rbdchain(pdb_name=pdb)

            pdb_less = structure.remove_chains(pdb_dir, pdb_name , chains, chain_type, out_dir)
            

    
    ''' Constrain the solved estimator '''
    
    def constrain(self, data_type: str, data_file: str, antibody: str, cutoff: float , top: float):

        """ Constrain data in constrainfile using cutoff 
        """
        pdb  = self.pdb(antibody)

        if data_type == 'estimator':
            estimator = e.read_estimator(data_file, antibody)
            constrained = e.constrain_estimator_features(estimator, cutoffmin =cutoff[0], cutoffmax=cutoff[1], topmuts = top, filter_name='chain', filter='H')

        elif data_type == 'energy':
            energies = pd.read_table(data_file, sep=',')
            constrained = e.constrain_energies(energies, cutoffmin= cutoff[0], cutoffmax= cutoff[1], topmuts=top)

        return constrained


    def scan (self, pdb_list:list=None, property: str ='repair', scan_type:str='location',scan_values:str=None, scan_molecule:str='virus'):

        """ Mutational scanning 
        """

        for pdb in pdb_list:            

            pdb_file = pdb + '.pdb'
            if 'TH28I' in pdb: #kludge fix
                pdb_name = '6xcm_Repair_TH28I_YH58F'
                
            else: 
                pdb_name = pdb[0:4]

            pdb_dir = self.pdbdir(pdb_name, property='repair')
            out_dir = self.outdir(pdb_name, property ='scan')
            self.check_dir(out_dir)

            # All mutations 
            if scan_type == 'all': 
                pdbname, pdb_loc = e.read_pdb_locations(file_location=file_ab_locations)
            elif scan_type =='location':
                posscan_str =  scan_values 
            elif scan_type == 'chain': # position scan entire chain
                posscan_str =  ","


            self.energy_estimator.run_position_scan (pdb_file, scan_molecule,  scan_values, pdb_dir, out_dir)



    def energy (self,antibody:str=None, pdb:str=None, pdb_less:str=None, scan_molecule:str='virus'):

        """ Calculate energies 
        """

        pdb_file = pdb + '.pdb'
        if 'TH28I' in pdb: #kludge fix
            pdb_name = '6xcm_Repair_TH28I_YH58F'
                
        else: 
            pdb_name = pdb[0:4]

        pdb_dir = self.pdbdir(pdb_name, property='scan')
        out_dir = self.outdir(pdb_name, property ='energy')
        self.check_dir(out_dir)

        ddg = e.calculate_ddg_bind(antibody,pdb, pdb_less, scan_molecule=scan_molecule, in_dir=pdb_dir, out_dir=out_dir)

        ddg.to_csv(out_dir + 'ddg_' + antibody + '_' + scan_molecule + '_scanning.txt', index=None)


    def mutate(self, filename: str, chain: str, mlist_filename: str = None, llist_filename: str = None, repair: bool = False, output_dir: str = None):
        # input: filename str of molecular structure to mutate ==> filename
        # input: chain str of the chain to mutate used when mutating locations ==> chain
        # input: mutations array list FILE of mutations or locations comma delimited, eg: TH28I, YH58F ==> mlist_filename
        # input: location array list FILE of locations to mutate, all amino acids comma delimited, eg: 28-58, 75 ==> llist_filename
        # input: repair bool True if structure(s) to mutate requires repair after mutating ==> repair (this is a boolean value)
        # output directory: output_dir (the output should go in output_dir/output/mutate/) and figs in output_dir/output/figs/

        if os.path.isdir(output_dir) != None:
            os.mkdir(out_dir)

            output_dir ='../../ouptut/mutate/'
            
        # parse out the pdb and the path 
        pdb_name = filename[filename.rfind('/'):]
        pdb_dir = filename[:filename.rfind('/')]

        if mlist_filename !=None:
            fx.run_build_model(pdb_name=pdb_name, mutations_file=mlist_filename, pdb_dir=pdb_dir, out_dir = output_dir)
        elif llist_filename != None:
            raise NotImplementedError
        '''
        def mutate (pdb, mutations, pdb_dir, out_dir):
        
        if os.path.isdir(out_dir) == False:
        os.mkdir(out_dir)
        
        self.energy_tool.create_individual_list(mutations,pdb,out_dir)
        foldx.run_build_model(pdb, 'individual_list.txt', pdb_dir, out_dir)
        foldx.rename_buildmodel_files(pdb[:-4], out_dir, './individual_list.txt')
        
        '''


    def construct_position_scan_string (self,antibody_name:str='',pdb_name:str ='', virus_sequence_file: str='',
                                        chain:str = None, filter_type:str='position', filter:str = None,
                                        scan_molecule:str='virus'):

        
        if scan_molecule == 'virus':
            
            rbd_chain = self.rbdchain(antibody = antibody_name)
            ab_pos = pd.read_csv(virus_sequence_file)

            if filter_type == 'chain': 
                ab_pos = ab_pos.fillna(0)
                ab_pos['scan_str'] = ab_pos.aa + rbd_chain + ab_pos.pdb_location.astype(int).astype(str) +'a'
                posscan = ab_pos.scan_str

            elif filter_type == 'mutation':
                posscan= [] 
                if filter:
                    for m in filter:
                        posscan.append(m[0] + rbd_chain + m[1:])
            
            elif filter_type=='location':
                if filter:
                    ab_pos  = ab_pos.loc[ab_pos.pdb_location.isin(filter)]
                posscan = ab_pos.scan_str

            posscan_str =  ",".join(posscan)
                
        else:
            raise NotImplementedError

        return posscan_str


    def import_fitness_landscapes(self,antibody_fitness_file, virus_fitness_file, antibody_metadata_file = None, antibodies:list =None, dropna:bool=False):

        ''' Import antibody and virus fitness into AnnData object 
        Returns AnnData object where X: rows=mutations, cols=antibodies, obs: host_receptor (virus fitness file) 
        '''
        """
        Imports antibody and virus fitness into AnnData object
        Arguments:
         - antibody_fitness_file: path to CSV file containing antibody fitness data
         - virus_fitness_file: path to CSV file containing virus fitness data
         - antibody_metadata_file: (optional) path to prepopulated antibody data
        Returns:
         abdata: AnnData object where X: rows=mutations, cols=antibodies, obs: host_receptor (virus fitness file) 
        """

        abdf = pd.read_csv(antibody_fitness_file, header=0, index_col=0)
        vdf = pd.read_csv(virus_fitness_file, header=0, index_col=0)

        host_receptor = vdf.columns[0]

        self.antibodies = abdf.columns
        
        ''' Get subset '''
        if antibodies:
            abdf = abdf [antibodies]
            self.antibodies = antibodies

        merged = abdf.merge(vdf,how='inner', on='mut')

        if dropna: 
            merged = merged.dropna(axis=0)
        
        layer= {'ddg': merged[self.antibodies]}
        self.fitdata = ad.AnnData(merged[self.antibodies], layers =layer)
        
        if antibody_metadata_file: # populate the var data
            metadata= pd.read_table(antibody_metadata_file, header=0, index_col=0, sep=',')
            metadata = metadata.loc[self.antibodies]

        for c in metadata.columns:
            self.fitdata.var[c] = metadata[[c]]    
    
        self.fitdata.obs[host_receptor] = merged[host_receptor].values
        
        return self.fitdata



    def calculate_antibody_distance(self, algorithm:str = 'UMAP', n_neighbors:int=2, min_dist:float=1, n_components:int=2):
        
        data = self.fitdata.to_df()
        data = data.dropna(axis=0)
            
        ''' Graph distances between antibodies  '''

        fitness_data = data.values.transpose()
            
        if algorithm == 'UMAP':
            reducer = umap.UMAP(n_neighbors=n_neighbors,min_dist=min_dist)
        if algorithm == 'MDS':
            reducer = MDS(metric=False, n_components=n_components)
                    
        self.embedding = pd.DataFrame(reducer.fit_transform(fitness_data))
        self.embedding['type'] = 'UMAP'
        self.embedding['antibody'] = data.columns
        self.embedding['abclass'] = self.fitdata.var.abclass.values
        self.embedding = self.embedding.set_index('antibody')
        return self.embedding 


    def compare_antibodies(self, antibody_1:str, antibody_2:str, test:str ='ttest',sig_min:float=5e-2, fc_min:float=0):

        ''' Calculate differences in binding energies between any two antibodies  
        Returns dataframe with locations 
        '''

        antibodies = [antibody_1, antibody_2]

        data = self.fitdata.to_df()[antibodies]
        data['location'] = data.index.str[1:-1]
        
        
        ''' Perform t test and compare accross locations designed ab vs wt  '''
        test = pd.DataFrame()

        for location in data['location'].unique():
            
            ds = data.loc[data['location'] == location]

            ab1mean = ds[[antibody_1]].values.mean()
            ab2mean = ds[[antibody_2]].values.mean()

            res = stats.ttest_ind(ds[[antibody_1]].values,ds[[antibody_2]].values)
            test = test.append(pd.Series([location, res.pvalue[0], ab1mean, ab2mean]), ignore_index=True)
            

        ab1m = antibody_1 +'_mean'
        ab2m = antibody_2 +'_mean'
        
        test = test.rename(columns={0:'location', 1:'pval', 2:ab1m, 3:ab2m})
        test['significant'] = test.pval < sig_min

        minval = min(test[ab2m].min(), test[ab1m].min())
    
        sm1 = test[ab1m]  + minval 
        sm2 = test[ab2m]  + minval
        test['fold_change'] = np.abs(sm1/sm2)

        return test


    def compare_antibody_groups (self, antibody_group_1:list, antibody_group_2:list,
                                 test:str ='ttest',sig_min:float=5e-2, fc_min:float=0):

        ''' Calculate differences in binding energies between any two antibody classes   
        Returns dataframe with locations 
        '''

        data = self.fitdata.to_df()
        antibodies = data.columns

        abs1 = [ab for ab in antibody_group_1 if ab in antibodies]
        abs2 = [ab for ab in antibody_group_2 if ab in antibodies]
        
        data['location'] = data.index.str[1:-1]
        
        ''' Perform t test and compare accross locations designed ab vs wt  '''
        test = pd.DataFrame()

        for location in data['location'].unique():
            
            ds = data.loc[data['location'] == location]

            ab1mean = ds[abs1].values.mean(axis=1).mean()
            ab2mean = ds[abs2].values.mean(axis=1).mean()

            res = stats.ttest_ind(ds[abs1].values.mean(axis=1),ds[abs2].values.mean(axis=1))
            # Correct for sample size
            s1 = len(abs1)
            s2 = len(abs2) 
            res_corr = res.pvalue*s1/s2
            test = test.append(pd.Series([location, res_corr, ab1mean, ab2mean]), ignore_index=True)            

        ab1m = 'group_1_mean'
        ab2m = 'group_2_mean'
        
        test = test.rename(columns={0:'location', 1:'pval', 2:ab1m, 3:ab2m})
        test['significant'] = test.pval < sig_min

        minval = min(test[ab2m].min(), test[ab1m].min())
    
        sm1 = test[ab1m]  + minval 
        sm2 = test[ab2m]  + minval
        test['fold_change'] = np.abs(sm1/sm2)

        return test
