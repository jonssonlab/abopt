import pandas as pd 
import os as os 


class FoldXInterface: 


    def __init__(self, foldx_path:str='/Applications/foldxMacC11.tar_'):
        """ Construct a FoldXInterface object and set path to FoldX 
        """        
        self.path = foldx_path

        
    def construct_position_scan_string (self, pdb_name='', location_file='', chain = None, filter= None):

        ''' read location file and extract locations to mutate including chain''' 
        ab_pos = pd.read_csv(location_file)
        ab_pos = ab_pos.fillna(0)
        ab_pos['pdb_location'] =  ab_pos.pdb_location.astype(int)
        ab_pos['scan_str'] = ab_pos.aa + ab_pos.chain + ab_pos.pdb_location.astype(str) +'a'

        if chain: 
            ab_pos = ab_pos.loc[ab_pos.chain == chain]

        if filter: 
            ab_pos  = ab_pos.loc[ab_pos.pdb_location.isin(filter)]

        posscan = ab_pos.scan_str
        posscan_str =  ",".join(posscan)
        return posscan_str


    def run_position_scan (self, pdb_file, scan_molecule, pos_scan, pdb_dir, out_dir):

        """ Run position scanning using FoldX
        """ 
        tag = pdb_file[:-4] + '_' + scan_molecule

        print(tag)
        print('Running position scan')
        command = self.path + "foldx --command=PositionScan --pdb-dir=" + pdb_dir + " --pdb=" + pdb_file
        command = command + " --rotabaseLocation=" + self.path + 'rotabase.txt'
        command = command + " --positions="+ pos_scan +" --out-pdb=false  --output-dir=" + out_dir 
        command = command + " --output-file=" + tag
        print(command)

        os.system(command)

    def run_build_model(self, pdb_name, mutations_file, pdb_dir, out_dir):
        """ Mutate a structure given a mutations file by running BuildModel using FoldX 
        """
        command = self.path + "foldx --command=BuildModel --pdb-dir=" + pdb_dir + " --pdb=" + pdb_name
        command = command + " --rotabaseLocation=" + self.path + 'rotabase.txt'
        command = command + " --mutant-file="+ mutations_file + " --output-dir=" + out_dir
        print(command)
        os.system(command)

    def rename_pdb_files(self, pdb_name, mutations, pdb_dir, out_dir):
        """ Rename PDB files after FoldX 
        """
        print(pdb_name)
        i = 1 
        for mut in mutations:
            pdb =  out_dir 
            command  = 'mv ' + pdb + '_' + str(i) + '.pdb ' + pdb + '_' + mut + '.pdb'
            print(command)
            os.system(command)
            i = i+1

    def create_individual_list(self, mutations, pdb, mutpath):
        mutstr = ''
        for mut in mutations.values:
            mutstr = mutstr + mut+ ';\n'            
            f  = open('./individual_list.txt', 'w')   
            f.write(mutstr)
            f.close()

        mutations.to_csv(mutpath + pdb[:-4] + '_mutations.txt', index=None)


    def rename_buildmodel_files(self, pdb_name, pdb_dir, indiv_list_path):
        """
        Renames files output from BuildModel to label with mutation
        :param pdb_name: original pdb that was mutated
        :param indiv_list_path: the mutation list used in BuildModel
        :return: None
        """
        with open(indiv_list_path, 'r') as f:
            full = f.read()
        broken = full.split(';\n')
        
        for ind in range(len(broken)):
            if broken[ind]:
                file_to_find = pdb_dir + pdb_name + '_' + str(ind+1) + '.pdb'
                new_name = pdb_dir + pdb_name + '_' + broken[ind] + '.pdb'
                os.rename(file_to_find, new_name)

    def run_repair_model(self,pdb_name,pdb_dir, out_dir):
 
        """ Repair a structural model by running RepairPDB using FoldX
        """
        command = self.path + "foldx --command=RepairPDB --pdb-dir=" + pdb_dir + " --pdb=" + pdb_name + '.pdb'
        command = command + " --rotabaseLocation=" + self.path + 'rotabase.txt'
        command = command +  " --output-dir=" + out_dir
        print(command)
        os.system(command)

    def run_multiple_repair_model(self,pdb_name, mutations, pdb_dir, out_dir):

        """ Repair multiple mutated structural models given a list of mutations by running RepairPDB using FoldX
        """
        i = 1
        for mut in mutations:
            print(mut)
            self.run_repair_model(pdb_name + '_' + mut +'.pdb',  pdb_dir, out_dir)
            i = i+1
