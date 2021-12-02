import pandas as pd 


'''
### merge 
`abopt merge` merges energy landscape data for multiple structures

> `input: files array filenames including pathnames, of ddG binding for merging`
> `input: normalization str normalization method for binding energies, if any`
> `output: merged_raw file with a merged matrix of binding energies, or coupling energies`
> `output: merged_norm file with a merged and normalized matrix of binding energies, or coupling energies`
'''

def merge_ab_bloom(ab_name, mut, normalize=True):
    bloom, bloomval = get_bloom_data(normalize=normalize)
    ddg, ddgval = get_ab_ddg_data(ab_name, mut, normalize=normalize)
    merged = bloom.merge(ddg, on='mutation')
    #merged.to_csv(out_tmp + mut+'tmp.csv')                                                                                                
    return merged,bloomval,ddgval

def combine_data_sets(ddg_array):

    merged = pd.DataFrame()
    merged = ddg_array[0]

    for ddg_data in ddg_array[1:]: 
        merged = merged.merge(ddg_data, on='mut') 
    
    merged = merged.drop_duplicates(subset=['mut'], keep='first')
    merged = merged.set_index('mut')
    return merged 


def get_melted_ddgs(ddgs):
    ddgs = ddgs.reset_index()
    ddgs['location'] = ddgs.mut.str[3:-1]
    ddgsm = pd.melt(ddgs, id_vars=['mut', 'location'], value_vars=ddgs.columns[1:-1], var_name = 'ab',value_name='ddg')
    return ddgsm

'''API'''
def combine_antibody_binding_energy_data(abnames, scantype='virus'):
    ddgv = pd.DataFrame()
    for ab in abnames:
        print(ab)
        abddg = get_ab_ddg_data(ab, mutation = None, scantype=scantype, normalize = False)
        ddgv = pd.concat([abddg, ddgv])

    u.write(ddgv, filename='ddgs_virus_scanning')
    return ddgv
