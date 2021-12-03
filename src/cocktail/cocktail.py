import numpy as np
import cvxpy as cp
import pandas as pd
import seaborn as sb

import scipy.stats as stats
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
import anndata as ad 
import colorcet as cc

if __package__ is None or __package__ == "":
    import colors as fmt
else:
    from . import colors as fmt

def import_cocktail_simulations(cocktail_sims_file, abdata):
    """
    Import algorithm simulations into AnnData object
    Arguments:
     - cocktail_sims_file: CSV file with cocktail simulation data
     - abdata: variable names for the cocktail file
    Returns:
     - simsdata: optimal cocktail on the simulation
    """
    
    allsims = pd.read_csv(cocktail_sims_file)    
        
    names = get_cocktail_names(allsims[abdata.var_names])
    allsims = allsims.set_index('sim_name')
    allsims['cocktail_str'] = names.astype(str)
    allsims['cov'] = allsims['cov'].round(2)

    obscols = ['num_abs', 'cov', 'lambda1', 'lambda2', 'objective', 'noise', 'cocktail_str']
    simsdata = ad.AnnData(X=allsims[abdata.var_names], obs=allsims[obscols])

    compute_cocktail_fitness_landscape(cocktail_data = simsdata, fitness_data = abdata, output_sims=False)
    compute_optimal_cocktail(simsdata,abdata)

    return simsdata


def get_cocktail_names(data):
    """
    Get cocktail names
    Arguments:
     - data: name data
    Returns:
     - cocktail_names.values: the names of the cocktails
    """ 
    data = data.astype(int).astype(str)
    cocktail_names  = data.apply(''.join, axis=1)
    return cocktail_names.values


def compute_cocktail (antibody_fitness,virus_fitness, k, lmbda1, lmbda2, noise, algorithm):
    """
    Compute antibody cocktail given k coverage and l1, l2
    Arguments:
     - antibody_fitness: antibody fitness panda
     - virus_fitness: virus fitness panda
     - k: the k of k coverage
     - lmbda1: lambda1 parameter
     - lmbda2: lambda2 parameter
     - noise: the noise value of the data
     - algorithm: either A1 or A2, which algorithm to use
    Returns:
     - c.value: numpy ndarray with the minimized variable
     - result: the computed cocktail
    """
    fitness_matrix = antibody_fitness.values + noise  # add noise if specified
    virus_matrix = virus_fitness.values

    m,n = fitness_matrix.shape

    incidence_matrix = np.sign(-fitness_matrix).clip(min=0) # less than zero 
    virus_incidence_matrix = np.sign(virus_matrix - fmt.MIN_ACE2_DDG_BIND).clip(min=0) # ones that greater than zero clip to zero
    
    c = cp.Variable(n, boolean = True)

    if algorithm == 'A1':
        num_unmanaged = int(m*k) ## PREVIOUS
        constraints = [cp.sum(c) >= 1, cp.sum_smallest((incidence_matrix@c), num_unmanaged+1) >= 1] ## PREVIOUS
        objective = cp.Minimize(lmbda1*cp.norm1(c)+ cp.matmul(cp.sum(fitness_matrix, axis=0),c)-lmbda2*incidence_matrix@c@virus_incidence_matrix) ## PREVIOUS
        
    elif algorithm == 'A2':
        num_unmanaged = int(sum(virus_incidence_matrix)*k) ## UPDATED
        constraints = [cp.sum(c) >= 1, (incidence_matrix@c@virus_incidence_matrix) >= num_unmanaged + 1] ## UPDATED fixed no strict inequalities
        objective = cp.Minimize(lmbda1*cp.norm1(c)+ cp.matmul(cp.sum(fitness_matrix, axis=0),c)) ## UPDATED

    else:
        print('Exit: Algorithm not specified')

    problem = cp.Problem(objective, constraints)
    result = problem.solve()

    return c.value, result



def run_simulations(abdata, layer, coverage,gamma1, gamma2, noise_sims=0, algorithm='A1', sim_name='cocktail_algorithm', output_sims=True):
    """ 
    Run simulations for antibody fitness and virus fitness 
    Arguments:
     - antibody_fitness: file with fitness landscape of antibodies with respect to virus binding
     - virus_fitness: file with fitness landscape of virus fitness with respect to cell receptor binding  
     - coverage: float or array of minimum virus coverage
     - gamma1: float or array  of lambda1 values, tuning parameter controlling the  number of antibodies in mix 
     - gamma2: float, or array of lambda 2 values, tuning parameter controlling the infection  
     - output_sims: if True,  writes data to '../../output/cocktail/cocktails_allsims.h5ad' 
     Returns:
      - simsdata: data from running simulations 
    """
    
    if len(layer) > 0:
        abdf = pd.DataFrame(abdata.layers[layer], columns=abdata.var_names)
    else:
        abdf = abdata

    vdf = abdata.obs

    allresults = pd.DataFrame()
    allresults['antibody'] = abdf.columns
    objective_values = []
    
    ks = coverage 
    allsims = pd.DataFrame()

    ''' Subject this to noise ''' 
    for i in range(noise_sims+1): 

        noise_str = 'noise_'+ str(i)

        ''' Generate random noise ''' 
        random_noise = np.zeros(abdf.shape)
        if i > 0:
            random_noise = np.random.normal(loc=0.0, scale=1, size=abdf.shape)*1e-2

        ''' Loop through coverage '''
        allcov = pd.DataFrame()
        for k in coverage:
            print('coverage=' + str(k) + noise_str) 
            '''Grid search on gamma1, gamma2 '''
            for g1 in gamma1: 
                for g2 in gamma2:
                    results, objvals = compute_cocktail(abdf,vdf, k, g1, g2, random_noise, algorithm=algorithm)
                    print(results)
                    choices = abdf.columns[results.astype(bool)]
                    colname = str(1-k) + '_' + str(g1) + '_' + str(g2)
                    allresults[colname]=results
                    objective_values.append(objvals) 

            ''' Find minimum abs ''' 
            abt = allresults.set_index('antibody')
            abt = abt.transpose() 
            abt['num_abs']  = abt.sum(axis=1).round(0).astype(int)
            tmp= pd.Series(abt.index).str.split(pat='_',  expand=True)

            abt ['cov'] = tmp.iloc[:,0].values.astype(float).round(2)
            abt ['lambda1'] = tmp.iloc[:,1].values
            abt ['lambda2'] = tmp.iloc[:,2].values
            abt ['objective'] = objective_values 
            abt ['noise'] = noise_str

        allsims = pd.concat([allsims, abt]) 

    # Name cocktails
    cocktail_str = get_cocktail_names(allsims[abdata.var_names]) 
    allsims['cocktail_str'] = cocktail_str.astype(str)
    allsims = allsims.rename(columns={'index': 'sim_name'})

    obscols = ['num_abs', 'cov', 'lambda1', 'lambda2', 'objective', 'noise', 'cocktail_str']
    simsdata = ad.AnnData(X=allsims[abdata.var_names], obs=allsims[obscols])

    compute_cocktail_fitness_landscape(cocktail_data = simsdata, fitness_data = abdata, output_sims=True)
    compute_optimal_cocktail(simsdata,abdata)
    
    if output_sims: 
        simsdata.write('../../output/cocktail/cocktails_allsims.h5ad')

    return simsdata 


def compute_optimal_cocktail(simsdata, abdata):
    """
    Computes the optimal cocktail given simsdata. For each input coverage, populates isopt observation in simsdata, given the minimum number of antibodies for the maximum coverage. 
    Arguments:
     - simsdata: data result of simulation
     - abdata: antibody data
    """    
    coverages = simsdata.obs['cov'].unique() 
    isopt = []
    
    for c in coverages: 
    
        tmp = simsdata[simsdata.obs['cov']== c]
        minabs = tmp.obs.num_abs.min()
        maxval = tmp[tmp.obs.num_abs == minabs].obs.pc_rcov.max()        

        t1 = tmp.obs[tmp.obs.num_abs ==minabs]

        opt_cocktail = t1.loc[t1.pc_rcov == maxval].cocktail_str.values[0]

        isopt.append(opt_cocktail)
        
    d = dict(zip(coverages, isopt))
    isopt_col = [d[c]  for c in simsdata.obs['cov']]
    simsdata.obs['isopt'] = isopt_col == simsdata.obs.cocktail_str

    set_cocktail_labels(simsdata)


def get_cocktail_matrix( simsdata, abdata):
    """
    Get a matrix of the cocktail mixes
    Arguments:
     - simsdata: data result of simulation
     - abdata: antibody data
    """
    figname = 'cocktail_mixes'
    optdata = simsdata[simsdata.obs.isopt]

    set_cocktail_labels(optdata)

    
    tmp = pd.DataFrame()
    for c in optdata.obs.cocktail_str:        
        s = pd.Series(list(c)).str.split(expand=True).transpose()
        tmp = pd.concat([tmp, s])
    
    d  = dict(zip(tmp.columns, abdata.var['label'] ))

    tmp = tmp.rename(columns=d).drop_duplicates()
    opt= optdata.obs.drop_duplicates('cocktail_label')
    tmp['cocktail_label']  = opt.cocktail_label.values
    tmp = tmp.set_index('cocktail_label').astype(int)

    
    abclass_colors = [fmt.CLASS_COLOR_DICT[i] for i in abdata.var.abclass]
    col_colors = [abclass_colors]

    tmp = tmp.sort_values('cocktail_label',ascending=True)
    


def set_cocktail_labels(simsdata):
    """
    Set cocktail labels, labelling the optimal cocktails first
    then following with other cocktails
    Arguments: 
     - simsdata: data result of simulation

    """

    data = simsdata.obs.sort_values('cov')
    cocktails = simsdata.obs.cocktail_str.unique()

    optc = data[data.isopt].cocktail_str.unique()
    n = data[~data.isopt].cocktail_str.unique()

    noptc = []
    for c in n:
        if c not in optc:
            noptc.append(c)

    csdict= dict()
    
    for i in range(len(cocktails)):
        pad = '' 
        if i+1 < 10:
            pad = '0'
            
        csdict[i+1] = 'C' + pad + str(i+1)

    dc1 = dict(zip(optc, [csdict[i+1] for i in range(len(optc))]))
    dc2 = dict(zip(noptc, [csdict[i+1] for i in range(len(optc), len(noptc)+ len(optc))]))
    dc12 = {**dc1, **dc2}

    palette = sb.color_palette(cc.glasbey_light, n_colors=len(cocktails))
    dcolors = dict(zip(cocktails, palette))

    labels = [ dc12[c] for c in simsdata.obs.cocktail_str]
    colors = [ dcolors[c] for c in simsdata.obs.cocktail_str]
    
    simsdata.obs['cocktail_label'] = labels
    simsdata.obs['cocktail_colors'] = colors
    


def scale (data):
    """
    Scale data
    Arguments:
     - data: the data to scale
    Returns:
     - scaled: scaled data
    """
    minval, maxval = data.min(), data.max()
    scaled = [] 
    for f in data:
        scaledval = -f/minval
        if f>0: scaledval = f/maxval
        scaled.append(scaledval)
    return scaled


def get_cocktail_subset(cocktails, abdata, simsdata):
    """
    Produces a scaled subset of cocktail data 
    Arguments:
     - cocktails: list of cocktails
     - simsdata: data result of simulation
     - abdata: antibody data
    Returns:
     - data: scaled subset of cocktail data

    """
    data = pd.DataFrame()

    cocktail_names= dict(zip(simsdata.obs.cocktail_label, simsdata.obs.cocktail_str))
    
    for c in cocktails:
        tmp = pd.DataFrame(abdata.layers[cocktail_names[c]],columns=abdata.var_names).dropna(axis=1)        
        tmp['mut'] = abdata.obs_names
        tmp['fitness'] = tmp.min(axis=1) # fitness of the cocktail 
        tmp['cocktail_label'] = c
        tmp['ACE2'] = abdata.obs['ACE2'].values
        tmp = tmp.reset_index().drop('index',axis=1)
        
        melted = tmp.melt(id_vars=['mut', 'ACE2','cocktail_label', 'fitness'])
        melted = melted.rename(columns=dict(zip(['variable', 'value'], ['antibody', 'ddg'])))
        data = pd.concat([data, melted])

    data['location'] = data.mut.str[1:-1]
    data['cocktail_resistant'] = data.fitness > 0
    data['infective'] = data.ACE2 >= fmt.MIN_ACE2_DDG_BIND

    scaled_fitness = scale(data.fitness)
    scaled_ddg = scale(data.ddg)

    data['cocktail_scaled_ddg'] = scaled_fitness
    data['antibody_scaled_ddg'] = scaled_ddg
    data = data.reset_index()

    return data

def get_differential_binding_locations(abdata, simsdata, c1, c2, property='cocktail_resistant', label='cocktail_label',test= 'ttest',sig_min=5e-2, fc_min=0):

    """
    Calculate differences in binding energies between any two cocktails 
    Arguments:
     - simsdata: data result of simulation
     - abdata: antibody data
     - c1: cocktail 1
     - c2: cocktail 2
    Returns:
     - stat_test: dataframe with locations 
    """ 
    ''' Get cocktail subset '''
    data = get_cocktail_subset([c1,c2], abdata, simsdata)

    ''' Perform t test and compare accross locations designed ab vs wt  '''
    stat_test = pd.DataFrame()

    for location in data['location'].unique():

        ab1 = data.loc[data['location'] == location].loc[data[label] == c1][property]
        ab2 = data.loc[data['location'] == location].loc[data[label] == c2][property]

        ab1mean = ab1.mean()
        ab2mean = ab2.mean()

        res = stats.ttest_ind(ab1, ab2)          
        stat_test = stat_test.append(pd.Series([location, res.pvalue, ab1mean, ab2mean]), ignore_index=True)

    ''' Extract statistically significant locations '''

    stat_test = stat_test.rename(columns={0:'location', 1:'pval', 2:c1 +'_mean', 3:c2 +'_mean'})
    stat_test['significant'] = stat_test.pval < sig_min
    stat_test['log2_fold_change'] = np.log(np.abs(stat_test[c1+'_mean']/stat_test[c2+'_mean']))
    stat_test['log10_p'] = -np.log10(stat_test.pval)

    show = False
    
    return stat_test


def get_differential_binding_locations_all(abdata, simsdata, cocktail, property='antibody_scaled_ddg', label='cocktail_label',test= 'ttest',sig_min=5e-2, fc_min=0):
    """
    Calculate differences in binding energies between any two antibodies  
        Returns dataframe with locations 
    Arguments:
     - cocktail: cocktail to analyze
     - simsdata: data result of simulation
     - abdata: antibody data
    Returns:
     - stat_test: dataframe with locations 

    """
    ''' Get cocktail subset '''
    data = get_cocktail_subset([cocktail], abdata, simsdata)

    simsdata_opt = simsdata.obs[simsdata.obs.isopt]
    other_cocktails = simsdata_opt.loc[simsdata_opt.cocktail_label != cocktail].cocktail_label.unique()

    tmp = get_cocktail_subset(other_cocktails, abdata, simsdata)

    ''' Get mean by location for other cocktails''' 
    other_data = tmp.groupby(['mut']).mean().reset_index()
    other_data['location'] = other_data.mut.str[1:-1]

    ''' Perform t test and compare accross locations designed ab vs wt  '''
    stat_test = pd.DataFrame()

    for location in data['location'].unique():

        ab1 = data.loc[data['location'] == location].loc[data[label] == cocktail][property]
        ab2 = other_data.loc[other_data['location'] == location][property]

        ab1mean = ab1.mean()
        ab2mean = ab2.mean()

        res = stats.ttest_ind(ab1, ab2)          
        stat_test = stat_test.append(pd.Series([location, res.pvalue, ab1mean, ab2mean]), ignore_index=True)

    ''' Extract statistically significant locations '''
    stat_test = stat_test.rename(columns={0:'location', 1:'pval', 2:cocktail +'_mean', 3:'other_mean'})
    stat_test['significant'] = stat_test.pval < sig_min
    stat_test['log2_fold_change'] = np.log(np.abs(stat_test[cocktail+'_mean']/stat_test['other_mean']))
    stat_test['log10_p'] = -np.log10(stat_test.pval)

    locations = stat_test.loc[stat_test.significant]['location'].unique() 
    palette = sb.color_palette(cc.glasbey_dark, n_colors=len(locations))
    dcolors = dict(zip(locations, palette))

    show = False

    if show: 
        figsize = (2.5,2.5)
        plt.figure(figsize=figsize)
        sb.scatterplot(data=stat_test, x='log2_fold_change', y='log10_p',color='grey', alpha=0.5)
        sb.scatterplot(data=stat_test.loc[stat_test.significant], x='log2_fold_change', y='log10_p',color='grey', hue='location', legend=True, palette=dcolors)
        plt.xlabel('log2FC(' + cocktail + '/' + 'other')
        plt.tight_layout()
        plt.show()
    
    return stat_test

            

def compare_all_cocktails(abdata,simsdata, property='cocktail_resistant', lowerbound_fc = -1, upperbound_fc = 1):
    """
    Compares all cocktails
    Arguments:
     - simsdata: data result of simulation
     - abdata: antibody data
    
    """
    show = True

    ''' Get cocktail subset '''
    stats_all = pd.DataFrame()

    cocktails = simsdata.obs.loc[simsdata.obs.isopt].cocktail_label.unique()
    for c in cocktails:        

        stat_test = get_differential_binding_locations_all(abdata, simsdata, c, property=property)
        stat_test['cocktail_condition'] = c
        stat_test =stat_test.rename (columns={c+'_mean':'mean'})
        stats_all = pd.concat([stats_all,stat_test], axis=0)

    stats_all.to_csv('../../output/cocktail/stat_all.csv')

    siglocneg = stats_all.loc[stats_all.log2_fold_change < lowerbound_fc]
    siglocpos = stats_all.loc[stats_all.log2_fold_change > upperbound_fc]
    sigloc  = pd.concat([siglocneg, siglocpos])
    sigloc = sigloc.loc[sigloc.significant].location.unique()
    stats_all = stats_all.loc[stats_all.location.isin(sigloc)]
    stats_all = stats_all[['location', 'mean','cocktail_condition']]

    melted = stats_all.pivot(index=['location'], columns =['cocktail_condition'], values=['mean']).transpose()

    g = sb.clustermap(melted,row_cluster=False, col_cluster=False , cmap='RdBu_r',alpha=1, center = 0.0,figsize=(6,2.5))
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    plt.savefig('../../output/figs/compare_cocktails_all_heatmap ' + c +'.png', dpi=300)
    plt.show()
    
    DF_corr = melted.T.corr()
    DF_dism = 1 - DF_corr   # distance matrix
    linkage = hc.linkage(sp.distance.squareform(DF_dism), method='complete')

    labels = melted.reset_index().cocktail_condition
    print(labels)
    
    matrix = np.triu(DF_dism)

    DF_dism = DF_dism.rename(columns=dict(zip(DF_dism.columns, labels)))

    sb.clustermap(DF_dism, mask = np.triu(DF_dism),row_cluster=False, col_cluster=False , cmap='Blues',alpha=1, center = 0.5,figsize=(3,3))
    plt.xticks( labels.values)
    plt.savefig('../../output/figs/compare_cocktails_all_ ' + c +'.png', dpi=300)
    plt.show()
    

def compare_cocktails(abdata,simsdata, c1, c2, property='cocktail_resistant', lowerbound_fc = -1, upperbound_fc = 1):
    """
    Compares two given cocktails
    Arguments:
     - simsdata: data result of simulation
     - abdata: antibody data
    """
    show = True

    ''' Get cocktail subset '''
    abdata_ss = get_cocktail_subset([c1,c2], abdata, simsdata)    
    stat_test = get_differential_binding_locations(abdata, simsdata, c1, c2, property=property)

    siglocneg = stat_test.loc[stat_test.log2_fold_change < lowerbound_fc]
    siglocpos = stat_test.loc[stat_test.log2_fold_change > upperbound_fc]
    sigloc  = pd.concat([siglocneg, siglocpos])                        
    sigloc = sigloc.loc[sigloc.significant].location.unique()

    stat_test.to_csv('../../output/cocktail/stat_test.csv')

    return abdata_ss, sigloc

    
    print(sigloc)

    if property == 'cocktail_resistant':
        plot_cocktail_resistance(abdata_ss, abdata, simsdata, c1, c2, sigloc)

    if property == 'cocktail_scaled_ddg':
        plot_cocktail_resistance(abdata_ss, abdata, simsdata, c1, c2, sigloc, property='cocktail_scaled_ddg')

    
        
def compare_antibodies_in_cocktail():
    ''' Get cocktail subset '''
    data = get_cocktail_subset([c1,c2], abdata, simsdata)
    
    abs = data[['antibody', 'ddg','cocktail_label', 'mut', 'location', 'antibody_scaled_ddg']]
    abs['antibody_resistant'] = abs.ddg > fmt.MAX_AB_DDG_BIND
    
    grouped_abs = abs.groupby(by=['antibody','cocktail_label']).sum().reset_index()
    grouped_abs['frac_antibody_coverage'] = 1. - grouped_abs.antibody_resistant/len(data.mut.unique())

    print(grouped_abs)

    for c in abs.cocktail_label.unique():
        color = colors[c]
        figsize = (3,3)

        data = grouped_abs.loc[grouped_abs.cocktail_label==c]
        data = data.rename(columns=col_dict)

        plt.figure(figsize=figsize)
        sb.barplot(data=data,x='antibody', y='frac_antibody_coverage',
                   color =color,alpha=1)
        plt.title(c)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('../../output/figs/compare_cocktails_bar_ ' + c +'.png', dpi=300)

        if show:
            plt.show()

        '''
        plt.figure(figsize=figsize)
        sb.barplot(data=grouped_abs.loc[grouped_abs.cocktail_label==c], x='antibody', y = 'frac_antibody_resistant',
                   color=color, alpha=0.5)

        plt.tight_layout()
        plt.savefig('../../output/figs/compare_cocktails_ barplot' + c  +'.png', dpi=300)
        plt.show()
        '''


    ### ttest 


    c1abs = ','.join(data.loc[data.cocktail_str == c1].antibody.unique())
    c2abs = ','.join(data.loc[data.cocktail_str == c2].antibody.unique())
    cdict = dict(zip([c1,c2], [c1abs, c2abs]))

    data['cocktail_str_mut'] = data.cocktail_str + '_' + data.mut

    # Get all resistant mutations to antibody 
    cdata = data.drop(['antibody', 'index', 'ddg'], axis=1).drop_duplicates(['cocktail_str_mut'])

    # For graphing 
    grouped = cdata.loc[cdata.fitness > fmt.MAX_AB_DDG_BIND].groupby(['location', 'cocktail_str']).sum().reset_index()
    grouped_mean = cdata.loc[cdata.fitness > fmt.MAX_AB_DDG_BIND].groupby(['location', 'cocktail_str']).mean().reset_index()

    # parse infective, for each location do ttest betwen c1 and c2
    # Compute ttest
    pvals, c1means, c2means = [] ,[], []
    for location in cdata.location.unique():

        c1cond = (cdata.cocktail_str==c1) & (cdata.location == location)
        c2cond = (cdata.cocktail_str==c2)& (cdata.location == location)
            
        c1data = cdata.loc[c1cond]
        c2data = cdata.loc[c2cond]

        c1mean = c1data.resistant.sum() 
        c2mean = c2data.resistant.sum() 

        s, pval = scipy.stats.ttest_ind(c1data.resistant.values,c2data.resistant.values, axis=0)
        
        pvals.append(pval)
        c1means.append(c1mean)
        c2means.append(c2mean)

    tmp = pd.DataFrame()
    tmp['mutation'] = cdata.location.unique() 
    tmp['pvals'] = -np.log10(pvals)
    tmp['c1mean'] = c1means
    tmp['c2mean'] = c2means
    tmp['logfc'] = tmp.c1mean - tmp.c2mean
    tmp['de'] = c1 + '_vs_' + c2

    melt = tmp.melt(id_vars=['mutation' , 'pvals', 'de', 'logfc'])





def generate_noise(type='Gaussian', numsims= 1000, dimension = 1000):
    """
    Generates Gaussian noise
    """
    all_noise = pd.DataFrame()

    for i in range(numsims):
        noise = pd.DataFrame(np.random.normal(0,2,dimension))
        all_noise = pd.concat([all_noise, noise], axis=1)

    cols = range(len(all_noise.columns))
    all_noise.columns = cols

    return all_noise


def perturb_fitness_landscape(abdata, nsims=1, perturbation='multiplicative', scaling=1, output_sims=False):
    """
    Perturbs a given fitness landscape
    Arguments:
     - abdata: antibody data
    Returns:
     - allp: all perturbations generated of the landscape
    """
    perturbed_landscape = pd.DataFrame()
    
    meandist = abdata.to_df().mean().max()

    allp = pd.DataFrame()
    for n in range(nsims):
        tmp = pd.DataFrame()
        
        noise =  scaling*np.abs(np.random.normal(0,meandist,(abdata.shape[0],abdata.shape[1])))
        perturbed = np.add(abdata.X, noise)
        
        if perturbation == 'multiplicative':
            perturbed = pd.DataFrame(np.multiply(abdata.X, noise))
           
        pstr = 'p' + str(n)
        tmp = pd.DataFrame(perturbed, columns = abdata.var_names, index=abdata.obs_names)
        tmp['noise'] = 'noise_' + str(n+1)

        allp = pd.concat([tmp, allp])

    ace2 = pd.concat([abdata.obs['ACE2']]*nsims, axis=0) 
    allp = allp.reset_index()

    allp['mut_noise'] = allp.mut + '_' + allp.noise
    allp['perturbation'] = perturbation
    allp['ACE2'] = ace2.values
    allp = allp.set_index('mut_noise')

    return allp



def perturb_cocktails(simsdata, pdata, abdata, cocktails='all'):
    """
    Compares all cocktails
    Arguments:
     - simsdata: data result of simulation
     - pdata: data to perturb
     - abdata: antibody data
    """
    figname = 'perturb_cocktails'
    
    ''' Return the min landscape for all cocktails all noise '''

    cocktail_names = simsdata.obs[['cocktail_str', 'cocktail_label']].drop_duplicates()
    cocktail_dict = dict(zip(cocktail_names.cocktail_label,cocktail_names.cocktail_str))

    cp = pd.DataFrame() 
    for c in cocktails:
        antibodies = pd.DataFrame(abdata.layers[cocktail_dict[c]], columns=abdata.var_names).dropna(axis=1).columns
        cp[c] = pdata[antibodies].min(axis=1)

    cp['ACE2'] = pdata.ACE2        
    cp['mut'] = pdata.mut
    cp['noise'] = pdata.noise

    melted = pd.melt(cp, id_vars=['mut', 'noise','ACE2'], value_vars=cp.columns[:len(cocktails)])
    melted = melted.rename(columns = dict(zip(melted.columns, ['mut','noise','ACE2','cocktail', 'cocktail_fitness'])))
    melted['cocktail_resistant'] = melted.cocktail_fitness > fmt.MAX_AB_DDG_BIND
    melted['infective'] = melted.ACE2.astype(float) >= fmt.MIN_ACE2_DDG_BIND
    melted = melted.loc[melted.infective]
    melted = melted.loc[melted.cocktail_resistant]    

    grouped = melted.groupby(by=['cocktail', 'mut']).sum().reset_index()
    
    melted  = melted.groupby(by=['cocktail', 'noise']).sum().reset_index()

    return melted


def set_cocktail_names(cocktails):
    """
    Names cocktails
    Arguments:
     - cocktails: input antibody cocktails
    Returns:
     - cocktail_names.values: array of cocktail names
    """
    cocktails = cocktails.astype(int).astype(str)    
    cocktail_names  = cocktails.apply(''.join, axis=1)
    return cocktail_names.values



def get_unique_cocktails(cocktails):
    """
    Returns unique set of cocktails
    Arguments:
     - cocktails: input antibody cocktails
    Returns:
     - uc: unique cocktails
    """
    uc = cocktails.drop_duplicates()
    names = set_cocktail_names(uc)
    uc['cocktail_names'] = names
    uc = uc.set_index('cocktail_names') 
    return uc



def compute_cocktail_fitness_landscape(cocktail_data, fitness_data, output_sims=False):
    """
    Calculates the fitness landscape of a cocktail and writes to file
    Arguments:
     - cocktail_data: cocktails and associated data
     - fitness_data: the fitness data of the cocktails
    """
    fitness_cocktails = pd.DataFrame()
    cocktails = cocktail_data.to_df()

    pc_resistant_covered_list, pc_infective_resistant_covered_list = [] , [] 
    unique_cocktails = cocktail_data.obs['cocktail_str'].unique()
    
    for c in unique_cocktails:

        tmp = cocktail_data[cocktail_data.obs.cocktail_str == c].to_df()

        cocktail_vector= tmp.drop_duplicates().astype(int).transpose() 
        antibodies = list(cocktail_vector[cocktail_vector.iloc[:,0]==1].index.values)
        fitness = pd.DataFrame(fitness_data.layers['ddg'].copy(),columns=fitness_data.var_names)

    
        #print (fitness)
        #print (antibodies) 
    
        for col in fitness.columns:
            if col not in antibodies:
                fitness[col] = np.NaN

    
        # Calculate percent infective and non infective virus covered with this cocktail
        reduced_fitness = fitness.dropna(axis=1)

        ace2 = fitness_data.obs['ACE2'].to_frame()

        cond1 = (reduced_fitness > fmt.MAX_AB_DDG_BIND) # resistant
        cond2 = (ace2 >= fmt.MIN_ACE2_DDG_BIND) # infective
        
        resistant = (cond1.sum(axis=1) - len(cond1.columns)) == 0 # resistant to all antibodies in cocktail
        resistant_infective_mut = resistant.to_frame() & cond2.values
    
        # Percent resistant virus covered
        pc_resistant_covered = 1. - resistant.sum()/len(resistant)
        pc_infective_resistant_covered = 1. - resistant_infective_mut.sum().values[0]/cond2.sum().values[0]

        pc_resistant_covered_list.append([pc_resistant_covered, pc_infective_resistant_covered])

        fitness_cocktails = pd.concat([fitness_cocktails, fitness.min(axis=1)], axis=1)
        fitness_data.layers[c] = fitness

        #print(fitness)
    
    a, b = [], []
    
        
    d = dict(zip(unique_cocktails,pc_resistant_covered_list))

    for c in cocktail_data.obs['cocktail_str']:
        a.append(d[c][0])
        b.append(d[c][1])


    cocktail_data.obs['pc_rcov'] = np.round(a, 2)
    cocktail_data.obs['pc_ricov'] = np.round(b,2)

    #print(cocktail_data.obs)
    #print(cocktail_data.layers)

    if output_sims: 
        cocktail_data.write('../../output/cocktail/cocktail_fitness.h5ad')


''' TODO ''' 
def compare_perturbed_nominal_landscape(cocktails, cocktail_landscape, perturbed_landscape, infection_landscape):

    pm_all, pr_all = pd.DataFrame(),pd.DataFrame()

    nsims = len(perturbed_landscape.noise.unique())

    pr_all = pd.concat([pr_all, cocktail_landscape])
    pr_all['perturbation'] = 'none'
    
    for n in perturbed_landscape.noise.unique():

        perturb = perturbed_landscape.loc[perturbed_landscape.noise == n].iloc[:,0:-3]        
        perturbed_cocktail_landscape = compute_cocktail_fitness_landscape(cocktails, fitness_landscape = perturb)

        # Set the index change 
        perturbed_cocktail_landscape['mut'] = cocktail_landscape.index
        perturbed_cocktail_landscape = perturbed_cocktail_landscape.set_index('mut')
        perturbed_cocktail_landscape['perturbation'] = n

        pr_all = pd.concat([pr_all, perturbed_cocktail_landscape])

    pr_all.to_csv('../../output/cocktail/optimal_cocktails_perturbed.csv')
    pr_all = pr_all.reset_index()

    fitness = pr_all.merge(infection_landscape.reset_index(), how='inner', left_on='mut', right_on='mut')
    fitness['infective']  = fitness.ACE2 >= fmt.MIN_ACE2_DDG_BIND
    fitness['location'] = fitness.mut.str[1:-1]

    ''' Melted '''
    melted = pd.melt(fitness, id_vars=['mut','location', 'perturbation', 'ACE2'], value_vars=pr_all.columns[1:-1])
    melted = melted.rename(columns = dict(zip(melted.columns, ['mut','location','perturbation', 'ddG_ACE2','cocktail', 'ddG_AB'])))
    
    ''' Plot perturbation kde '''
    #cond = (melted.perturbation !='none') & (melted.ddG_AB > fmt.MAX_AB_DDG_BIND)
    cond1 = (melted.perturbation !='none') & (melted.ddG_AB > fmt.MAX_AB_DDG_BIND)& (melted.ddG_ACE2 > fmt.MIN_ACE2_DDG_BIND)
    cond2 = (melted.perturbation !='none') & (melted.ddG_AB > fmt.MAX_AB_DDG_BIND)& (melted.ddG_ACE2 > fmt.MIN_ACE2_DDG_BIND)

    nominal = melted.loc[cond1]
    perturbed = melted.loc[cond2]


    ''' Pertubed and infective ''' 
    cond1 = (fitness.perturbation =='none') & (fitness.ACE2 > fmt.MIN_ACE2_DDG_BIND)
    cond2 = (fitness.perturbation !='none') & (fitness.ACE2 > fmt.MIN_ACE2_DDG_BIND)


    #nominal = fitness.loc[cond1].groupby(['mut']).mean().iloc[:,0:-2].transpose()
    #perturbed = fitness.loc[cond2].groupby(['mut']).mean().iloc[:,0:-2].transpose()


    data = [nominal, perturbed]
    names = ['nominal', 'perturbed']
    d = dict(zip(names, data))

    '''
    for n in names:
        grouped = d[n].groupby(['mut']).mean().drop('ddG_ACE2', axis=1).transpose()

        figsize = (10,3)
        sb.set(context='paper', style='white')
        g = sb.clustermap(data=grouped,row_cluster=False, col_cluster=False, cmap='RdBu_r', vmin = -1, vmax=1,center=0,figsize=figsize)
        plt.xlabel('Location') 
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
        plt.title(n)
        plt.tight_layout()
        plt.savefig('../../output/figs/' + n + '_landscape.png', dpi=300, transparent=True)
        plt.show()
    '''
    
    
    nominal['noise'] = False
    perturbed['noise'] = True
    nom_pert = pd.concat([nominal, perturbed])


    cocktail_distance = pd.DataFrame()

    de = pd.DataFrame()
    
    for cocktail in perturbed.cocktail.unique():

        pvals , cmeans,omeans, tmp = [] ,[],[], pd.DataFrame()
        
        tmp['mut'] = perturbed.mut.unique()
        for mutation in perturbed.mut.unique():

            cond1 = (perturbed.cocktail==cocktail) & (perturbed.mut == mutation)
            cond2 = (perturbed.cocktail!=cocktail)& (perturbed.mut == mutation)
            
            C = perturbed.loc[cond1]
            O = perturbed.loc[cond2]
            O = O.groupby(['perturbation']).mean().reset_index()

            Cmean = C.ddG_AB.mean()
            Omean = O.ddG_AB.mean()

            s, pval = scipy.stats.ttest_ind(C.ddG_AB,O.ddG_AB, axis=0)
            
            pvals.append(pval)
            cmeans.append(Cmean)
            omeans.append(Omean)
        
        tmp['pvals'] = pvals
        tmp['cmean'] = cmeans
        tmp['omean'] = omeans
        tmp['de'] = cocktail + '_vs_rest'
        
        cocktail_distance= pd.concat([cocktail_distance, tmp], axis=0)

    cocktail_distance = cocktail_distance.fillna(0)
    significant = cocktail_distance.pvals< 5e-3
    cocktail_distance['sig'] = significant 
    cocktail_distance.to_csv('../../output/cocktail/cocktail_distance.csv')

    exit()


