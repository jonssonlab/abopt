import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import cocktail 
import colors as fmt 
from itertools import combinations
import scipy.stats as stats
import scipy
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
import anndata as ad 
import colorcet as cc
import numpy as np




def plot_cocktail_coverage(simsdata, show = True, legend=True):
    """
    Plot coverage of optimal cocktails for each coverage
    Arguments:
     - simsdata: data result of simulation
    """    
    figname  = 'cocktail_coverage'
    optdata = simsdata[simsdata.obs.isopt]
    
    data = optdata.obs.reset_index().sort_values('cocktail_label',ascending=False)

    tmp = data[['cocktail_label','cocktail_colors']].drop_duplicates() 
    colors = dict(zip(tmp.cocktail_label, tmp.cocktail_colors))

    ''' Graph cocktail coverage '''
    area = data.num_abs*20

    plt.figure(figsize=(3,2))
    sb.set(context='paper', style='ticks', font_scale=1.2)
    sb.scatterplot(data=data, x='cov', y='cocktail_label', hue='cocktail_label', legend=legend, palette=colors,s=area, marker='s', alpha=1)
    plt.xlim([0.4,1])
    plt.xlabel('Fraction $\Delta$RBD coverage')
    plt.tight_layout()
    plt.savefig('../../output/figs/' + figname  + '.png', dpi=300, transparent=True)
    
    if show:    
        plt.show()


def plot_cocktail_mixes(simsdata, abdata, show=True, legend=True):
    """
    Produce a plot of input cocktail mixes
    Arguments:
     - simsdata: data result of simulation
     - abdata: antibody data

    """
    figname = 'cocktail_mixes'

    optdata = simsdata[simsdata.obs.isopt]

    tmp = pd.DataFrame()
    for c in optdata.obs.cocktail_str:        
        s = pd.Series(list(c)).str.split(expand=True).transpose()
        tmp = pd.concat([tmp, s])
    
    d  = dict(zip(tmp.columns, abdata.var['label'] ))

    tmp = tmp.rename(columns=d).drop_duplicates()
    opt = optdata.obs.drop_duplicates('cocktail_label')
    tmp['cocktail_label']  = opt.cocktail_label.values
    tmp = tmp.set_index('cocktail_label').astype(int)
    tmp = tmp.sort_values('cocktail_label',ascending=False)
    
    col_colors = [fmt.CLASS_COLOR_DICT[i] for i in abdata.var.abclass]
    
    data = optdata.obs.reset_index().sort_values('cocktail_label',ascending=False)
    data = optdata.obs[['cocktail_label','cocktail_colors']].drop_duplicates()
    
    colors = dict(zip(data.cocktail_label, data.cocktail_colors))
    row_colors = [colors[c] for c in tmp.index]

    figsize = (3.8,3) 
    sb.set(context='paper', style='white')
    g = sb.clustermap(data=tmp,cmap=fmt.INCOCKTAIL, linewidths= 0.5, linecolor='white', row_cluster=False, col_cluster=False, row_colors= row_colors,col_colors=col_colors, figsize=figsize, square=True)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    plt.tight_layout() 
    plt.savefig('../../output/figs/' + figname + '.png', dpi=300, transparent=True)
    if show: 
        plt.show()

        
def plot_cocktail_landscape(simsdata, abdata, show=True, legend=True):
    """
    PLot the cocktail landscape
    Arguments:
     - simsdata: data result of simulation
     - abdata: antibody data
    """
    figname = 'cocktail_landscape'
    optdata = simsdata[simsdata.obs.isopt]

    d = dict(zip(optdata.obs.cocktail_str, optdata.obs.cocktail_label))

    ''' Get the number of resistant mutations for each cocktail per location and plot '''
    data = pd.DataFrame()
    for c in optdata.obs.cocktail_str.unique():
        layer = pd.DataFrame(abdata.layers[c], columns=abdata.var_names, index=abdata.obs_names)
        layer[c] = layer.min(axis=1)
        layer[d[c]] = layer[c]>0 
        layer['location'] = layer.index.str[1:-1]
        grouped = layer.groupby('location').sum()[[d[c]]]
        data = pd.concat([data, grouped], axis=1)

    data = data.transpose().reset_index().sort_values('index',ascending=False).set_index('index')
    t = optdata.obs[['cocktail_label','cocktail_colors']].drop_duplicates()
    colors = dict(zip(t.cocktail_label, t.cocktail_colors))
    row_colors = [colors[c] for c in data.index]

    figsize = (6,2.5)
    sb.set(context='paper', style='white')
    g = sb.clustermap(data=data,row_cluster=False, col_cluster=False, row_colors = row_colors, cmap='Reds',vmin = 0, vmax = 20,  linewidths= 0.5, linecolor='white',figsize=figsize)
    plt.xlabel('Location')
    plt.ylabel('Cocktail') 
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig('../../output/figs/' + figname + '_sum.png', dpi=300, transparent=True)
    if show: 
        plt.show()


def plot_simulation_properties(simsdata, coverages=None, height=1.5, aspect=1.5, col_wrap=4, xmin=0.1, xmax=6000, ymin=0.1, ymax=20, show=True ):
    """
    Plot the cocktail landscape
    Arguments
     - simsdata: data result of simulation

    """

    figname = 'simulation_properties'
    
    subsetsims = simsdata.obs


    if coverages:
        subsetsims = subsetsims.loc[subsetsims['cov'].isin(coverages)]

        for c in coverages:
            s = subsetsims.loc[subsetsims['cov'] == c].loc[subsetsims.isopt].cocktail_label.unique()[0]
            print('Coverage: ' + str(c) + '; Optimal cocktail: ' + str(s))  
                                      
    labels = ['lambda1','lambda2']

    hues = ['num_abs', 'isopt', 'pc_rcov', 'cocktail_label', 'objective']

    colordata = subsetsims[['cocktail_label', 'cocktail_colors']].reset_index().drop('sim_name', axis=1).drop_duplicates()

    cocktail_palette = dict(zip(colordata.cocktail_label, colordata.cocktail_colors))    
    colors = ['RdPu', 'Greys', 'Blues', cocktail_palette, 'Greens']

    hue_colors = dict(zip(hues, colors))

    sb.set(context='paper', style='ticks')
    
    for hue in hues:

        g = sb.FacetGrid(data=subsetsims, row='cov', hue=hue, palette=hue_colors[hue],
                         height=height, aspect=aspect, legend_out=True)
        g.map(plt.plot,labels[0], labels[1], marker='s', ms=4,ls='', alpha=0.7)
        plt.ylim([ymin,ymax])
        plt.xlim([xmin,xmax])
        plt.semilogy()
        plt.semilogx()
        g.set_axis_labels('$\lambda_1$','$\lambda_2$')
        plt.tight_layout() 
        plt.savefig('../../output/figs/' + figname + '_' + hue +'.png', dpi=300, transparent=True)
        g.add_legend( loc='lower right')
        plt.savefig('../../output/figs/' + figname + '_legend_' + hue  +'.png', dpi=300, transparent=True)

        if show: 
            plt.show()

def plot_cocktail_perturbations(perturbed_landscapes, simsdata, show=True , legend=True):

    figname = 'cocktail_perturbations'

    tmp = simsdata.obs[['cocktail_label','cocktail_colors']].drop_duplicates()

    colors = dict(zip(tmp.cocktail_label,tmp.cocktail_colors))
    
    sb.kdeplot(data=perturbed_landscapes, hue='cocktail' , x='cocktail_resistant', palette=colors, fill=True, legend=legend)
    plt.tight_layout()
    plt.savefig('../../output/figs/' + figname + '.png',dpi=300, transparent=True)
    
    if show:
        plt.show()
    


# Deprecated 
    
def analyze_cocktail(abdata, simsdata):
    """
    Analyze cocktail in terms of its virus sensitvity 
        Graph virus location sensitivities 
    Arguments:
     - simsdata: data result of simulation
     - abdata: antibody data
    Returns:
     - merged: merged cocktail data
    """

    cocktails = simsdata.obs['cocktail_str'].unique()

    for c in cocktails:

        layer = abdata.layers[c].dropna(axis=1).transpose()
        resistant = layer > 0

        resistant['sum'] = 1. - resistant.sum(axis=1)/len(layer.columns)
        resistant = resistant.reset_index()
        resistant['cocktail_str'] = c


        plt.figure(figsize=(3,3))
        sb.barplot(x= 'index', y='sum', data=resistant)
        plt.xticks(rotation=90)
        plt.title(c)
        plt.tight_layout()
        if show:
            plt.show()

    
    ''' Bar graph with the antibodies that are chosen, with virus coverage '''
    dfc = antibody_fitness[cocktail]
    dfc = dfc.reset_index()
    
    ''' Find min in rows and then pick those minimum positive ddgs => best chance at coverage '''
    dfc['min_ddg'] = dfc.iloc[:,1:].min(axis=1)
    dfc['ispos'] = dfc.min_ddg>0 
    
    cov = float(cocktailname)
    allmuts = len(dfc.mut.unique())
    uncovmuts = len(dfc.loc[dfc.ispos].mut.unique())
    covmuts = allmuts - uncovmuts 
    descov = allmuts*(cov)

    dfc['location'] = dfc.mut.str[1:-1]
    
    ''' All positive ispos ddgs are uncovered''' 
    dfu = dfc.loc[dfc.ispos]
    dfu['ddgn'] = dfu.min_ddg/dfu.min_ddg.max()    
    
    merged = vdf.merge(dfu, on='mut', how='inner')
    merged['minACE2']= -merged.ACE2
    merged['cocktail'] = cocktailname 

    ''' Scatterplot for each antibody '''
    plt.figure(figsize=(2.5,2.5))
    sb.set(context='paper', style='white')
    ax = sb.scatterplot(data=merged, y='ddgn', x = 'minACE2', alpha=0.8,hue ='location', palette='Set3')
    plt.axhline(y=0, color='grey',xmin=-1, xmax=1, lw=0.5)
    plt.axvline(x=0, color='grey',ymin=-1, ymax=1, lw=0.5)
    ax.legend().remove()
    plt.xlabel('ddg(ACE2/RBD)')
    plt.ylabel('ddg(Ab/RBD)')
    plt.xticks(rotation=0)
    plt.tight_layout() 
    plt.savefig('../../output/figs/combination_sensitivity' + cocktailname + '.png', dpi=300)

    ''' Do histogram/KDE plot '''
    plt.figure(figsize=(2.5,2.5))
    sb.set(context='paper', style='white')
    ax = sb.histplot(data=merged,  x = 'minACE2', alpha=0.5)
    ax.legend().remove()
    plt.xlabel('ddg(ACE2/RBD)')
    plt.xticks(rotation=0)
    plt.tight_layout() 
    plt.savefig('../../output/figs/combination_kde' + cocktailname + '.png', dpi=300)

    ''' Violin plot of all sensitive locations '''
    plt.figure(figsize=(3.5,1.5))
    sb.set(context='paper', style='ticks')
    sb.violinplot(data=merged, y='minACE2', x = 'location', color='white',scale='width', inner='point')
    ax = sb.stripplot(data=merged, y='minACE2', x = 'location', alpha=0.5, color='Red', jitter=True)
    ax.legend().remove()
    plt.axhline(y=0, color='grey', lw=0.5)
    plt.xticks(rotation=90)
    plt.ylabel('ddg(ACE2)')
    plt.tight_layout() 
    plt.savefig('../../output/figs/combination_infectionsensitivity_' + cocktailname +'.png', dpi=300)

    if graph: 
        plt.show()
        
    return merged 


# deprecated 
def plot_significance(stat_test, cond1, cond2, show=True):
    """
    Plot the significance of the data
    Arguments:
     - stat_test: dataframe with locations
     - cond1: condition 1
     - cond2: condition 2
    """
    locations = stat_test.loc[stat_test.significant]['location'].unique() 
    palette = sb.color_palette(cc.glasbey_dark, n_colors=len(locations))
    dcolors = dict(zip(locations, palette))
    show = False

    figsize = (2.5,2.5)
    plt.figure(figsize=figsize)
    sb.scatterplot(data=stat_test, x='log2_fold_change', y='log10_p',color='grey', alpha=0.5)
    sb.scatterplot(data=stat_test.loc[stat_test.significant], x='log2_fold_change', y='log10_p',color='grey', hue='location', legend=True, palette=dcolors)
    plt.xlabel('log2FC(' + cond1 + '/' +cond2)
    plt.tight_layout()
    if show:
        plt.show()

        
def plot_cocktail_differences(abdata, simsdata, c1, c2, sigloc, property='cocktail_resistant', show=True):
    """
    Plot significant differences
    Arguments:
     - abdata: antibody data
     - simsdata: data result of simulation
     - c1: cocktail 1
     - c2: cocktail 2
     - sigloc: sequence locations significantly different in c1 and c2.
    """

    figname = 'cocktail_differences'

    abdata_ss = cocktail.get_cocktail_subset([c1,c2], abdata, simsdata)    

    resistant = abdata_ss 
    resistant = resistant.drop(['antibody','antibody_scaled_ddg','index', 'ddg'], axis=1).drop_duplicates()
    grouped_res = resistant.groupby(by=['location','cocktail_label']).sum().reset_index().drop_duplicates()
    grouped_res = grouped_res.loc[grouped_res.infective > 0]

    ''' Rank most sensitive mutations  '''
    sensitive = resistant[resistant.infective == True]
    sensitive['resistance'] = sensitive.ACE2 - sensitive.cocktail_scaled_ddg
    sensitive = sensitive.sort_values('resistance', ascending=False)
    

    ''' First plot entire set  '''
    ace2 = abdata.obs
    ace2['location'] = ace2.index.str[1:-1]
    ace2['infective'] = ace2.ACE2 >= fmt.MIN_ACE2_DDG_BIND

    ace2_grouped = ace2.groupby(by='location').sum().reset_index()  
    ace2_grouped = ace2_grouped.loc[ace2_grouped.location.isin(sigloc)][['location', 'infective']].set_index('location')

    ''' Get cocktail colors ''' 
    tmp = simsdata.obs[['cocktail_label','cocktail_colors']].drop_duplicates() 
    colors = dict(zip(tmp.cocktail_label, tmp.cocktail_colors))

    ''' Get antibody labels '''
    col_dict = dict(zip(abdata.var_names, abdata.var['label']))

    ''' Find Top '''
    area = (sensitive.iloc[0:48,:].cocktail_scaled_ddg+1)*100

    sensitive['area'] = (np.abs(sensitive.cocktail_scaled_ddg)).values**2*10
    
    figsize = (5,2)
    plt.figure(figsize=figsize)
    sb.scatterplot(data=sensitive.iloc[0:48,:],size='area', x= 'location', y= 'resistance', hue='cocktail_label',palette = colors,alpha=0.6, style='cocktail_label')
    plt.legend([])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('../../output/figs/' + figname + '_striplot_top' + c1 + '_' +  c2 +'.png', dpi=300, transparent=True)
    plt.show()

    figsize = (5,2)
    plt.figure(figsize=figsize)
    sb.barplot(data=sensitive.iloc[0:48,:], x='mut', y= 'resistance', hue='cocktail_label' ,palette = colors,alpha=0.6)
    plt.legend([])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('../../output/figs/' + figname + '_barplot_top' + c1 + '_' +  c2 +'.png', dpi=300, transparent=True)
    plt.show()


    ''' Group by location and plot ''' 
    legend = False
    show = True

    jpdata = abdata_ss.loc[abdata_ss.ACE2 >= fmt.MIN_ACE2_DDG_BIND]
    jpdata = jpdata.loc[jpdata.cocktail_scaled_ddg > fmt.MAX_AB_DDG_BIND]

    g = sb.jointplot(data=jpdata, y='cocktail_scaled_ddg',x ='ACE2', hue='cocktail_label',
                    alpha= 0.6, legend=legend, palette=colors, linewidth=0.8,height=3, kind='kde', fill=False, ratio=5)

    g.plot_marginals(sb.rugplot, color="r", height=-0.15, clip_on=False)

    plt.tight_layout()
    plt.savefig('../../output/figs/compare_cocktails_jointplot_ ' + c1 + '_' +  c2 +'.png', dpi=300, transparent=True)
    if show:
        plt.show()

    ''' Individual locations '''
    legend=False
    figsize = (3,1)

    plt.figure(figsize=figsize)
    sb.set(font_scale=0.9, style='ticks')
    sb.heatmap(data=ace2_grouped.transpose(), cmap='Reds', annot=True)
    plt.tight_layout()
    plt.savefig('../../output/figs/' + figname + 'map_heatmap' + c1 + '_' +  c2 +'.png', dpi=300)
    plt.show()

    figsize = (4,2)
    plt.figure(figsize=figsize)    
    ax = sb.violinplot(data=resistant.loc[resistant.location.isin(sigloc)], x='location', y = property ,split=True, hue='cocktail_label',
                       palette=colors, legend=legend,  inner='stick', linewidth=0.2, dodge=False, scale='size')
    plt.legend([])
    plt.setp(ax.collections, alpha=.8)
    plt.axhline(y=0, lw=0.5, color='grey')
    plt.xticks(rotation=90)
    plt.title(property)
    plt.tight_layout()
    plt.savefig('../../output/figs/' + figname + '_sig_violin' + c1 + '_' +  c2 +'.png', dpi=300)

    ''' Group by location and plot ''' 
    legend = False
    show = True
    plt.figure(figsize=(2,2))
    sb.kdeplot(data=abdata_ss[abdata_ss.cocktail_resistant], x='cocktail_scaled_ddg', hue='cocktail_label',
                   alpha=1, legend=legend, palette=colors, linewidth=1.3, fill=False)
    plt.axvline(x=fmt.MAX_AB_DDG_BIND, linewidth=0.5, color='grey')
    plt.tight_layout()
    plt.savefig('../../output/figs/' + figname + '_kde_ ' + c1 + '_' +  c2 +'.png', dpi=300, transparent=True)
    if show:
        plt.show()

    area = (grouped_res.loc[grouped_res.location.isin(sigloc)].infective)*10
    legend=False
    figsize = (5,2)

    plt.figure(figsize=figsize)
    sb.scatterplot(data=grouped_res.loc[grouped_res.location.isin(sigloc)], x='location', y = property, hue='cocktail_label',
                   palette=colors, s=area, marker='s', alpha=0.9, legend=legend)
    plt.axhline(y=fmt.MAX_AB_DDG_BIND, linewidth=0.5, color='grey')
    plt.xticks(rotation=90)
    plt.title('cocktail_resistant')
    plt.tight_layout()
    plt.savefig('../../output/figs/' + figname + '_ map_sig' + c1 + '_' +  c2 +'.png', dpi=300)

    if show:
        plt.show()
