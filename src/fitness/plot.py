import sys

if __package__ is None or __package__ == "":
    import colors as colors
else:
    from . import colors
#import random 
#import pandas as pd
import seaborn as sb 
#import foldx as fx
import os 
#import structure 
#import energy as e     
import plotutils as pu 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import gcf

#import numpy as np 
#import scipy.stats as stats 
#import sklearn.preprocessing as pp

import utils
#from scipy import stats
#import itertools as it 

    


figs_dir = '../../output/figs/'

if os.path.isdir(figs_dir) == False:
    os.mkdir(figs_dir)


def plot_fitness_heatmap(fitness, data, show:bool=True, legend:bool=True, figsize=(10,3)):

    figname = 'fitness_heatmap'
    abclass = fitness.fitdata.var.abclass
    vhgene = fitness.fitdata.var.vhgene

    VH_GENE_COLOR_DICT = dict(zip(vhgene.unique(),sb.color_palette('Set3', len(vhgene.unique()))))
    CLASS_AB_COLOR_DICT = dict(zip(abclass.unique(), sb.color_palette('colorblind', len(abclass.unique()))))
    
    vhgene_colors = [ VH_GENE_COLOR_DICT[vh] for vh in vhgene]
    abclass_colors = [CLASS_AB_COLOR_DICT[c] for c in abclass]
    row_colors = [abclass_colors, vhgene_colors]
    
    sb.set(context='paper', style='ticks', font_scale=1)
    g = sb.clustermap(data=data,cmap='RdBu_r', center = 0,vmin=-1, vmax=1,
                      row_cluster=False, col_cluster= False, figsize=figsize, row_colors=row_colors)
    plt.yticks(rotation=90)
    plt.tight_layout()

    # Draw the legend bar for the classes                 
    for label in abclass.unique():
        g.ax_col_dendrogram.bar(0, 0, color=CLASS_AB_COLOR_DICT[label],
                            label=label, linewidth=0)
        g.ax_col_dendrogram.legend(loc="center", ncol=5, title ='Class')


    for label in vhgene.unique():
        g.ax_col_dendrogram.bar(0, 0, color=VH_GENE_COLOR_DICT[label],
                            label=label, linewidth=0)
        g.ax_col_dendrogram.legend(loc="center", ncol=5, title ='Class, VH gene')
        
    plt.savefig(figs_dir + figname + '.png', dpi=300, transparent=True)
    if show: 
        plt.show()


def plot_fitness_violin(fitness,data, hue:str=None, show:bool=True, legend:bool = True,
                        figsize=(10,3), scale_data: bool=True):

    figname = 'fitness_violin'
    abclass = fitness.fitdata.var.abclass
    vhgene = fitness.fitdata.var.vhgene

    ab_class_dict = dict(zip(fitness.fitdata.var.index, abclass))
    
    VH_GENE_COLOR_DICT = dict(zip(vhgene.unique(),sb.color_palette('Set3', len(vhgene.unique()))))
    CLASS_AB_COLOR_DICT = dict(zip(abclass.unique(), sb.color_palette('colorblind', len(abclass.unique()))))
    
    vhgene_colors = [ VH_GENE_COLOR_DICT[vh] for vh in vhgene]
    abclass_colors = [CLASS_AB_COLOR_DICT[c] for c in abclass]

    data = data.reset_index()

    melted = data.melt(id_vars='location', value_vars=data.columns[1:-1] )

    value = 'value' 
    if scale: 
        melted = scale(melted)
        value = 'scaled_value'
        
    if hue == 'abclass':
        melted['abclass'] = [ ab_class_dict[ab] for ab in melted.variable]
        hue_color = CLASS_AB_COLOR_DICT
    else:
        hue = 'variable'
        hue_color = 'Set1'


    grouped = melted.groupby(['location', 'variable']).sum().reset_index().groupby('variable').sum().reset_index()

    plt.figure(figsize=figsize)
    sb.set(context='paper', style='ticks', font_scale=1.1)    
    g = sb.violinplot(data=melted,x = 'location', y=value, hue = hue, palette = hue_color, split=True, inner='stick', linewidth=0.5, figsize=figsize, scale = 'width')
    plt.axhline(y =0, linewidth=0.3)
    plt.xticks(rotation=90)
    if legend == False:
        g.get_legend().remove()
    plt.tight_layout()        
    plt.savefig('../../output/figs/' + figname + '.png', dpi=300, transparent=True)
    
    if show: 
        plt.show()



def scale(data):
    
    maxval = data.value.max()
    minval = data.value.min()
    
    vals  = []
    for r in data.value:
        val = r/maxval
        if r  < 0:
            val = -r/minval
        vals.append(val)

    data['scaled_value'] = vals 

    return data 



def plot_fitness_stripplot(fitness,data, hue:str=None, show:bool=True, legend:bool = True, figsize=(10,3), scale_data: bool=True, markersize:int=20, alpha:float=1.0):

    figname = 'fitness_stripplot'
    abclass = fitness.fitdata.var.abclass
    vhgene = fitness.fitdata.var.vhgene

    ab_class_dict = dict(zip(fitness.fitdata.var.index, abclass))
    
    VH_GENE_COLOR_DICT = dict(zip(vhgene.unique(),sb.color_palette('Set3', len(vhgene.unique()))))
    CLASS_AB_COLOR_DICT = dict(zip(abclass.unique(), sb.color_palette('colorblind', len(abclass.unique()))))
    
    vhgene_colors = [ VH_GENE_COLOR_DICT[vh] for vh in vhgene]
    abclass_colors = [CLASS_AB_COLOR_DICT[c] for c in abclass]

    melted = data.melt(id_vars='location', value_vars=data.columns)

    value = 'value' 
    if scale: 
        melted = scale(melted)
        value = 'scaled_value'

    if hue == 'abclass':
        melted['abclass'] = [ ab_class_dict[ab] for ab in melted.variable]
        hue_color = CLASS_AB_COLOR_DICT
    else:
        hue = 'variable'

    plt.figure(figsize=figsize)
    sb.set(context='paper', style='ticks', font_scale=1.2)
    g = sb.stripplot(data=melted, x='location', y=value, hue = hue, palette = hue_color, alpha=alpha, jitter=0.1,s=markersize)
    if legend == False:
        g.get_legend().remove()
    plt.axhline(y = 0, linewidth=0.3)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('../../output/figs/' + figname +  '.png', dpi = 300, transparent=True)

    if show: 
        plt.show()



def plot_fitness_landscape(fitness, groupby:str = None, hue:str=None, antibodies:list=None, locations:list=None,
                           style ='heatmap',show=True, legend:bool=True, figsize=(10,3), markersize:int=None, alpha:float=1.0):

    figname = 'fitness_landscape'

    data = fitness.fitdata.to_df()
    data['location'] = data.index.str[1:-1] 


    if antibodies:
        a = data.columns
        abs1 = [ab for ab in antibodies if ab in a]
        data = data[abs1]
        antibodies = abs1

    data['location'] = data.index.str[1:-1]

    grouped = data

    if locations:
        grouped = grouped.loc[data['location'].isin(locations)].reset_index().drop('mut', axis=1)        
        if groupby == 'location':
            grouped = grouped.groupby('location').mean().reset_index().drop('mut',axis=1)
    
    else:
        grouped = data.drop('location', axis=1)

    if style == 'heatmap':
        plot_fitness_heatmap(fitness,data = grouped.transpose(), legend=legend, show=show, figsize=figsize)
    elif style == 'violin':
        plot_fitness_violin(fitness,data = grouped, hue = hue, legend=legend, show=show, figsize=figsize)
    elif style == 'stripplot':
        plot_fitness_stripplot(fitness,data = grouped, hue = hue, legend=legend, show=show, figsize=figsize, markersize=markersize, alpha=alpha)


    
def plot_antibody_distance(fitness, antibodies:list = None, algorithm='UMAP', annotate = True, show= True, legend = True, figsize = (3.75, 3.75)):

    ''' UMAP of antibody RBD fitness with respect to point mutations of RBD   
    '''

    figname = 'antibody_distance '

    ''' Read ab configuration file '''

    data = fitness.embedding
    if antibodies:
        data = fitness.embedding[antibodies]

    data = data.drop('type', axis =1)


    ''' Plot ''' 
    sb.set(context='paper', style='white', font_scale=1.2)
    f,ax = plt.subplots(figsize=figsize)
    sb.set(context='paper', font_scale=1)
    sb.scatterplot(data = data, x=0, y=1, hue ='abclass', palette='colorblind', legend=legend)
    plt.gca().set_aspect('equal', 'datalim')

    if annotate: 
        for j, lab in enumerate(data.index):
            ax.annotate(lab, (data.iloc[j,0]-0.8, data.iloc[j,1]+ 0.3))                
    
    plt.title(algorithm)
    plt.xlabel(algorithm + '1')
    plt.ylabel(algorithm + '2')
    plt.tight_layout()
    plt.savefig('../../output/figs/' + figname +  '_' + algorithm + '.png', dpi = 300, transparent=True)
    if show: 
        plt.show()



    
        
