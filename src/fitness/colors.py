import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 


''' Colors ''' 
WILDTYPE =['#636363','#e6550d']
INCOCKTAIL =['#f0f0f0','#636363']


NUM_ABS = 18
c105opt = ['#d0d1e6', '#3690c0']

CLASS2_ESCAPE_VARIANTS =['E484K', 'Q493R']
CLASS3_ESCAPE_VARIANTS = ['R346S', 'N439K', 'N440K']
CLASS12_ESCAPE_VARIANTS = ['V367F','A475V', 'S477N', 'V483A']
CIRCULATING_VARIANTS  = ['N501Y', 'K417N', 'Y453F', 'S477R', 'N501Y', 'D614G', 'R683G','L452R'] 

ALL_VARIANTS = CLASS2_ESCAPE_VARIANTS + CLASS3_ESCAPE_VARIANTS + CLASS12_ESCAPE_VARIANTS + CIRCULATING_VARIANTS

CLASS_NAME = ['1', '2', '3', 'Unknown']
CLASS1_AB = ['B38','C105', 'CV30', 'CB6', 'CC121', 'C105_TH28I_YH58F']
CLASS2_AB = ['C119','C002', 'C112', 'C144', 'COVA2-39', 'C121-S1' ,'C121']
CLASS3_AB = ['C135','C110', 'REGN10987']
CLASSU_AB = ['REGN10933']


# COLORS 

NUM_AB_COLOR = sb.color_palette('Blues', 18)

CLASS_COLOR = sb.color_palette('colorblind', 4)
CLASS1_COLOR = sb.color_palette('Blues', len(CLASS1_AB))
CLASS2_COLOR = sb.color_palette('YlOrBr', len(CLASS2_AB))
CLASS3_COLOR = sb.color_palette('Greens', len(CLASS3_AB))
CLASSU_COLOR = sb.color_palette('Oranges', len(CLASSU_AB))

CLASS_COLOR_DICT = dict(zip(CLASS_NAME, CLASS_COLOR))
CLASS1_COLOR_DICT = dict(zip(CLASS1_AB, CLASS1_COLOR))
CLASS2_COLOR_DICT = dict(zip(CLASS2_AB, CLASS2_COLOR))
CLASS3_COLOR_DICT = dict(zip(CLASS3_AB, CLASS3_COLOR))
CLASSU_COLOR_DICT = dict(zip(CLASSU_AB, CLASSU_COLOR))

CLASS_AB_COLOR_DICT = {**CLASS1_COLOR_DICT, **CLASS2_COLOR_DICT, **CLASS3_COLOR_DICT, **CLASSU_COLOR_DICT}

NUM_AB_COLOR_DICT = dict(zip(range(NUM_ABS), NUM_AB_COLOR))

MAX_AB_DDG_BIND = -0.0
MIN_ACE2_DDG_BIND = -0.1  # based on finding minimum of infective variants

def get_antibodies(class_number):

    if class_number == '1':
        return CLASS1_AB
    if class_number == '2':
        return CLASS2_AB
    if class_number == '3':
        return CLASS3_AB
    if class_number == 'Unknown':
        return CLASSU_AB

def get_antibody_colors(ab_list):

    colorv = []
    for c in ab_list:
        colorv.append(CLASS_AB_COLOR_DICT[c])
    return colorv 


def get_number_antibody_colors(num_abs):

    colorv = []
    for c in num_abs:
        colorv.append(NUM_AB_COLOR_DICT[c])
    return colorv 


def get_class_colors(classes_list):

    colorv = []
    for c in classes_list:
        colorv.append(CLASS_COLOR_DICT[c])

    return colorv 


def palplot(pal, names, colors=None, size=1, label=True):
    n = len(pal)
    f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n).reshape(1, n),
              cmap=mpl.colors.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(n) - .5)
    ax.set_yticks([-.5, .5])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    colors = n * ['k'] if colors is None else colors
    if label: 
        for idx, (name, color) in enumerate(zip(names, colors)):
            ax.text(0.0+idx, 0.0, name, color=color, horizontalalignment='center', verticalalignment='center')
    return f

