import fitness as fit ## TODO
if __package__ is None or __package__ == "":
    import plot as pl
    import colors as fmt
else:
    from . import plot as pl
    from . import colors as fmt
import seaborn as sb
import matplotlib.pyplot as plt 
import pandas as pd

### Filenames for virus and antibody fitness landscapes 
antibody_fitness = '../../output/merge/rbd_ab_fitness_opt.csv'
virus_fitness = '../../output/merge/rbd_ace2_fitness.csv'
antibody_metadata= '../../data/meta/antibody_list.tsv'


### Create a fitness object from existing antibody and virus fitness data 
fitness = fit.Fitness(antibody_metadata, antibody_fitness, virus_fitness)

### Exclude some antibodies 
fitness.remove_antibodies(['C135', 'C002-S2'])

### Figure 3a, Figure 3b 
fitness.calculate_antibody_distance(algorithm ='UMAP')
pl.plot_fitness_landscape(fitness) 
pl.plot_antibody_distance(fitness) 


### Compare antibodies accross classes 

### Figure 3b 
def calculate_antibody_class_differences(fitness, group_1, group_2, group_3, group_4): 

    print('CLASS DIFF') 
    test12 = fitness.compare_antibody_groups(antibody_group_1 = group_1, antibody_group_2 = group_2)
    test13 = fitness.compare_antibody_groups(antibody_group_1 = group_1, antibody_group_2 = group_3)
    test14 = fitness.compare_antibody_groups(antibody_group_1 = group_1, antibody_group_2 = group_4)
    test23 = fitness.compare_antibody_groups(antibody_group_1 = group_2, antibody_group_2 = group_3)
    test24 = fitness.compare_antibody_groups(antibody_group_1 = group_2, antibody_group_2 = group_4)
    test34 = fitness.compare_antibody_groups(antibody_group_1 = group_3, antibody_group_2 = group_4)

    sig = test12.significant | test13.significant | test14.significant | test23.significant | test24.significant | test34.significant
    test= pd.concat([test12.fold_change, test13.fold_change, test14.fold_change, test23.fold_change, test24.fold_change, test34.fold_change], axis=1)

    test['fc'] = test.max(axis=1).values
    test['significant'] = sig
    test['location'] =test12.location

    c = test.significant & ((test.fc >=1.25) | (test.fc <= 0.75))
    locations = list(test.loc[c].location.values)

    return locations 

### Get the 4 classes of antibodies 
group_1 = fmt.get_antibodies(class_number='1')
group_2 = fmt.get_antibodies(class_number='2')
group_3 = fmt.get_antibodies(class_number='3')
group_4 = fmt.get_antibodies(class_number='Unknown')

locations = calculate_antibody_class_differences(fitness, group_1, group_2, group_3, group_4)

antibodies = group_1 +group_2 + group_3 + group_4

### Figure 3b


### Compare C105 and C105 optimized 
### Figure 3c 
antibodies = ['C105', 'C105_TH28I_YH58F']
test = fitness.compare_antibodies(antibody_1 = antibodies[0], antibody_2 = antibodies[1])
locations = list(test.loc[test.significant].location.values)
pl.plot_fitness_landscape(fitness, antibodies= antibodies, groupby = None, locations=locations, style='violin', figsize=(4,2.5), legend=True) 










