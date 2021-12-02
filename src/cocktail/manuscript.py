import sys

import cocktail
import fitness.fitness as fit
import plot as pl
import numpy as np 

# Filenames for virus and antibody fitness landscapes

antibody_fitness = '../../output/merge/rbd_ab_fitness_opt.csv'
virus_fitness = '../../output/merge/rbd_ace2_fitness.csv'
antibody_metadata= '../../data/meta/antibody_list.tsv'


# Create an antibody fitness landscape from existing fitness data 
fitness = fit.Fitness(antibody_metadata, antibody_fitness, virus_fitness, dropna=True)

# Exclude any antibodies not used in the algorithm 
fitness.remove_antibodies(['C135', 'C002-S2'])


# Run the simulation given the ligand1 and ligand2 fitness landscapes,
# fitness.fitdata is the AnnData object with antibody and virus landscapes 
# lam1, lam2: Ranges for lam1 and lam2 parameter values for cocktail algorithm 
# coverage: a minimum desired virus mutation coverage
# noise_sims:  number of simulations with added noise
# algorithm: algorithm choice A1, the only choice for now

coverage = np.arange(start =0.06, stop=0.55, step = 0.01)
numsims = 1

# Ranges for lambda1 and lambda2 parameter values for cocktail algorithm 
lam1 = np.logspace(start=-3 , stop=15, num=60, endpoint=True, base=2.0)
lam2 = np.logspace(start=-3 , stop=8, num=60, endpoint=True, base=2.0)

layer = 'ddg'

# The number of simulations with added noise
noise_sims = 0

''' Run simulations ''' 

cocktails = cocktail.run_simulations(fitness.fitdata, layer, coverage,lam1, lam2, noise_sims=noise_sims, algorithm = 'A1')

cocktail_sims_file = '../../output/cocktail/cocktails_allsims_current.csv'
simsdata = cocktail.import_cocktail_simulations(cocktail_sims_file, fitness.fitdata)

pl.plot_cocktail_coverage(simsdata)
pl.plot_cocktail_mixes(simsdata,fitness.fitdata)
pl.plot_cocktail_landscape(simsdata, fitness.fitdata)
pl.plot_simulation_properties(simsdata , coverages=[0.55, 0.65, 0.75, 0.85, 0.94], aspect=1.15)

''' Compare cocktails '''

c1 = 'C02'
c2 = 'C04'

sigdiff  = cocktail.compare_cocktails(fitness.fitdata, simsdata, c1, c2, property='cocktail_scaled_ddg', lowerbound_fc=-1.5, upperbound_fc=2)
pl.plot_cocktail_differences(fitness.fitdata, simsdata, c1, c2, sigdiff, property='cocktail_scaled_ddg')


''' Run perturbations ''' 

nsims = 500
perturbation = 'additive'

pdata = cocktail.perturb_fitness_landscape(fitness.fitdata, nsims=nsims, perturbation='additive',output_sims=True)

cocktails = ['C01','C02','C03','C04','C05','C06','C07']
perturbed_landscapes = cocktail.perturb_cocktails(simsdata, pdata, fitness.fitdata,cocktails)
pl.plot_cocktail_perturbations(perturbed_landscapes, simsdata, show=True , legend=True)
