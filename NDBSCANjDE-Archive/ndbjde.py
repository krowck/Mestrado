#imports
import matplotlib
from os import mkdir
import math
import numpy as np
import copy
import sys
import sobol_seq
import argparse
from nelder import *
from anneal import *
from hj import *
from statistics import median, stdev
from matplotlib import pyplot as plt
from time import gmtime, strftime, localtime, time, sleep
from random import uniform, choice, randint, gauss, sample
from cec2013 import *
from scipy.spatial import distance
from collections import Counter
from eucl_dist.cpu_dist import dist
#from eucl_dist.gpu_dist import dist as gdist
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from heapq import nlargest


import uuid
#import cProfile

def equal_maxima(x):
    return np.sin(5.0 * np.pi * x[0])**6

def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "... Done!\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:.0f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100.0, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def r8vec_print ( n, a, title ):
  print ( '' )
  print ( title )
  print ( '' )
  for i in range ( 0, n ):
    print ( '%6d:  %12g' % ( i, a[i] ) )



class DE:

    def __init__(self, pop_size):
        self.pop = [] #population's positions
        self.pop_gg = []
        self.pop_aux2 = []       
        self.m_nmdf = 0.00 #diversity variable
        self.diversity = []
        self.fbest_list = []
        self.nclusters_list = []
        self.full_euclidean = []   
        self.full_euclidean_aux = []
        self.crossover_rate = [gauss(0.5, 0.1) for i in range(pop_size)]
        self.mutation_rate = [0.5] * pop_size
        self.crossover_rate_T = [gauss(0.5, 0.1) for i in range(pop_size)]
        self.mutation_rate_T = [0.5] * pop_size

    def generateGraphs(self, fbest_list, diversity_list, max_iterations, uid, run, dim, hora, nclusters_list):
        plt.plot(range(0, max_iterations), fbest_list, 'r--')
        plt.savefig(str(uid) + '_' + str(dim) + 'D_' + str(hora) + '/graphs/run' + str(run) + '_' + 'convergence.png')
        plt.clf()     
        plt.plot(range(0, max_iterations), nclusters_list, 'r--')
        plt.savefig(str(uid) + '_' + str(dim) + 'D_' + str(hora) + '/graphs/run' + str(run) + '_' + ' clusters.png')
        plt.clf()                                                
        plt.plot(range(0, max_iterations), diversity_list, 'b--')
        #plt.ylim(ymin=0)
        plt.savefig(str(uid) + '_' + str(dim) + 'D_' + str(hora) + '/graphs/run' + str(run) + '_' + 'diversity.png')
        plt.clf()
        plt.plot(range(0, max_iterations), diversity_list, 'b--')
        plt.ylim(bottom=0)
        plt.savefig(str(uid) + '_' + str(dim) + 'D_'  + str(hora) + '/graphs/run' + str(run) + '_' + 'diversity_normalizado.png')
        plt.clf()
                                                       
        
    def updateDiversity(self):
        diversity = 0
        aux_1 = 0
        aux2 = 0
        a = 0
        b = 0
        d = 0
        
       
        for a in range(0, len(self.pop)):
            b = a+1
            for i in range(b, len(self.pop)):
                aux_1 = 0
    
                ind_a = self.pop[a]
                ind_b = self.pop[b]
    
                for d in range(0, len(self.pop[0])):
                    aux_1 = aux_1 + (pow(ind_a[d] - ind_b[d], 2).real)
                aux_1 = (math.sqrt(aux_1).real)
                aux_1 = (aux_1 / len(self.pop[0]))
    
                if b == i or aux_2 > aux_1:
                    aux_2 = aux_1
            diversity = (diversity) + (math.log((1.0) + aux_2).real)
            
        if self.m_nmdf < diversity:
            self.m_nmdf = diversity

        return (diversity/self.m_nmdf).real

    def f(self, x, y):
        return (10 + 9*np.cos(2*np.pi*3*x)) + (10 + 9*np.cos(2*np.pi*4*y))


    def contour_plot(self, xplot, yplot, sc, iteration, fig, ax):
        #plt.ion()
        #fig, ax = plt.subplots()

        x = np.linspace(-0.05, 1.05, 50)
        y = np.linspace(-0.05, 1.05, 50)

        #print(xplot, yplot)

        X, Y = np.meshgrid(x, y)
        Z = self.f(X, Y)

        #print(Z)

        plt.contourf(X, Y, Z, 20, cmap='RdBu');
        plt.scatter(xplot, yplot, s = 10, c= 'Green')
        plt.draw()
        sc.set_offsets(np.c_[xplot,yplot])
        fig.canvas.draw_idle() 
        plt.savefig('figure_' + str(iteration))
        plt.pause(0.0001)

    #fitness_function
    def fitness(self, individual):
        'to override'
        'rastrigin' 
        result = 0.00
        for dim in individual:
            result += (dim - 1)**2 - 10 * math.cos(2 * math.pi * (dim - 1))
        return (10*len(individual) + result)
   
    def generatePopulation(self, pop_size, dim, f): 
        ub = [0] * dim
        lb = [0] * dim
        for k in range(dim):
            ub[k] = f.get_ubound(k)
            lb[k] = f.get_lbound(k)

        vec = sobol_seq.i4_sobol_generate(dim, pop_size)
        
        for i in range(pop_size):
            lp = []
            for d in range(dim):
                #print(vec[i][d])
                lp.append(lb[d] + vec[i][d]*(ub[d] -  lb[d]))
            self.pop.append(lp)

        # s = np.random.uniform(0, 1, (pop_size, dim))

        # for i in range(pop_size):
        #     lp = []
        #     for d in range(dim):
        #         print(s[i][d])
        #         lp.append(lb[d] + s[i][d]*(ub[d] -  lb[d]))
        #     self.pop.append(lp)
        
        # for ind in range(pop_size):
        #     lp = []
        #     for d in range(dim):
        #         lp.append(uniform(lb[d],ub[d]))
        #     self.pop.append(lp)
        # print(self.pop)
        #sleep(10)

    def generateIndividual(self, alvo, dim, f):
        ub = [0] * dim
        lb = [0] * dim
        for k in range(dim):
            ub[k] = f.get_ubound(k)
            lb[k] = f.get_lbound(k)
        lp = []
        for d in range(dim):
            lp.append(uniform(lb[d],ub[d]))
        self.pop[alvo] = lp

    def generateNormalIndividual(self, best, dim, alvo, alpha, f):
        ub = [0] * dim
        lb = [0] * dim
        for k in range(dim):
            ub[k] = f.get_ubound(k)
            lb[k] = f.get_lbound(k)
        lp = []
        for d in range(dim):
            lp.append(np.random.normal(best[d], alpha))
            if lp[d] >= ub[d]:
                lp[d] = ub[d]
            elif lp[d] <= lb[d]:
                lp[d] = lb[d]
        self.pop[alvo] = lp

    def evaluatePopulation(self, pop, func, f):
        fpop = []
        for ind in pop:
            fpop.append(f.evaluate(ind))
        return fpop

    def evaluateIndividual(self, ind, func, f):
        find = 0
        find = f.evaluate(ind)
        return find
    def getBestSolution(self, maximize, fpop):
        fbest = fpop[0]
        best = [values for values in self.pop[0]]
        for ind in range(1,len(self.pop)):
            if maximize == True:
                if fpop[ind] >= fbest:
                    fbest = float(fpop[ind])
                    best = [values for values in self.pop[ind]]
            else:     
                if fpop[ind] <= fbest:
                    fbest = float(fpop[ind])
                    best = [values for values in self.pop[ind]]

        return fbest,best

    def printPopulation(self, fpop):
        for ind in range(0,len(self.pop)):
            print(self.pop[ind], fpop[ind])

    def printSubPopulation(self, subpop):
        for i in subpop:
            print(self.pop[i])

    def generateNewIndividualsFromSubPopulation(self, subpop, dim, f):
        ub = [0] * dim
        lb = [0] * dim
        #rint(subpop)
        for k in range(dim):
            ub[k] = f.get_ubound(k)
            lb[k] = f.get_lbound(k)
        
        for i in subpop:
            lp = []
            for d in range(dim):

                lp.append(uniform(lb[d],ub[d]))
            #print(lp)
            self.pop[i] = lp


    def takeSecond(self, elem):
        return elem[0]

    def generateNewIndividualsFromSubPopulationBiggerThan(self, subpop, dim, f, fpop):
        ub = [0] * dim
        lb = [0] * dim
        for k in range(dim):
            ub[k] = f.get_ubound(k)
            lb[k] = f.get_lbound(k)

        #print(len(subpop))

        fsubpop = []

        for i in subpop:
            fsubpop.append(fpop[i])
        
        subpop_better = nlargest(5, enumerate(fsubpop), key=lambda x:x[1])
        #print(subpop_better)
        subpop_better.sort(key=self.takeSecond, reverse=True)
        #print(subpop_better)

        for i in subpop_better:
            #print(i[0])
            subpop.pop(i[0])

        for i in subpop:
            lp = []
            for d in range(dim):
                lp.append(uniform(lb[d],ub[d]))
            #print(lp)
            self.pop[i] = lp        


    def rand_1_bin(self, ind, alvo, dim, wf, cr, neighborhood_list, m, f):
        vec_candidates = []
        ub = [0] * dim
        lb = [0] * dim

        vec_aux = sample(neighborhood_list, m)

        p1 = self.pop[vec_aux[0]]
        p2 = self.pop[vec_aux[1]]
        p3 = self.pop[vec_aux[2]]

        cutpoint = randint(0, dim-1)
        candidateSol = []
        
        for i in range(dim):            
            ub[i] = f.get_ubound(i)
            lb[i] = f.get_lbound(i)
            if(i == cutpoint or uniform(0,1) < cr):
                candidateSol.append(p3[i]+wf*(p1[i]-p2[i])) # -> rand(p3) , vetor diferença (wf*(p1[i]-p2[i]))i
            else:
                candidateSol.append(ind[i])
            #print("antes", candidateSol[i])
            #candidateSol[i] = self.michalewicz(candidateSol[i], lb[i], ub[i])
            #print("depois", candidateSol[i])




        #self.michalewicz(candidateSol, )

        return candidateSol

    def rand_2_bin(self, ind, alvo, dim, wf, cr, neighborhood_list, m):

        vec_aux = sample(neighborhood_list, m)

        p1 = self.pop[vec_aux[0]]
        p2 = self.pop[vec_aux[1]]
        p3 = self.pop[vec_aux[2]]
        p4 = self.pop[vec_aux[3]]
        p5 = self.pop[vec_aux[4]]

        cutpoint = randint(0, dim-1)
        candidateSol = []
        
        for i in range(dim):
            if(i == cutpoint or uniform(0,1) < cr):
                candidateSol.append(p5[i] + wf*(p1[i] - p2[i]) + wf*(p3[i] - p4[i]))
                #candidateSol.append(ind[i]+wf*(best[i]-ind[i])+wf*(p1[i]-p2[i])) # -> rand(p3) , vetor diferença (wf*(p1[i]-p2[i]))
            else:
                candidateSol.append(ind[i])

        return candidateSol
    
    def currentToBest_2_bin(self, ind, alvo, best, dim, wf, cr, neighborhood_list, m):
        vec_candidates = []

        vec_aux = sample(neighborhood_list, m)

        p1 = self.pop[vec_aux[0]]
        p2 = self.pop[vec_aux[1]]

        cutpoint = randint(0, dim-1)
        candidateSol = []
        
        for i in range(dim):
            if(i == cutpoint or uniform(0,1) < cr):
                candidateSol.append(ind[i]+wf*(best[i]-ind[i])+wf*(p1[i]-p2[i])) # -> rand(p3) , vetor diferença (wf*(p1[i]-p2[i]))
            else:
                candidateSol.append(ind[i])

        return candidateSol

    def currentToRand_1_bin(self, ind, alvo, dim, wf, cr, neighborhood_list, m):
        vec_candidates = []

        vec_aux = sample(neighborhood_list, m)

        p1 = self.pop[vec_aux[0]]
        p2 = self.pop[vec_aux[1]]
        p3 = self.pop[vec_aux[2]]

        cutpoint = randint(0, dim-1)
        candidateSol = []

        for i in range(dim):
            if(i == cutpoint or uniform(0,1) < cr):
                candidateSol.append(ind[i]+wf*(best[i]-ind[i])+wf*(p1[i]-p2[i])) # -> rand(p3) , vetor diferença (wf*(p1[i]-p2[i]))
            else:
                candidateSol.append(ind[i])

        return candidateSol

    def boundsRes(self, ind, f, dim):
        ub = [0] * dim
        lb = [0] * dim
        for k in range(dim):
            ub[k] = f.get_ubound(k)
            lb[k] = f.get_lbound(k)

        for d in range(len(ind)):
            if ind[d] <= lb[k]:
                ind[d] = lb[k] 
            if ind[d] >= ub[k]:
                ind[d] = ub[k] 

    def euclidean_distance_full2(self, dim):
        dist1 = np.zeros((len(self.pop), dim))
        self.pop = np.asarray(self.pop) #necessario para utilizar a funcao eucl_dist -- otimizacao da distancia euclidiana
        dist1 = dist(self.pop, self.pop)
        self.pop = self.pop.tolist() #necessario voltar para lista para nao afetar a programacao feita anteriormente
        dist1 = dist1.tolist()
        self.full_euclidean.append(dist1)

    def euclidean_distance_full3(self, dim, alvo):
        dist1 = np.zeros((len(alvo), dim))
        alvo = np.asarray(alvo) #necessario para utilizar a funcao eucl_dist -- otimizacao da distancia euclidiana
        dist1 = dist(alvo, alvo)
        alvo = alvo.tolist() #necessario voltar para lista para nao afetar a programacao feita anteriormente
        dist1 = dist1.tolist()
        return dist1

    def euclidean_distance(self, alvo, k, dim, archive):
        s = 0
        dist = []
        #print(archive)
        #print(alvo)
        for i in range(len(archive)):
            s = 0
            if k == i:
                dist.append(math.inf)
            else:
                for j in range(dim):
                    diff = archive[j] - alvo[j]
                    s += np.sqrt(np.linalg.norm(diff))
                #dist.append(s)
        #print(archive)
        #print("DISTANCIA >>>: ", dist)
        #sleep(1)
        #print(s)
        return s

    def euclidean_distance2(self, alvo, dim):
        dist1 = np.zeros((len(self.pop), dim))
        alvo = np.asarray([alvo])
        self.pop = np.asarray(self.pop) #necessario para utilizar a funcao eucl_dist -- otimizacao da distancia euclidiana
        dist1 = dist(alvo, self.pop)
        self.pop = self.pop.tolist() #necessario voltar para lista para nao afetar a programacao feita anteriormente
        dist1 = dist1.tolist()
        dist1 = dist1.pop()
        return dist1, dist1.index(min(dist1))

    def euclidean_distance_generic(self, alvo, population, dim):
        dist1 = np.zeros((len(population), dim))
        alvo = np.asarray([alvo])
        population = np.asarray(population) #necessario para utilizar a funcao eucl_dist -- otimizacao da distancia euclidiana
        dist1 = dist(alvo, population)
        population = population.tolist() #necessario voltar para lista para nao afetar a programacao feita anteriormente
        dist1 = dist1.tolist()
        dist1 = dist1.pop()
        return dist1, dist1.index(min(dist1))

    def euclidean_distance_individual(self, alvo, archive, dim):
        dist1 = np.zeros((len(archive), dim))
        alvo = np.asarray([alvo])
        archive = np.asarray(archive) #necessario para utilizar a funcao eucl_dist -- otimizacao da distancia euclidiana
        dist1 = dist(alvo, archive)
        archive = archive.tolist() #necessario voltar para lista para nao afetar a programacao feita anteriormente
        dist1 = dist1.tolist()
        dist1 = dist1.pop()
        return dist1, dist1.index(min(dist1))

    def generate_neighborhood(self, ind, m, dim, f):
        vec_dist = []
        vec_dist = list(self.full_euclidean[ind])
        neighborhood_list = [-1] * m
        for k in range(m):
            neighborhood_list[k] = vec_dist.index(min(vec_dist))            
            vec_dist[vec_dist.index(min(vec_dist))] = math.inf
        return neighborhood_list

    def reset_pop(self, labels, counter, ncluster, m, dim, f):
        temp = []
        temp_aux = []
        dist = []
        alvo = 0
        for k in range(ncluster):
            temp = [i for i,x in enumerate(labels) if x==k]
            temp_aux = sample(temp, len(temp)-m)
            print(temp, len(temp), m)
            for x in temp_aux:
                self.generateIndividual(x, dim, f)
        self.euclidean_distance_full2(dim)

    def normalize_pop_around_peaks(self, best_individuals, nclusters, popsize, dim, individuals_toReplace, f, fpop):
        k = 0
        individuals_per_peak = int(len(individuals_toReplace)/nclusters)

        for i in best_individuals:
            alpha = 0.05
            for j in range(0, individuals_per_peak):          
                self.generateNormalIndividual(self.pop[i], dim, individuals_toReplace[j + k], alpha, f)
                alpha = alpha + 0.05
                fpop[individuals_toReplace[j + k]] = f.evaluate(self.pop[individuals_toReplace[j + k]]) 
            k = k + individuals_per_peak
        for i in best_individuals:
            if k >= len(individuals_toReplace):
                break
            alpha = 0.05
            self.generateNormalIndividual(self.pop[i], dim, individuals_toReplace[k], alpha, f)
            k = k + 1

    def generate_individual_neighborhood(self, labels):
        aux = []
        indices = []
        temp = []
        dist = []
        alvo = 0
        indices = [i for i, x in enumerate(labels) if x == -1]
        aux = sample(indices, k)

    def normalized_distance(self, maximum, minimum):
        for i in range(0, len(self.full_euclidean)):
            for j in range(0, len(self.full_euclidean[i])):
                self.full_euclidean[i][j] = (self.full_euclidean[i][j] - minimum) / (maximum-minimum)

    def normalized_distance3(self, alvo):
        maximum = max([max(p) for p in alvo])
        minimum = min([min(p) for p in alvo])
        for i in range(0, len(alvo)):
            for j in range(0, len(alvo[i])):
                alvo[i][j] = (alvo[i][j] - minimum) / (maximum-minimum)
        return alvo

    def normalized_distance2(self, cvf):
        maximum = max(cvf) 
        minimum = min(cvf)

        if maximum == 0:
            maximum = 1

        for i in range(0, len(cvf)):
            cvf[i] = (cvf[i] - minimum) / (maximum-minimum)
        return cvf
        #print(cvf) 

    def update_jDE(self, pop_size):
        Fl = 0.1
        Fu = 0.9
        tau1 = tau2 = 0.1


        rand1 = uniform(0, 1)
        rand2 = uniform(0, 1)
        rand3 = uniform(0, 1)
        rand4 = uniform(0, 1)

        for ind in range(0,len(self.pop)):
            if rand2 < tau1:
                self.mutation_rate_T[ind] = Fl + (rand1 * Fu)
            else:                   
                self.mutation_rate_T[ind] = self.mutation_rate[ind]

            if rand4 < tau2:
                self.crossover_rate_T[ind] = rand3
            else:
                self.crossover_rate_T[ind] = self.crossover_rate[ind]     


    def michalewicz(self, x, minimum, maximum):  
        b_a = uniform(0, 1)
        a = uniform(0, 1)
        r = uniform(0.75, 1.0)

        if b_a < 0.5:
            y = x - minimum
            pw = math.pow(1 - r, 5)
            delta = y * (1.0 - pow(a, pw) )
            return x - delta
        else:
            y = maximum - x
            pw = math.pow(1 - r, 5)
            delta = y * (1.0 - pow(a, pw) )
            return x + delta

    def diferentialEvolution(self, pop_size, dim, max_iterations, runs, func, f, nfunc, accuracy, flag_plot, eps_value, archive_flag, maximize=True):

        crowding_target = 0
        neighborhood_list = []
        funcs = ["haha", "five_uneven_peak_trap", "equal_maxima", "uneven_decreasing_maxima", "himmelblau", "six_hump_camel_back", "shubert", "vincent", "shubert", "vincent", "modified_rastrigin_all", "CF1", "CF2", "CF3", "CF3", "CF4", "CF3", "CF4", "CF3", "CF4", "CF4"]
        #print(">>>>>>>>>>", str(funcs[1]))
        m = 0
        PR = [] #PEAK RATIO
        SR = 0.0
        hora = strftime("%Hh%Mm%S", localtime())
        mkdir(str(funcs[nfunc]) + '_' + str(dim) + 'D_' + str(hora))
        mkdir(str(funcs[nfunc]) + '_' + str(dim) + 'D_' + str(hora) +'/graphs')
        #to record the results
        results = open(str(funcs[nfunc]) + '_' + str(dim) + 'D_' + str(hora) + '/results.txt', 'a')
        records = open(str(funcs[nfunc]) + '_' + str(dim) + 'D_' + str(hora) + '/records.txt', 'a')
        clusters = open(str(funcs[nfunc]) + '_' + str(dim) + 'D_' + str(hora) + '/clusters.txt', 'a')
        results.write('ID: %s\tDate: %s\tRuns: %s\n' % (str(funcs[nfunc] ), strftime("%Y-%m-%d %H:%M:%S", gmtime()), str(runs)))
        results.write('=================================================================================================================\n')
        records.write('ID: %s\tDate: %s\tRuns: %s\n' % (str(funcs[nfunc] ), strftime("%Y-%m-%d %H:%M:%S", gmtime()), str(runs)))
        records.write('=================================================================================================================\n')
        avr_fbest_r = []
        avr_diversity_r = []
        avr_nclusters_r = []
        fbest_r = []
        best_r = []
        elapTime_r = []
        ub = f.get_ubound(0)
        lb = f.get_lbound(0)
        min_value_vector = []
        porcentagem = 0
        maximum_in_all_list = 0
        minimum_in_all_list = 0
        #runs
        for r in range(runs):
            list_aux = [0] * pop_size
            seconds_nelder_start = 0
            seconds_nelder_end = 0
            seconds_hj_start = 0
            seconds_hj_end = 0
            count_global = 0.0
            CVF = [0] * pop_size
            CVF_old = [9999] * pop_size
            CVF_test = [0] * pop_size
            niter = [0] * pop_size
            niter_best = [0] * pop_size
            elapTime = []
            archive = []
            fpop_archive = []
            niter_flag = 100
            start = time()
            records.write('Run: %i\n' % r)
            records.write('Iter\tGbest\tAvrFit\tDiver\tETime\t\n')
            cont_com_append = 0
            cont_sem_append = 0

            clusters.write('Run: %i\n' % r)
            best = [] #global best positions
            fbest = 0.00                    
            #global best fitness
            if maximize == True:
                fbest = 0.00
            else:
                fbest = math.inf

            #initial_generations
            self.generatePopulation(pop_size, dim, f)
            #fpop = f.evaluate
            fpop = self.evaluatePopulation(self.pop, func, f)
            self.pop_aux2 = self.pop
            
            # X = StandardScaler(with_mean=False).fit_transform(self.pop)

            # db = DBSCAN(eps=0.01, min_samples=1).fit(X)
            # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            # core_samples_mask[db.core_sample_indices_] = True
            # labels = db.labels_

            # # Number of clusters in labels, ignoring noise if present.
            # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

            #print('Estimated number of clusters: %d' % n_clusters_)

            # unique_labels = set(labels)
            # colors = [plt.cm.Spectral(each)
            #           for each in np.linspace(0, 1, len(unique_labels))]
            # for k, col in zip(unique_labels, colors):
            #     if k == -1:
            #         # Black used for noise.
            #         col = [0, 0, 0, 1]

            #     class_member_mask = (labels == k)

            #     xy = X[class_member_mask & core_samples_mask]
            #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            #              markeredgecolor='k', markersize=14)

            #     xy = X[class_member_mask & ~core_samples_mask]
            #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            #              markeredgecolor='k', markersize=6)

            # plt.title('Estimated number of clusters: %d' % n_clusters_)


            # plt.show()

            #Euclidean distance to calculate the neighborhood mutation
            self.euclidean_distance_full2(dim)
            self.full_euclidean = self.full_euclidean.pop()
            #print((self.full_euclidean))
            #sleep(100)
            maximum_in_all_list = max([max(p) for p in self.full_euclidean])
            
            for control in range(pop_size):
                self.full_euclidean[control][control] = maximum_in_all_list
            minimum_in_all_list = min([min(p) for p in self.full_euclidean])
            
            self.normalized_distance(maximum_in_all_list, minimum_in_all_list)

            self.full_euclidean_aux = self.full_euclidean

            

            fbest,best = self.getBestSolution(maximize, fpop)
            myCR = 0.0
            myF = 0.0
            cr_list = []

            if dim == 2 and flag_plot == 1:
                plt.ion()
                xplot = [0]
                yplot = [0]
                fig, ax = plt.subplots()
                #print("entrou")                
                sc = ax.scatter(xplot,yplot, s=2)
                self.contour_plot(xplot, yplot, sc, 0, fig, ax)
            avrFit = 9999999

            for i in range(0, pop_size):
                list_aux[i] = i

            #print(max_iterations)
            fpop_old = [0] * pop_size
            cont = 0
            #sleep(5)
            for iteration in range(max_iterations):

                update_progress(iteration/(max_iterations-1))

                if pop_size <= 200:
                    m=math.floor(3+10*((max_iterations-iteration)/max_iterations))
                else:
                    m=math.floor(3+10*((max_iterations-iteration)/max_iterations))
                m = 5

                avrFit = 0.00

                if flag_plot == 1:
                    xplot = []
                    yplot = []
                
                self.update_jDE(pop_size)                
                

                for ind in range(0,len(self.pop)):

                    if dim == 2 and flag_plot == 1:
                        xplot.append(self.pop[ind][0])
                        yplot.append(self.pop[ind][1])
                    
                    #crossover_rate[ind] = 0.1
                    myCR = self.crossover_rate_T[ind]
                    myF = self.mutation_rate_T[ind]

                    if uniform(0,1) < 1:
                        neighborhood_list = self.generate_neighborhood(ind, m, dim, f)
                        candSol = self.rand_1_bin(self.pop[ind], ind, dim, myCR, myF, neighborhood_list, m, f)
                    else:
                        neighborhood_list = self.generate_neighborhood(ind, m, dim, f)
                        candSol = self.currentToBest_2_bin(self.pop[ind], ind, best, dim, mutation_rate[ind], crossover_rate[ind], neighborhood_list, m)
                    
                    self.boundsRes(candSol, f, dim)

                    fcandSol = f.evaluate(candSol)

                    dist, crowding_target = self.euclidean_distance2(candSol, dim)

                    if maximize == True:
                        if fcandSol >= fpop[crowding_target]:
                            self.pop[crowding_target] = candSol
                            fpop[crowding_target] = fcandSol                            
                            self.mutation_rate[ind] = self.mutation_rate_T[ind]
                            self.crossover_rate[ind] = self.crossover_rate_T[ind]
                    avrFit += fpop[crowding_target]

                



                X = StandardScaler(with_mean=False).fit_transform(self.pop)

                db = DBSCAN(eps=eps_value, min_samples=1).fit(X)
                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                labels = db.labels_

                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

                temp = [0] * n_clusters_
                best_individuals = [0] * n_clusters_
                

                k = pop_size - Counter(labels).most_common(1)[0][1]
                idx = np.argpartition(fpop, -k)

                min_value_vector = [fpop[i] for i in idx[-k:] if fpop[i] < -accuracy]

                # --> Individuos em cada subpopulação.

                for j in range(n_clusters_):
                    temp[j] = [i for i,x in enumerate(labels) if x==j] 

                #print(n_clusters_)
                #print(len(max(temp, key=len)))
                #print(iteration)       
                

                # ------------------------------- START ARCHIVE TECHNIQUE ---------------------------------

                for i in range(n_clusters_):
                    if(len(temp[i]) > 1):
                        for a in range(len(temp[i])):
                            b = a+1
                            for k in range(b, len(temp[i])):
                                CVF[i] += np.linalg.norm(np.array(self.pop[temp[i][a]]) - np.array(self.pop[temp[i][k]]))
                        if (abs(CVF[i] - CVF_old[i]) < 0.1):
                            niter[i] += 1
                        else:
                            CVF_old[i] = CVF[i]
                            niter[i] = 0
                    else:
                        CVF[i] += 0
                        if (abs(CVF[i] - CVF_old[i]) < 0.1):
                            niter[i] += 1.0
                        else:
                            CVF_old[i] = CVF[i]
                            niter[i] = 0
                    #print(iteration, niter[i], temp[i])
                #print(n_clusters_)
                CVF = self.normalized_distance2(CVF)

                for i in range(n_clusters_):
                    dist_found = math.inf
                    temp_best = -999999
                    indice_best = -1
                    for x in temp[i]:
                        if fpop[x] > temp_best:
                            temp_best = fpop[x]
                            indice_best = x
                        best_individuals[i] = indice_best

                    if niter[i] >= niter_flag:
                        #print("entrou")
                        if (len(archive) == 0):
                           archive.append(self.pop[indice_best])
                           fpop_archive.append(fpop[indice_best])
                           #print(fpop_archive)
                           #sleep(10)
                        else:                        
                            for j in archive:
                                dist_arq = np.linalg.norm(np.array(self.pop[indice_best]) - np.array(j)) 
                                #print(self.pop[indice_best], j)
                                if dist_arq < dist_found:
                                    dist_found = dist_arq
                                    #print(dist_found)

                            if dist_found > 0.1:
                                #if len(archive) < 400:
                                #cont_com_append += 1
                                archive.append(self.pop[indice_best])
                                fpop_archive.append(fpop[indice_best])
                                self.generateNewIndividualsFromSubPopulation(temp[i], dim, f)
                                niter[i] = 0
                                CVF[i] = 0
                                CVF_old[i] = 99999
                                niter_flag += 0.5
                            else: 
                                #print("RESETANDO SEM APPEND")
                                cont_sem_append += 1
                                self.generateNewIndividualsFromSubPopulation(temp[i], dim, f)
                                niter[i] = 0
                                CVF[i] = 0
                                CVF_old[i] = 999999
                    if len(temp[i]) > 5:
                    #if len(max(temp, key=len)) > 20:
                        #print("entrou", len(max(temp, key=len)))

                        #archive.append(self.pop[indice_best])
                        self.generateNewIndividualsFromSubPopulationBiggerThan(temp[i], dim, f, fpop)
                        #niter[i] = 0
                        #CVF[i] = 0
                        #CVF_old[i] = 99999
 
                # ----------------------------- END ARCHIVE TECHNIQUE ----------------------------------

                #print(sorted(self.pop) == sorted(self.pop_aux2))
                X = StandardScaler(with_mean=False).fit_transform(self.pop)

                db = DBSCAN(eps=eps_value, min_samples=1).fit(X)
                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                labels = db.labels_

                # Number of clusters in labels, ignoring noise if present.
                n_clusters_2 = len(set(labels)) - (1 if -1 in labels else 0)

                self.nclusters_list.append(n_clusters_2)

                fpop = self.evaluatePopulation(self.pop, func, f)

                self.euclidean_distance_full2(dim)
                self.full_euclidean = self.full_euclidean.pop()
                for control in range(pop_size):
                    self.full_euclidean[control][control] = math.inf

                
                if dim == 2 and flag_plot == 1:# and (iteration == 0 or iteration == math.floor(max_iterations*0.25) or iteration == math.floor(max_iterations*0.50) or iteration == math.floor(max_iterations * 0.75)):
                    self.contour_plot(xplot, yplot, sc, iteration, fig, ax)
                    if iteration == 0:
                        sleep(7)
                #     plt.xlim(lb, ub)
                #     plt.ylim(lb, ub)

                #     plt.draw()                
                    
                #     sc.set_offsets(np.c_[xplot,yplot])
                #     fig.canvas.draw_idle()
                #     plt.pause(0.1)


                avrFit = avrFit/pop_size

                self.diversity.append(self.updateDiversity())

                fbest,best = self.getBestSolution(maximize, fpop)
                
                self.fbest_list.append(fbest)
                elapTime.append((time() - start))
                records.write('%i\t%.4f\t%.4f\t%.4f\t%.4f\n' % (iteration, round(fbest,4), round(avrFit,4), round(self.diversity[iteration],4), elapTime[iteration]))

            #print(len(archive))
            #print(archive)
            
            # distance_archive = []
            # distance_archive = self.euclidean_distance_full3(dim, archive)
            # print(distance_archive)
            # distance_archive = self.normalized_distance3(distance_archive)
            # print(distance_archive)
            #sleep(1)

            if(dim == 1):
                eps_value = 0.05

            X = StandardScaler(with_mean=False).fit_transform(self.pop)

            db = DBSCAN(eps=eps_value, min_samples=1).fit(X)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

            temp = [0] * n_clusters_
            best_individuals = [0] * n_clusters_

            #print(fpop)

            k = pop_size - Counter(labels).most_common(1)[0][1]
            idx = np.argpartition(fpop, -k)

            min_value_vector = [fpop[i] for i in idx[-k:] if fpop[i] < -accuracy]

            # --> Individuos em cada subpopulação.

            for j in range(n_clusters_):
                temp[j] = [i for i,x in enumerate(labels) if x==j] 


            for i in range(n_clusters_):
                temp_best = -999999
                indice_best = -1
                for x in temp[i]:
                    if fpop[x] > temp_best:
                        temp_best = fpop[x]
                        indice_best = x
                    best_individuals[i] = indice_best
                archive.append(self.pop[best_individuals[i]])

            #print(archive)

            archive2 = []
            fpop_archive = []
            fpop_archive = self.evaluatePopulation(archive, nfunc, f)
            #print(fpop_archive)
            X = StandardScaler(with_mean=False).fit_transform(archive)

            db = DBSCAN(eps=eps_value, min_samples=1).fit(X)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            n_clusters_2 = len(set(labels)) - (1 if -1 in labels else 0)

            temp = [0] * n_clusters_2
            best_individuals = [0] * n_clusters_2

            k = len(archive) - Counter(labels).most_common(1)[0][1]
            idx = np.argpartition(fpop_archive, -k)

            min_value_vector = [fpop_archive[i] for i in idx[-k:] if fpop_archive[i] < -accuracy]

            # --> Individuos em cada subpopulação.

            for j in range(n_clusters_2):
                temp[j] = [i for i,x in enumerate(labels) if x==j] 

            #print(len(temp), n_clusters_2)
            for i in range(n_clusters_2):
                temp_best = -999999
                indice_best = -1
                for x in temp[i]:
                    if fpop_archive[x] > temp_best:
                        temp_best = fpop_archive[x]
                        indice_best = x
                    best_individuals[i] = indice_best
                archive2.append(archive[best_individuals[i]])

            #print(archive2)
            #print(self.pop)

            #itermax = int((f.get_maxfes()*0.3/len(best_individuals))/dim)
            #itermax_archive = int((f.get_maxfes()*0.5/len(archive2))/dim)/2

            itermax_archive = 500
            print("Arquivo sem DBSCAN: ", len(archive), "Arquivo com DBSCAN", len(archive2), (itermax_archive), "niter_flag", niter_flag)
            rho = 0.75
            eps = 1.0E-30

            # print(itermax)

            #print(best_individuals, len(best_individuals))

            #LOCAL-SEARCH ROUTINE (HOOKE-JEEVES)

            #archive2 = [[-4.399190784181648, -1.8162558974403198], [-4.97130607209443, -4.06394231411632], [-1.4740581984204002, -5.0], [1.7548796511770928, 3.270465838520437], [2.621372308168648, 3.164828155490121], [-4.272576583939842, 1.8933544754675604], [1.5116683367831103, 0.16521749713350098], [-3.2374618582509833, 4.347676590541841], [-1.24375834336125, -1.9787273258726765], [-4.1590891177609315, 4.704064775690078], [-3.8888446744172427, 0.23894133592658115], [-0.48180657571474145, 3.1312094061740874], [-1.8645615075821151, 1.5888651909998153], [-4.88590820852977, -1.5728224115509128], [-0.2261949155224555, -1.9221199812901577], [0.10628118549022701, -4.310131193793298], [-0.1828658795002683, -3.425153868264696], [-3.6315030443549743, 3.3894131392722726], [3.3467932564126492, 0.7748079800085809], [-1.3442019135581191, 0.28060983147917273], [1.2666601666254156, -0.5679049097289061], [3.1713587913058, -0.8289719115563059], [2.1857602971948884, -0.7845133256331318], [-4.9380167044558405, 0.32978561817462604], [-1.5180943120446226, 0.6809985639909175], [-1.394518294122572, 3.5272537548560274], [-2.4559882816020457, -1.4370969184306344], [2.5672933037649472, -3.275445244390521], [0.8328658391134949, 3.66046460459078], [-4.944166265463501, 2.8649560750903325], [-1.3651279652006982, -1.203991574947801], [-3.802724616535142, 1.2989607997275705], [1.850383889291473, -2.2164696704800306], [-4.977553295848615, -2.5588208016645857], [-3.3684012518124535, 1.4580262506914385], [2.4556902117858646, -2.480457698698939], [-2.2802838293726384, 0.6366376625513164], [0.08039050579866709, 2.8640158144779337], [-3.1825534527714407, 1.0242966357920438], [2.6823119734654877, 2.0374492725518403], [-4.927760446203599, -0.4042500736128154], [-2.798171208702864, 1.950220127769502], [0.9716927346610851, -2.5100432688795875], [-2.946838127624515, 5.0], [-2.9580560447269844, -4.494175194597774], [-1.2000001947103667, -0.07651765788002951], [-0.9550131647129071, 1.162328388776517], [-3.8832236706413257, -2.2966074148979794], [-3.1089459873634375, 2.07939717409105], [3.698915622374114, -1.6805212598759909], [0.6618052423701224, -1.6215614419095346], [2.436172887984423, -1.0037003171338739], [-1.1529020638937737, -0.928181401027577], [-2.9695857332621833, -0.15217921599606649], [-0.9142226015477395, -3.8109609522742534], [4.07788729878062, 4.508656611629853], [-4.025768304531081, 2.4723706660199176], [-4.38000660579726, -2.8499188944236105], [-3.2278428624799615, -0.7558350993266506], [1.9816737368777426, -3.4815066761009237], [-4.858317323980683, 3.915257712647175], [-0.5353368302830852, -1.1956051429317611], [-2.046549403831175, -0.5374627396960644], [-0.6068488727828674, 0.2884131067611335], [-0.8659910413378064, 2.213518003249555], [-4.811825831056202, 0.6333170117993305], [-1.619874003191928, -3.248609123374787], [-1.3901930484014495, -4.3122305057996515], [1.7577435814843456, 0.4742590402492852], [-1.2648661983665166, 1.2954448008775017], [4.017752238179063, -1.7928859136623616], [3.467070043952051, 2.54111971396697], [4.447169400148064, -1.9598099097166648], [-4.38996014729009, -4.337791907836462], [4.5687013126061995, -2.749878653019339], [3.28605535633503, -0.12983703983132777], [-4.719437267747506, 1.681122576159829], [-3.7721755549488636, -4.646671649403606], [-4.11082663814106, 1.421477095210701], [3.3077034443064974, -3.8030076987745867], [-2.1045263032270984, -2.737991876938009], [1.8489330127149814, -0.6890158766542518], [1.6644128459951992, 4.747004433972307], [2.2523994748480853, 1.5957372548098387], [-3.3625747531753727, -1.0732306605171962], [-4.416965605199391, 1.5522140595584186], [-1.5714076478057424, 1.4256209041590366], [0.532405080346157, -4.999890956238866], [-2.4471794237606495, 1.0726241912495809], [-0.7943395225736444, -4.617061680300663], [0.05550535218314681, 4.358367601445294], [4.845898226307823, -0.46938735238848095], [-5.0, 4.704064775690078], [3.1027139256129668, -4.225318363834675], [4.845898226307823, -2.1598679279092083], [-3.7394662336015108, -1.8832138546538928], [-3.172498126659006, -4.937944809454775], [-0.7370259860228117, -0.017762651879935358], [-0.022335422145522976, 1.165666395361419], [-0.3445576225088886, 0.901886614917351], [-4.878715523932023, 5.0], [-5.0, 1.8823131887911497], [3.6809957297030276, -4.560349337145268], [4.265104024722111, -3.363311109697913], [-3.3951130126263944, -3.3173071873779736], [-4.421697374954198, 4.087696873370499], [-4.682544178616501, 3.4774623712294526], [1.290144607018575, -1.9161567101719594], [-2.805639702801412, 3.0153305042407745], [-3.4969545551858503, 1.1600213497395746], [-4.81330889547183, 3.1709116242631885], [-2.661931846276306, -0.2800527818830636], [-1.5615446962987858, 4.400020663355298], [0.18166524373700627, -2.613976586656517], [3.0461363838034714, -1.2878428291709463], [2.1730687223682352, -3.0877113325194063], [1.5765814716741218, -2.7895022094916553], [-2.297840217588068, -3.137022396823691], [1.7577435771369871, 1.5957372548045494], [-3.250262347287506, 3.981628266322061], [-1.7938526412663895, 2.6030687080499035], [3.9721478155308962, -3.960969203782872], [-2.1848893345532767, 1.687021211368902], [-0.49961922386929725, -4.012619156265195], [4.141256918608512, 2.038313784714212], [0.6997853176851706, -4.594949360187749], [1.5906232441495716, -4.286693745077206], [-1.6587416454183253, 0.37488490462840945], [-2.5773446865852083, 0.7673000163994514], [-4.281493547348879, -0.6784097624105104], [1.3000714271471336, -4.8862827792924355], [-4.551762932852439, 3.7839465056478643], [0.3741653674902719, -2.213971329732687], [1.3684878429876344, 1.5957372548045494], [3.014317667006319, -0.2834923636179691], [-1.6502398251945016, -1.8038480588329817], [4.141256918608512, 2.477011811037782], [3.0536689562915975, -2.7749298000873233], [0.2303931733112271, 2.1099886902196228], [2.53202323643381, 4.825141495726529], [-3.8942099921189754, 2.7779970808094436], [2.1388455967095785, -4.514872031073448], [-2.2022905680767577, 4.687720064377247], [-4.290454019025498, 4.396713969792819], [2.8159227083712706, -4.873610837696039], [-3.9281753317487023, -0.10362181511649587]]
                        #[-2.1849843511337519, 1.6870539388189698]
            #archive2 =  [[-2.184984287994826,  1.687053876764728]]
                        #[-2.1849843511337523, 1.6870539388189703]
                        #[-2.184984287994826,  1.687053876764728]
                        #[-2.1849843521397636, 1.687053936459804]
            #archive2 = [[-0.49969506538258013, -4.0125970848020875]]
                        #[-0.49969506538256958, -4.0125970848019490]
            #archive2 = [[-0.49969513504451857, -4.012597050969996]]
                            #[-0.49969506538256914, -4.01259708480195]
                            #[-0.49969506538256958, -4.0125970848019490]

            #archive2 = [[-0.49968798513322815, -4.0125684842053495]]
            #archive2.append([-2.184972206978734, 1.68643120056627])
            #print(f.evaluate(archive2[0]))
            #print("ANTES DE TUDO", archive2)

            #archive2 = []
            #archive2.append([-0.4998629111178183, -4.012953903305015])
            #archive2.append([-2.1962545618733444, 1.6937516005176196])
            #archive2.append([-0.49958565012365774, -4.012563892056049])
            #print(archive2)
            #sleep(100)

            if archive_flag == 0:
                for ind in best_individuals:
                    it, endpt = hooke(dim, self.pop[ind], rho, eps, itermax_archive, f)                
                    self.pop[ind] = endpt
            elif archive_flag == 1:
            #     #print("a")
            # #LOCAL SERACH ROUTINE WITH ARCHIVE
                
                #print("DEPOIS DO HOOKE JEEVES ", archive2)
                seconds_nelder_start = time()
                for ind in range(0,len(archive2)):
                    ### NELDER MEAD                    
                    it, endpt = nelder_mead(f, archive2[ind], itermax_archive)
                    archive2[ind] = it.tolist()
                    #print("Nelder Mead: ", archive2[ind] )
                seconds_nelder_end = time()

                seconds_hj_start = time()
                for ind in range(0,len(archive2)):
                    ### HOOKE JEEVES
                    it, endpt = hooke(dim, archive2[ind], rho, eps, itermax_archive, f) 
                    print(it)
                    archive2[ind] = endpt
                seconds_hj_end = time()
                #print(archive2)
                    
                
            #archive2 = []
            #archive2.append([-0.49969507852491586, -4.012597082427802])
            #archive2.append([-2.184984258552632, 1.6870538993191966])

            #archive2.append([-0.49969506538256958, -4.0125970848019490])

            #print("DEPOIS DE TUDO", archive2)
            #archive2.append([-2.1849843511337519, 1.6870539388189698])

            fpop = self.evaluatePopulation(archive2, func, f)


            records.write('Pos: %s\n\n' % str(best))
            fbest_r.append(fbest)
            best_r.append(best)
            elapTime_r.append(elapTime[max_iterations-1])
            self.generateGraphs(self.fbest_list, self.diversity, max_iterations, funcs[nfunc], r, dim, hora, self.nclusters_list)
            avr_fbest_r.append(self.fbest_list)
            avr_diversity_r.append(self.diversity)
            avr_nclusters_r.append(self.nclusters_list)

            pop_aux = []
            pop_aux = self.pop

            pop_aux3 = []
            pop_aux3 = archive2

            fpop = self.evaluatePopulation(self.pop, nfunc, f)

            #a = list(filter(lambda x: x != 0, archive))
            if archive_flag == 0:
                count, seeds = how_many_goptima(self.pop, f, accuracy, len(self.pop), pop_aux)
            elif archive_flag == 1:
                count, seeds = how_many_goptima(archive2, f, accuracy, len(archive2), pop_aux3)

            #print(seeds)
            count_global += count

            self.pop = []
            self.m_nmdf = 0.00 
            self.diversity = []
            self.fbest_list = []
            self.nclusters_list = []

            PR.append(count_global/f.get_no_goptima())
            print("Peak Ratio: %.4f" % PR[r])
            if(PR[r] == 1):
                SR += 1
        hora_final = strftime("%Hh%Mm%S", localtime())
        fbestAux = [sum(x)/len(x) for x in zip(*avr_fbest_r)]
        diversityAux = [sum(x)/len(x) for x in zip(*avr_diversity_r)]
        nclustersAux = [sum(x)/len(x) for x in zip(*avr_nclusters_r)]
        self.generateGraphs(fbestAux, diversityAux, max_iterations, funcs[nfunc], 'Overall', dim, hora, nclustersAux)
        records.write('=================================================================================================================')
        if maximize==False:
            results.write('Gbest Overall: %.4f\n' % (min(fbest_r)))
            results.write('Positions: %s\n\n' % str(best_r[fbest_r.index(min(fbest_r))]))
        else:
            results.write('Tamanho da populacao: %d\n' % pop_size)
            results.write('Iteracoes Maximas: %d\n' % max_iterations)
            results.write('Funcao Otimizada: %s\n' % func)
            results.write('Gbest Overall: %.4f\n' % (max(fbest_r)))
            results.write('Niter: %.4f\n , ' % niter_flag)
            results.write('Positions: %s\n' % str(best_r[fbest_r.index(max(fbest_r))]))
            for i in range(0, runs):
                results.write('Mean Peaks Found on Run %d: %f (%f)\n' % (i, PR[i], (PR[i]*f.get_no_goptima())))
            if runs > 1:
                results.write('Mean Peaks Found: %.4f\n' % (sum(PR)/runs))
                results.write('Peak Ratio Standard Deviation: %.4f\n' % (stdev(PR)))
                results.write('[')
                for i in range(0, runs):
                    results.write('%.5f, ' % PR[i])
                results.write(']\n')            
            results.write('Success rate: %.4f\n\n' % (SR/runs))


        results.write('Gbest Average: %.4f\n' % (sum(fbest_r)/len(fbest_r)))
        results.write('Gbest Median: %.4f #probably should use median to represent due probably non-normal distribution (see Shapiro-Wilk normality test)\n' % (median(fbest_r)))
        if runs > 1:
            results.write('Gbest Standard Deviation: %.4f\n\n' % (stdev(fbest_r)))
        tempo = (sum(elapTime_r)/len(elapTime_r))
        #tempo = float(tempo/60000.0)
        results.write('Elappsed Time Average: %.4f\n' % (tempo))
        results.write('Elappsed Time Hooke-Jeeves: %.4f\n' % (seconds_hj_end - seconds_hj_start))
        results.write('Elappsed Time Nelder-Mead: %.4f\n' % (seconds_nelder_end - seconds_nelder_start))
        results.write('Final Time recorded: %s\n' % str(hora_final))
        if runs > 1:
            results.write('Elappsed Time Standard Deviation: %.4f\n' % (stdev(elapTime_r)))

        results.write('=================================================================================================================\n')

#FUNCTIONS AVAIABLE 
    
    # 1:five_uneven_peak_trap, 2:equal_maxima, 3:uneven_decreasing_maxima, 
    #       4:himmelblau, 5:six_hump_camel_back, 6:shubert, 7:vincent, 8:shubert, 9:vincent,
    #       10:modified_rastrigin_all, 11:CF1, 12:CF2, 13:CF3, 14:CF3, 15:CF4, 16:CF3, 
    #       17:CF4, 18:CF3, 19:CF4, 20:CF4

if __name__ == '__main__': 
    from ndbjde import DE
    funcs = ["haha", five_uneven_peak_trap, equal_maxima, uneven_decreasing_maxima, himmelblau, six_hump_camel_back, shubert, vincent, shubert, vincent, modified_rastrigin_all, CF1, CF2, CF3, CF3, CF4, CF3, CF4, CF3, CF4, CF4]
    #nfunc = 1
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', action='store', type=int, help='Function to be optimized.')
    parser.add_argument('-p', action='store', type=int, help='Population size.')
    parser.add_argument('-acc', action='store', type=float, help='Accuracy of the algorithm.')
    parser.add_argument('-r', action='store', type=int, help='Number of runs.')
    parser.add_argument('-a', action='store', type=int, help='Archive flag (0 for No, 1 for Yes)')
    parser.add_argument('-flag', action='store', type=int, help='Flag to plot (0 or 1).')

    args = parser.parse_args()

    nfunc = (args.f)
    pop_size = (args.p)
    accuracy = (args.acc)
    runs = (args.r)
    flag_plot = (args.flag)
    archive_flag = (args.a)

    #print(nfunc, pop_size, accuracy)

    f = CEC2013(nfunc)
    cost_func = funcs[nfunc]             # Fitness Function
    dim = f.get_dimension()
    max_iterations = int((f.get_maxfes() // pop_size) * 0.7 )
    #ISDA RESULTS EPS_VALUE == 0.1!!!!
    eps_value = 0.1

    p = DE(pop_size)
    p.diferentialEvolution(pop_size, dim, max_iterations, runs, cost_func, f, nfunc, accuracy, flag_plot, eps_value, archive_flag, maximize=True)


