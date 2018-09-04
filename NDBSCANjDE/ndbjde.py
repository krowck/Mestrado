#imports
from os import mkdir
import math
import numpy
import copy
import sobol_seq
from statistics import median, stdev
from matplotlib import pyplot as plt
from time import gmtime, strftime, localtime, time, sleep
from random import uniform, choice, randint, gauss, sample
from cec2013 import *
from scipy.spatial import distance
from collections import Counter
from eucl_dist.cpu_dist import dist
from eucl_dist.gpu_dist import dist as gdist
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


import uuid
#import cProfile


class DE:

    def __init__(self):
        self.pop = [] #population's positions
        self.pop_aux2 = []       
        self.m_nmdf = 0.00 #diversity variable
        self.diversity = []
        self.fbest_list = []
        self.full_euclidean = []   
        self.full_euclidean_aux = []
        self.ns1 = 1
        self.ns2 = 1
        self.nf1 = 1
        self.nf2 = 1

    def generateGraphs(self, fbest_list, diversity_list, max_iterations, uid, run, dim, hora):
        plt.plot(range(0, max_iterations), fbest_list, 'r--')
        plt.savefig(str(uid) + '_' + str(dim) + 'D_' + str(hora) + '/graphs/run' + str(run) + '_' + 'convergence.png')
        plt.clf()                                                   
        plt.plot(range(0, max_iterations), diversity_list, 'b--')
        #plt.ylim(ymin=0)
        plt.savefig(str(uid) + '_' + str(dim) + 'D_' + str(hora) + '/graphs/run' + str(run) + '_' + 'diversity.png')
        plt.clf()
        plt.plot(range(0, max_iterations), diversity_list, 'b--')
        plt.ylim(ymin=0)
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

        # vec = sobol_seq.i4_sobol_generate(dim, pop_size)
        

        # for i in range(pop_size):
        # 	lp = []
        # 	for d in range(dim):
        # 		print(vec[i][d])
        # 		lp.append(lb[d] + vec[i][d]*(ub[d] -  lb[d]))
        # 	self.pop.append(lp)

        
        for ind in range(pop_size):
            lp = []
            for d in range(dim):
                lp.append(uniform(lb[d],ub[d]))
            self.pop.append(lp)
        #print(self.pop)
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

    def evaluatePopulation(self, func, f):
        fpop = []
        for ind in self.pop:
            fpop.append(f.evaluate(ind))
        return fpop 

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

    def rand_1_bin(self, ind, alvo, dim, wf, cr, neighborhood_list, m):
        vec_candidates = []

        vec_aux = sample(neighborhood_list, m)

        p1 = self.pop[vec_aux[0]]
        p2 = self.pop[vec_aux[1]]
        p3 = self.pop[vec_aux[2]]

        cutpoint = randint(0, dim-1)
        candidateSol = []
        
        for i in range(dim):
            if(i == cutpoint or uniform(0,1) < cr):
                candidateSol.append(p3[i]+wf*(p1[i]-p2[i])) # -> rand(p3) , vetor diferença (wf*(p1[i]-p2[i]))i
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
            candidateSol.append(ind[i] + uniform(0,1)*(p1[i] - ind[i]) + wf*(p2[i] - p3[i]))

        # for i in range(dim):
        #     if(i == cutpoint or uniform(0,1) < cr):
        #         candidateSol.append(ind[i]+wf*(best[i]-ind[i])+wf*(p1[i]-p2[i])) # -> rand(p3) , vetor diferença (wf*(p1[i]-p2[i]))
        #     else:
        #         candidateSol.append(ind[i])

        return candidateSol

    def boundsRes(self, ind, f, dim):
        ub = [0] * dim
        lb = [0] * dim
        for k in range(dim):
            ub[k] = f.get_ubound(k)
            lb[k] = f.get_lbound(k)

        for d in range(len(ind)):
            if ind[d] < lb[k]:
                ind[d] = lb[k] 
            if ind[d] > ub[k]:
                ind[d] = ub[k] 

    # def euclidean_distance_full(self, alvo, k, dim):
    #     #print(self.pop)
    #     #sleep(5)
    #     s = 0
    #     dist = []
    #     for i in range(len(self.pop)):
    #         s = 0
    #         if k == i:
    #             dist.append(math.inf)
    #         else:
    #             for j in range(dim):
    #                 diff = self.pop[i][j] - alvo[j]
    #                 s += np.linalg.norm(diff)
    #             dist.append(s)
    #     self.full_euclidean.append(dist)
        #return dist

    def euclidean_distance_full2(self, dim):
        dist1 = np.zeros((len(self.pop), dim))
        self.pop = np.asarray(self.pop) #necessario para utilizar a funcao eucl_dist -- otimizacao da distancia euclidiana
        #print(self.pop)
        dist1 = dist(self.pop, self.pop)
        #dist1 = dist(self.pop, self.pop)
        #print(dist1)
        self.pop = self.pop.tolist() #necessario voltar para lista para nao afetar a programacao feita anteriormente
        dist1 = dist1.tolist()
        #print(dist1)
        self.full_euclidean.append(dist1)

    def euclidean_distance(self, alvo, k, dim, f):
        s = 0
        dist = []

        for i in range(len(self.pop)):
            s = 0
            if k == i:
                dist.append(math.inf)
            else:
                for j in range(dim):
                    diff = self.pop[i][j] - alvo[j]
                    s += np.linalg.norm(diff)
                # if(s < 0.00000001):
                #     print("Individuals too close! Generating new one.")
                #     self.generateIndividual(i, dim, f)
                #     i = i-1
                dist.append(s)
        return dist, dist.index(min(dist))

    def euclidean_distance2(self, alvo, k, dim):
        dist1 = np.zeros((len(self.pop), dim))
        alvo = np.asarray([alvo])
        self.pop = np.asarray(self.pop) #necessario para utilizar a funcao eucl_dist -- otimizacao da distancia euclidiana
        #print(self.pop, alvo)
        dist1 = dist(alvo, self.pop)
        #dist1 = dist(alvo, self.pop)
        #print(dist1)
        #print(np.argmin(dist1))
        self.pop = self.pop.tolist() #necessario voltar para lista para nao afetar a programacao feita anteriormente
        dist1 = dist1.tolist()
        dist1 = dist1.pop()
        #print(dist1)
        #print(min(dist1))
        #print(dist1.index(min(dist1)))
        #sleep(10)
        return dist1, dist1.index(min(dist1))
        #self.full_euclidean.append(dist1)

    def generate_neighborhood(self, ind, m, dim, f):
        vec_dist = []
        vec_dist = list(self.full_euclidean[ind])
        neighborhood_list = [-1] * m
        #print("VEC-DIST ANTES:", vec_dist)
        #print(ind, m, dim)

        #vec_dist = self.euclidean_distance_vec(self.pop[ind], ind, dim, f)
        for k in range(m):
            neighborhood_list[k] = vec_dist.index(min(vec_dist))            
            vec_dist[vec_dist.index(min(vec_dist))] = math.inf
        #print("vec_aux>>>>>>>", vec_aux)
        #print("VEC_DIST -->", vec_dist)
        #print(self.full_euclidean)
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
                dist, alvo = self.euclidean_distance2(self.pop[x], x, dim)
                self.full_euclidean[x] = dist
        

    def diferentialEvolution(self, pop_size, dim, max_iterations, runs, func, f, nfunc, accuracy, maximize=True):

        crowding_target = 0
        neighborhood_list = []
        funcs = ["haha", "five_uneven_peak_trap", "equal_maxima", "uneven_decreasing_maxima", "himmelblau", "six_hump_camel_back", "shubert", "vincent", "shubert", "vincent", "modified_rastrigin_all", "CF1", "CF2", "CF3", "CF3", "CF4", "CF3", "CF4", "CF3", "CF4", "CF4"]
        #print(">>>>>>>>>>", str(funcs[1]))
        m = 0
        PR = [] #PEAK RATIO
        SR = 0.0
        #generate execution identifier
        #uid = uuid.uuid4()
        hora = strftime("%Hh%Mm%S", localtime())
        mkdir(str(funcs[nfunc]) + '_' + str(dim) + 'D_' + str(hora))
        mkdir(str(funcs[nfunc]) + '_' + str(dim) + 'D_' + str(hora) +'/graphs')
        #to record the results
        results = open(str(funcs[nfunc]) + '_' + str(dim) + 'D_' + str(hora) + '/results.txt', 'a')
        records = open(str(funcs[nfunc]) + '_' + str(dim) + 'D_' + str(hora) + '/records.txt', 'a')
        results.write('ID: %s\tDate: %s\tRuns: %s\n' % (str(funcs[nfunc] ), strftime("%Y-%m-%d %H:%M:%S", gmtime()), str(runs)))
        results.write('=================================================================================================================\n')
        records.write('ID: %s\tDate: %s\tRuns: %s\n' % (str(funcs[nfunc] ), strftime("%Y-%m-%d %H:%M:%S", gmtime()), str(runs)))
        records.write('=================================================================================================================\n')
        avr_fbest_r = []
        avr_diversity_r = []
        fbest_r = []
        best_r = []
        elapTime_r = []
        
        #runs
        for r in range(runs):
            count_global = 0.0
            elapTime = []
            start = time()
            records.write('Run: %i\n' % r)
            records.write('Iter\tGbest\tAvrFit\tDiver\tETime\t\n')
            
            #start the algorithm
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
            fpop = self.evaluatePopulation(func, f)
            self.pop_aux2 = self.pop

            
            # X = StandardScaler(with_mean=False).fit_transform(self.pop)

            # db = DBSCAN(eps=0.01, min_samples=5).fit(X)
            # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            # core_samples_mask[db.core_sample_indices_] = True
            # labels = db.labels_

            # # Number of clusters in labels, ignoring noise if present.
            # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

            # print('Estimated number of clusters: %d' % n_clusters_)

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

            #for ind in range(pop_size):                
            #    self.euclidean_distance_full(self.pop[ind], ind, dim)
            #print(self.full_euclidean)
            #self.full_euclidean = []
            self.euclidean_distance_full2(dim)
            self.full_euclidean = self.full_euclidean.pop()
            for control in range(pop_size):
                self.full_euclidean[control][control] = math.inf
            #print(self.full_euclidean)
            #sleep(10)
            self.full_euclidean_aux = self.full_euclidean


            fbest,best = self.getBestSolution(maximize, fpop)
            #evolution_step
            # generates crossover rate values
            crm = 0.5
            Fl = 0.1
            Fu = 0.9
            tau1 = tau2 = 0.1
            crossover_rate = [gauss(crm, 0.1) for i in range(pop_size)]
            mutation_rate = [0.5] * pop_size
            cr_list = []
            for iteration in range(max_iterations):
                print(iteration)
                if pop_size <= 200:
                    m=math.floor(5+20*((max_iterations-iteration)/max_iterations))
                else:
                    m=math.floor(5+20*((max_iterations-iteration)/max_iterations))
                avrFit = 0.00 
                # #update_solutions
                strategy = 0
                #print(mutation_rate)
                #print(crossover_rate)
                #sleep(5)
                for ind in range(0,len(self.pop)):

                    rand1 = uniform(0, 1)
                    rand2 = uniform(0, 1)
                    rand3 = uniform(0, 1)
                    rand4 = uniform(0, 1)

                    if rand2 < tau1:
                        mutation_rate[ind] = Fl + (rand1 * Fu)
                    

                    if rand4 < tau2:
                        crossover_rate[ind] = rand3
                    
                    # generate weight factor values
                    weight_factor = gauss(0.5, 0.3)
                    #weight_factor = 0.5
                    #crossover_rate[ind] = 0.1
                    if uniform(0,1) < 1:
                        neighborhood_list = self.generate_neighborhood(ind, m, dim, f)
                        candSol = self.rand_1_bin(self.pop[ind], ind, dim, mutation_rate[ind], crossover_rate[ind], neighborhood_list, m)
                        #candSol = self.currentToRand_1_bin(self.pop[ind], ind, dim, mutation_rate[ind], crossover_rate[ind], neighborhood_list, m)
                        strategy = 1
                    else:
                        neighborhood_list = self.generate_neighborhood(ind, m, dim)
                        candSol = self.currentToBest_2_bin(self.pop[ind], ind, best, dim, mutation_rate[ind], crossover_rate[ind], neighborhood_list, m)
                        strategy = 2
                    
                    self.boundsRes(candSol, f, dim)

                    fcandSol = f.evaluate(candSol)

                    dist, crowding_target = self.euclidean_distance2(candSol, ind, dim)

                    if maximize == True:
                        if fcandSol >= fpop[crowding_target]:
                            #self.pop_aux2[crowding_target] = candSol
                            self.pop[crowding_target] = candSol
                            dist_correta, aux = self.euclidean_distance2(candSol, crowding_target, dim)
                            #self.full_euclidean_aux[crowding_target] = dist_correta
                            self.full_euclidean[crowding_target] = dist_correta
                            fpop[crowding_target] = fcandSol
                            
 
                    avrFit += fpop[crowding_target]

                avrFit = avrFit/pop_size
                self.diversity.append(0)
                #self.pop = self.pop_aux2
                #self.full_euclidean = self.full_euclidean_aux

                fbest,best = self.getBestSolution(maximize, fpop)
                
                self.fbest_list.append(fbest)
                elapTime.append((time() - start)/60.0)
                records.write('%i\t%.4f\t%.4f\t%.4f\t%.4f\n' % (iteration, round(fbest,4), round(avrFit,4), round(self.diversity[iteration],4), elapTime[iteration]))

                if iteration%50 == 0:
                    X = StandardScaler(with_mean=False).fit_transform(self.pop)

                    db = DBSCAN(eps=0.1, min_samples=m).fit(X)
                    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                    core_samples_mask[db.core_sample_indices_] = True
                    labels = db.labels_

                    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

                    if n_clusters_ > 0:
                        self.reset_pop(labels, Counter(labels), n_clusters_, m, dim, f)


                    print(Counter(labels))
            

            X = StandardScaler(with_mean=False).fit_transform(self.pop)

            db = DBSCAN(eps=0.1, min_samples=5).fit(X)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

            print('Estimated number of clusters: %d' % n_clusters_)

            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each)
                      for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = (labels == k)

                xy = X[class_member_mask & core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=14)

                xy = X[class_member_mask & ~core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=6)

            plt.title('Estimated number of clusters: %d' % n_clusters_)
            plt.show()

            records.write('Pos: %s\n\n' % str(best))
            fbest_r.append(fbest)
            best_r.append(best)
            elapTime_r.append(elapTime[max_iterations-1])
            self.generateGraphs(self.fbest_list, self.diversity, max_iterations, funcs[nfunc], r, dim, hora)
            avr_fbest_r.append(self.fbest_list)
            avr_diversity_r.append(self.diversity)

            pop_aux = []
            pop_aux = self.pop
            count, seeds = how_many_goptima(self.pop, f, accuracy, len(self.pop), pop_aux)

            #print(count, seeds)
            count_global += count

            self.pop = []
            self.m_nmdf = 0.00 
            self.diversity = []
            self.fbest_list = []

            print("ENTROu")
            PR.append(count_global/f.get_no_goptima())
            print(PR[r])
            if(PR[r] == 1):
                SR += 1

        
        fbestAux = [sum(x)/len(x) for x in zip(*avr_fbest_r)]
        diversityAux = [sum(x)/len(x) for x in zip(*avr_diversity_r)]
        self.generateGraphs(fbestAux, diversityAux, max_iterations, funcs[nfunc], 'Overall', dim, hora)
        records.write('=================================================================================================================')
        if maximize==False:
            results.write('Gbest Overall: %.4f\n' % (min(fbest_r)))
            results.write('Positions: %s\n\n' % str(best_r[fbest_r.index(min(fbest_r))]))
        else:
            results.write('Tamanho da populacao: %d\n' % pop_size)
            results.write('Iteracoes Maximas: %d\n' % max_iterations)
            results.write('Funcao Otimizada: %s\n' % func)
            results.write('Gbest Overall: %.4f\n' % (max(fbest_r)))
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
    nfunc = 8
    f = CEC2013(nfunc)
    cost_func = funcs[nfunc]             # Fitness Function
    dim = f.get_dimension()
    pop_size = 200
    accuracy = 0.001
    max_iterations = (f.get_maxfes() // pop_size) 
    #m = 10
    runs = 1
    p = DE()
    p.diferentialEvolution(pop_size, dim, max_iterations, runs, cost_func, f, nfunc, accuracy, maximize=True)
