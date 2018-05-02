#imports
from os import mkdir
import math
import numpy
from statistics import median, stdev
from matplotlib import pyplot as plt
from time import gmtime, strftime, time
from random import uniform, choice, randint, gauss, sample
from cec2013 import *
from scipy.spatial import distance
import uuid


class DE:

    def __init__(self):
        self.pop = [] #population's positions        
        self.m_nmdf = 0.00 #diversity variable
        self.diversity = []
        self.fbest_list = []
        self.ns1 = 0
        self.ns2 = 0
        self.nf1 = 0
        self.nf2 = 0

    def generateGraphs(self, fbest_list, diversity_list, max_iterations, uid, run):
        plt.plot(range(0, max_iterations), fbest_list, 'r--')
        plt.savefig(str(uid) + '/graphs/run' + str(run) + '_' + 'convergence.png')
        plt.clf()                                                   
        plt.plot(range(0, max_iterations), diversity_list, 'b--')
        plt.savefig(str(uid) + '/graphs/run' + str(run) + '_' + 'diversity.png')
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

        for ind in range(pop_size):
            lp = []
            for d in range(dim):
                lp.append(uniform(lb[d],ub[d]))
            self.pop.append(lp)

    def evaluatePopulation(self, func):
        fpop = []
        for ind in self.pop:
            fpop.append(func(ind)) 
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

    def rand_1_bin(self, ind, dim, wf, cr):
        vec_aux = sample(self.pop, len(self.pop))

        p1 = vec_aux[0]
        p2 = vec_aux[1]
        p3 = vec_aux[3]

        #p1 = ind
        #while(p1 == ind):
        #    p1 = choice(self.pop)
        #p2 = ind
        #while(p2 == ind or p2 == p1):
        #    p2 = choice(self.pop)
        #p3 = ind
        #while(p3 == ind or p3 == p1 or p3 == p2):
        #    p3 = choice(self.pop)
        #print(p1, p2, p3)
        cutpoint = randint(0, dim-1)
        candidateSol = []
        
        for i in range(dim):
            if(i == cutpoint or uniform(0,1) < cr):
                candidateSol.append(p3[i]+wf*(p1[i]-p2[i])) # -> rand(p3) , vetor diferença (wf*(p1[i]-p2[i]))i
            else:
                candidateSol.append(ind[i])

        return candidateSol
    
    def currentToBest_2_bin(self, ind, best, dim, wf, cr):
        vec_aux = sample(self.pop, len(self.pop))

        p1 = vec_aux[0]
        p2 = vec_aux[1]
        #p3 = vec_aux[3]

        #p1 = ind
        #while(p1 == ind):
        #    p1 = choice(self.pop)
        #p2 = ind
        #while(p2 == ind or p2 == p1):
        #    p2 = choice(self.pop)
        #print(p1, p2)
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
            if ind[d] < lb[k]:
                ind[d] = lb[k] 
            if ind[d] > ub[k]:
                ind[d] = ub[k] 

    def euclidean_distance(self, alvo, j):
        dist = []
        for i in range(len(self.pop)):
            if(j == i):
                pass
            else:
                #dist.append(numpy.linalg.norm(self.pop[i] - alvo))
                dist.append(distance.euclidean(self.pop[i], alvo))
        return dist.index(min(dist))

    def diferentialEvolution(self, pop_size, dim, max_iterations, runs, func, f, maximize=True, p1=0.5, p2=0.5, learningPeriod=50, crPeriod=5, crmUpdatePeriod=25):
        count_global = 0.0
        crowding_target = 0
        PR = 0.0 #PEAK RATIO
        #generate execution identifier
        uid = uuid.uuid4()
        mkdir(str(uid))
        mkdir(str(uid) + '/graphs')
        #to record the results
        results = open(str(uid) + '/results.txt', 'a')
        records = open(str(uid) + '/records.txt', 'a')
        results.write('ID: %s\tDate: %s\tRuns: %s\n' % (str(uid ), strftime("%Y-%m-%d %H:%M:%S", gmtime()), str(runs)))
        results.write('=================================================================================================================\n')
        records.write('ID: %s\tDate: %s\tRuns: %s\n' % (str(uid ), strftime("%Y-%m-%d %H:%M:%S", gmtime()), str(runs)))
        records.write('=================================================================================================================\n')
        avr_fbest_r = []
        avr_diversity_r = []
        fbest_r = []
        best_r = []
        elapTime_r = []
        
        #runs
        for r in range(runs):
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
            fpop = self.evaluatePopulation(func)
            #print(self.pop)
            #print(fpop)
            fbest,best = self.getBestSolution(maximize, fpop)
            #evolution_step
            # generates crossover rate values
            crm = 0.5
            crossover_rate = [gauss(crm, 0.1) for i in range(pop_size)]
            cr_list = []
            for iteration in range(max_iterations):
                #print(iteration)
                #print(self.pop)
                avrFit = 0.00
                # #update_solutions
                strategy = 0
                for ind in range(0,len(self.pop)):
                    # generate weight factor values
                    weight_factor = gauss(0.5, 0.3)
                    if uniform(0,1) < p1:
                        #print("IND A MUTAR:", self.pop[ind])
                        candSol = self.rand_1_bin(self.pop[ind], dim, weight_factor, crossover_rate[ind])
                        strategy = 1
                    else:
                        candSol = self.currentToBest_2_bin(self.pop[ind], best, dim, weight_factor, crossover_rate[ind])
                        strategy = 2
                    
                    self.boundsRes(candSol, f, dim)

                    fcandSol = func(candSol)

                    crowding_target = self.euclidean_distance(candSol, ind)

                    if maximize == False:
                        if fcandSol <= fpop[ind]:
                            self.pop[ind] = candSol
                            fpop[ind] = fcandSol
                            cr_list.append(crossover_rate[ind])
                            if strategy == 1:
                                self.ns1+=1
                            elif strategy == 2:
                                self.ns2+=1
                        else:
                            if strategy == 1:
                                self.nf1+=1
                            elif strategy == 2:
                                self.nf2+=1
                    else:
                        if fcandSol >= fpop[crowding_target]:
                            self.pop[crowding_target] = candSol
                            fpop[crowding_target] = fcandSol
                            cr_list.append(crossover_rate[crowding_target])
                            if strategy == 1:
                                self.ns1+=1
                            elif strategy == 2:
                                self.ns2+=1
                        else:
                            if strategy == 1:
                                self.nf1+=1
                            elif strategy == 2:
                                self.nf2+=1
 
                    avrFit += fpop[crowding_target]
                avrFit = avrFit/pop_size
                self.diversity.append(self.updateDiversity())

                fbest,best = self.getBestSolution(maximize, fpop)
                
                self.fbest_list.append(fbest)
                elapTime.append((time() - start)*1000.0)
                records.write('%i\t%.4f\t%.4f\t%.4f\t%.4f\n' % (iteration, round(fbest,4), round(avrFit,4), round(self.diversity[iteration],4), elapTime[iteration]))
                
                if iteration%crPeriod == 0 and iteration!=0:
                    crossover_rate = [gauss(crm, 0.1) for i in range(pop_size)]
                    if iteration%crmUpdatePeriod == 0:
                        crm = sum(cr_list)/len(cr_list)
                        cr_list = []

                if iteration%learningPeriod == 0 and iteration!=0: 
                    p1 = (self.ns1*(self.ns2+self.nf2))/(self.ns2*(self.ns1+self.nf1)+self.ns1*(self.ns2+self.nf2))
                    p2 = 1-p1
                    self.nf2 = 0
                    self.ns1 = 0
                    self.ns2 = 0
                    self.nf1 = 0

            records.write('Pos: %s\n\n' % str(best))
            fbest_r.append(fbest)
            best_r.append(best)
            elapTime_r.append(elapTime[max_iterations-1])
            self.generateGraphs(self.fbest_list, self.diversity, max_iterations, uid, r)
            avr_fbest_r.append(self.fbest_list)
            avr_diversity_r.append(self.diversity)

            pop_aux = []
            #for i in range(pop_size):
            #    for j in range(dim):
            #        pop_aux.append(self.pop[i][j])
            pop_aux = self.pop
            #print("POP_AUX: ", pop_aux)
            count, seeds = how_many_goptima(self.pop, f, 0.001, len(self.pop), pop_aux)

            #print(count, seeds)
            count_global += count

            self.pop = []
            self.m_nmdf = 0.00 
            self.diversity = []
            self.fbest_list = []
            p1 = p2 = 0.5
            self.nf2 = 0
            self.ns1 = 0
            self.ns2 = 0
            self.nf1 = 0

        PR = (count_global/runs)/f.get_no_goptima()
        #print("Media de picos encontrados = ", PR)
        

        fbestAux = [sum(x)/len(x) for x in zip(*avr_fbest_r)]
        diversityAux = [sum(x)/len(x) for x in zip(*avr_diversity_r)]
        self.generateGraphs(fbestAux, diversityAux, max_iterations, uid, 'Overall')
        records.write('=================================================================================================================')
        if maximize==False:
            results.write('Gbest Overall: %.4f\n' % (min(fbest_r)))
            results.write('Positions: %s\n\n' % str(best_r[fbest_r.index(min(fbest_r))]))
        else:
            results.write('Gbest Overall: %.4f\n' % (max(fbest_r)))
            results.write('Positions: %s\n' % str(best_r[fbest_r.index(max(fbest_r))]))
            results.write('Mean Peaks Found: %f\n\n' % PR)

        results.write('Gbest Average: %.4f\n' % (sum(fbest_r)/len(fbest_r)))
        results.write('Gbest Median: %.4f #probably should use median to represent due probably non-normal distribution (see Shapiro-Wilk normality test)\n' % (median(fbest_r)))
        if runs > 1:
            results.write('Gbest Standard Deviation: %.4f\n\n' % (stdev(fbest_r)))
        results.write('Elappsed Time Average: %.4f\n' % (sum(elapTime_r)/len(elapTime_r)))
        if runs > 1:
            results.write('Elappsed Time Standard Deviation: %.4f\n' % (stdev(elapTime_r)))
        results.write('=================================================================================================================\n')

if __name__ == '__main__': 
    from ncsade import DE
    nfunc = 4
    f = CEC2013(nfunc)
    cost_func = himmelblau             # Fitness Function
    dim = f.get_dimension()
    max_iterations = 200 
    pop_size = 20
    runs = 10
    p = DE()
    p.diferentialEvolution(pop_size, dim, max_iterations, runs, cost_func, f, maximize=True)

