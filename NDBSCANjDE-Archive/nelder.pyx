import copy
import numpy as np



def nelder_mead(f, x_start, int max_iter,
                float step, float no_improve_thr=10e-15,
                int no_improv_break=500, 
                float alpha=1.0, float gamma=2.75, float rho=-0.5, float sigma=0.5):
    '''
        @param f (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
        @param x_start (numpy array): initial position
        @param step (float): look-around radius in initial step
        @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
        @max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        @alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)
        return: tuple (best parameter array, best score)
    '''

    x_start = np.asarray(x_start)
    # init
    cdef int dim
    dim = len(x_start)
    cdef float prev_best
    prev_best = f.evaluate(list(x_start))
    cdef int no_improv
    no_improv = 0
    res = [[x_start, prev_best]]
    #print('res1: ', res)
    cdef int i
    cdef float score
    for i in range(dim):
        x = copy.copy(x_start)
        if(x[i] + step > f.get_ubound(i)):
            x[i]
        else:
            x[i] = x[i] + step
        score = f.evaluate(list(x))
        res.append([x, score])
    #print('res2: ', res)

    # simplex iter
    cdef int iters = 0
    cdef float rscore, escore, cscore

    while 1:
        # order
        res.sort(key=lambda x: x[1], reverse=True)
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            #print(iters)
            return res[0]
        iters += 1

        # break after no_improv_break iterations with no improvement
        #print('...best so far:', best, prev_best, (prev_best - no_improve_thr), no_improv, iters)

        if -best > -(prev_best + no_improve_thr):
            no_improv += 1            
        else:
            no_improv = 0
            prev_best = best

        if no_improv >= no_improv_break:
            return res[0]

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        for i in range(dim):
            if(xr[i] > f.get_ubound(i)):
                xr[i] = f.get_ubound(i)
            elif(xr[i] < f.get_lbound(i)):
                xr[i] = f.get_lbound(i)
        
        rscore = f.evaluate(list(xr))
        
        if res[0][1] >= rscore > res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore > res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            for i in range(dim):
                if(xe[i] > f.get_ubound(i)):
                    xe[i] = f.get_ubound(i)
                elif(xe[i] < f.get_lbound(i)):
                    xe[i] = f.get_lbound(i)
            escore = f.evaluate(list(xe))
            
            if escore > rscore:                
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f.evaluate(list(xc))
        if cscore > res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = f.evaluate(list(redx))
            nres.append([redx, score])
        res = nres