import numpy as np

def best_nearby ( delta, point, prevbest, nvars, f, funevals, itermax ):

  z = point.copy ( )
  minf = prevbest
  for i in range ( 0, nvars ):

    z[i] = point[i] + delta[i]
    if(z[i] > f.get_ubound(i)):
      z[i] = f.get_ubound(i)
    elif(z[i] < f.get_lbound(i)):
      z[i] = f.get_lbound(i)

    ftmp = f.evaluate( z )

    funevals = funevals + 1

    if ( ftmp > minf ):
      minf = ftmp

    else:

      delta[i] = - delta[i]
      z[i] = point[i] + delta[i]
      if(z[i] > f.get_ubound(i)):
        z[i] = f.get_ubound(i)
      elif(z[i] < f.get_lbound(i)):
        z[i] = f.get_lbound(i)

      ftmp = f.evaluate( z )

      funevals = funevals + 1


      if ( ftmp > minf ):
        minf = ftmp
      else:
        z[i] = point[i]


  point = z.copy ( )
  newbest = minf

  # if funevals >= itermax:
  #   return newbest, point, funevals

  return newbest, point, funevals

def hooke (nvars, startpt, rho, eps, itermax, f):

  verbose = False
  newx = startpt.copy ( )
  xbefore = startpt.copy ( )
  
  delta = np.zeros ( nvars )

  for i in range ( 0, nvars ):
    if ( startpt[i] == 0.0 ):
      delta[i] = rho
    else:
      delta[i] = rho * abs ( startpt[i] )

  funevals = 0
  steplength = rho
  iters = 0
  fbefore = f.evaluate( newx )
  funevals = funevals + 1
  newf = fbefore

  while ( iters < itermax and eps < steplength ):
    iters = iters + 1

    if ( verbose ):

      print ( '' )
      print ( '  FUNEVALS = %d, F(X) = %g' % ( funevals, fbefore ) )
      for i in range ( 0, nvars ):
        print ( '  %8d  %g' % ( i, xbefore[i] ) )
#
#  Find best new point, one coordinate at a time.
#
    for i in range ( 0, nvars ):
      newx[i] = xbefore[i]

    newf, newx, funevals = best_nearby ( delta, newx, fbefore, nvars, f, funevals, itermax )
#
#  If we made some improvements, pursue that direction.
#
    keep = True

    while ( newf > fbefore and keep ):

      for i in range ( 0, nvars ):
#
#  Arrange the sign of DELTA.
#
        if ( newx[i] >= xbefore[i] ):
          delta[i] = - abs ( delta[i] )
        else:
          delta[i] = abs ( delta[i] )
#
#  Now, move further in this direction.
#
        tmp = xbefore[i]
        xbefore[i] = newx[i]
        newx[i] = newx[i] + newx[i] - tmp
        if newx[i] < f.get_lbound(i):
            newx[i] = f.get_lbound(i)
        elif newx[i] > f.get_ubound(i):
            newx[i] = f.get_ubound(i)

      fbefore = newf
      newf, newx, funevals = best_nearby ( delta, newx, fbefore, nvars, f, funevals, itermax )
#
#  If the further (optimistic) move was bad...
#
      if ( fbefore >= newf ):
        break
#
#  Make sure that the differences between the new and the old points
#  are due to actual displacements; beware of roundoff errors that
#  might cause NEWF < FBEFORE.
#
      keep = False

      for i in range ( 0, nvars ):
        if ( 0.5 * abs ( delta[i] ) < abs ( newx[i] - xbefore[i] ) ):
          keep = True
          break
      if funevals >= itermax:
        break

    if ( eps <= steplength and fbefore >= newf ):
      steplength = steplength * rho
      for i in range ( 0, nvars ):
        delta[i] = delta[i] * rho
    

  endpt = xbefore.copy ( )

  return iters, endpt


  #! /usr/bin/env python
#


def nelmin ( fn, n, start, reqmin, step, konvge, kcount ):

#*****************************************************************************80
#
## NELMIN minimizes a function using the Nelder-Mead algorithm.
#
#  Discussion:
#
#    This routine seeks the minimum value of a user-specified function.
#
#    Simplex function minimisation procedure due to Nelder+Mead(1965),
#    as implemented by O'Neill(1971, Appl.Statist. 20, 338-45), with
#    subsequent comments by Chambers+Ertel(1974, 23, 250-1), Benyon(1976,
#    25, 97) and Hill(1978, 27, 380-2)
#
#    The function to be minimized must have the form:
#
#      function value = fn ( x )
#
#    where X is a vector, and VALUE is the (scalar) function value.
#    The name of this function must be passed as the argument FN.
#
#    This routine does not include a termination test using the
#    fitting of a quadratic surface.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    04 October 2015
#
#  Author:
#
#    Original FORTRAN77 version by R ONeill.
#    Python version by John Burkardt.
#
#  Reference:
#
#    John Nelder, Roger Mead,
#    A simplex method for function minimization,
#    Computer Journal,
#    Volume 7, 1965, pages 308-313.
#
#    R ONeill,
#    Algorithm AS 47:
#    Function Minimization Using a Simplex Procedure,
#    Applied Statistics,
#    Volume 20, Number 3, 1971, pages 338-345.
#
#  Parameters:
#
#    Input, real value = FN ( X ), the name of the MATLAB function which 
#    evaluates the function to be minimized, preceded by an "@" sign.
#
#    Input, integer N, the number of variables.
#
#    Input, real START(N).  On input, a starting point
#    for the iteration.  On output, this data may have been overwritten.
#
#    Input, real REQMIN, the terminating limit for the variance
#    of function values.
#
#    Input, real STEP(N), determines the size and shape of the
#    initial simplex.  The relative magnitudes of its elements should reflect
#    the units of the variables.
#
#    Input, integer KONVGE, the convergence check is carried out
#    every KONVGE iterations.
#
#    Input, integer KCOUNT, the maximum number of function
#    evaluations.
#
#    Output, real XMIN(N), the coordinates of the point which
#    is estimated to minimize the function.
#
#    Output, real YNEWLO, the minimum value of the function.
#
#    Output, integer ICOUNT, the number of function evaluations.
#
#    Output, integer NUMRES, the number of restarts.
#
#    Output, integer IFAULT, error indicator.
#    0, no errors detected.
#    1, REQMIN, N, or KONVGE has an illegal value.
#    2, iteration terminated because KCOUNT was exceeded without convergence.
#

  xmin = np.zeros ( n )
  ynewlo = 0.0
  icount = 0
  numres = 0
  ifault = 0

  ccoeff = 0.5
  ecoeff = 2.0
  eps = 0.001
  rcoeff = 1.0
#
#  Check the input parameters.
#
  if ( reqmin <= 0.0 ):
    ifault = 1
    return xmin, ynewlo, icount, numres, ifault

  if ( n < 1 ):
    ifault = 1
    return xmin, ynewlo, icount, numres, ifault

  if ( konvge < 1 ):
    ifault = 1
    return xmin, ynewlo, icount, numres, ifault
#
#  Initialization.
#
  jcount = konvge
  delta = 1.0
  rq = reqmin * float ( n )

  p = np.zeros ( [ n, n + 1 ] )
  p2star = np.zeros ( n )
  pbar = np.zeros ( n )
  pstar = np.zeros ( n )
  y = np.zeros ( n + 1 )
#
#  Initial or restarted loop.
#
  while ( True ):

    for i in range ( 0, n ):
      p[i,n] = start[i]

    y[n] = fn ( start )
    icount = icount + 1
#
#  Define the initial simplex.
#
    for j in range ( 0, n ):
      x = start[j]
      start[j] = start[j] + step[j] * delta
      for i in range ( 0, n ):
        p[i,j] = start[i]
      y[j] = fn ( start )
      icount = icount + 1
      start[j] = x
#
#  The simplex construction is complete.
#
#  Find highest and lowest Y values.  YNEWLO = Y(IHI) indicates
#  the vertex of the simplex to be replaced.
#
    ylo = y[0]
    ilo = 0

    for i in range ( 1, n + 1 ):
      if ( y[i] < ylo ):
        ylo = y[i]
        ilo = i
#
#  Inner loop.
#
    while ( icount < kcount ):

      ynewlo = y[0]
      ihi = 0

      for i in range ( 1, n + 1 ):
        if ( ynewlo < y[i] ):
          ynewlo = y[i]
          ihi = i
#
#  Calculate PBAR, the centroid of the simplex vertices
#  excepting the vertex with Y value YNEWLO.
#
      for i in range ( 0, n ):
        z = 0.0
        for j in range ( 0, n + 1 ):
          z = z + p[i,j]
        z = z - p[i,ihi]
        pbar[i] = z / float ( n )
#
#  Reflection through the centroid.
#
      for i in range ( 0, n ):
        pstar[i] = pbar[i] + rcoeff * ( pbar[i] - p[i,ihi] )

      ystar = fn ( pstar )
      icount = icount + 1
#
#  Successful reflection, so extension.
#
      if ( ystar < ylo ):

        for i in range ( 0, n ):
          p2star[i] = pbar[i] + ecoeff * ( pstar[i] - pbar[i] )

        y2star = fn ( p2star )
        icount = icount + 1
#
#  Check extension.
#
        if ( ystar < y2star ):

          for i in range ( 0, n ):
            p[i,ihi] = pstar[i]

          y[ihi] = ystar
#
#  Retain extension or contraction.
#
        else:

          for i in range ( 0, n ):
            p[i,ihi] = p2star[i]

          y[ihi] = y2star
#
#  No extension.
#
      else:

        l = 0
        for i in range ( 0, n + 1 ):
          if ( ystar < y[i] ):
            l = l + 1

        if ( 1 < l ):

          for i in range ( 0, n ):
            p[i,ihi] = pstar[i]

          y[ihi] = ystar
#
#  Contraction on the Y(IHI) side of the centroid.
#
        elif ( l == 0 ):

          for i in range ( 0, n ):
            p2star[i] = pbar[i] + ccoeff * ( p[i,ihi] - pbar[i] )

          y2star = fn ( p2star )
          icount = icount + 1
#
#  Contract the whole simplex.
#
          if ( y[ihi] < y2star ):

            for j in range ( 0, n + 1 ):
              for i in range ( 0, n ):
                p[i,j] = ( p[i,j] + p[i,ilo] ) * 0.5
                xmin[i] = p[i,j]
              y[j] = fn ( xmin )
              icount = icount + 1

            ylo = y[0]
            ilo = 0

            for i in range ( 1, n + 1 ):
              if ( y[i] < ylo ):
                ylo = y[i]
                ilo = i

            continue
#
#  Retain contraction.
#
          else:

            for i in range ( 0, n ):
              p[i,ihi] = p2star[i]
            y[ihi] = y2star

#
#  Contraction on the reflection side of the centroid.
#
        elif ( l == 1 ):

          for i in range ( 0, n ):
            p2star[i] = pbar[i] + ccoeff * ( pstar[i] - pbar[i] )

          y2star = fn ( p2star )
          icount = icount + 1
#
#  Retain reflection?
#
          if ( y2star <= ystar ):

            for i in range ( 0, n ):
              p[i,ihi] = p2star[i]
            y[ihi] = y2star

          else:

            for i in range ( 0, n ):
              p[i,ihi] = pstar[i]
            y[ihi] = ystar
#
#  Check if YLO improved.
#
      if ( y[ihi] < ylo ):
        ylo = y[ihi]
        ilo = ihi

      jcount = jcount - 1

      if ( 0 < jcount ):
        continue
#
#  Check to see if minimum reached.
#
      if ( icount <= kcount ):

        jcount = konvge

        z = 0.0
        for i in range ( 0, n + 1 ):
          z = z + y[i]
        x = z / float ( n + 1 )

        z = 0.0
        for i in range ( 0, n + 1 ):
          z = z + ( y[i] - x ) ** 2

        if ( z <= rq ):
          break
#
#  Factorial tests to check that YNEWLO is a local minimum.
#
    for i in range ( 0, n ):
      xmin[i] = p[i,ilo]

    ynewlo = y[ilo]

    if ( kcount < icount ):
      ifault = 2
      break

    ifault = 0

    for i in range ( 0, n ):
      delta = step[i] * eps
      xmin[i] = xmin[i] + delta
      z = fn ( xmin )
      icount = icount + 1
      if ( z < ynewlo ):
        ifault = 2
        break
      xmin[i] = xmin[i] - delta - delta
      z = fn ( xmin )
      icount = icount + 1
      if ( z < ynewlo ):
        ifault = 2
        break
      xmin[i] = xmin[i] + delta

    if ( ifault == 0 ):
      break
#
#  Restart the procedure.
#
    for i in range ( 0, n ):
      start[i] = xmin[i]

    delta = eps
    numres = numres + 1

  return xmin, ynewlo, icount, numres, ifault