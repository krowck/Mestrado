def hooke_jeeves(point, func, dx=0.5, e=EPSILON, print_stats=False):
    """
    Hooke-Jeeves optimization algorithm
    :param point: Starting point from which algorithm starts
    :param func: Function that is subjected to the optimization
    :param dx: Initial step that algorithm uses to move points, every iteration step is divided by 2
    :param e: Error epsilon
    :param print_stats: If set to True function will print steps to the console
    :return: Point in which algorithm found the minimum of the function
    """

    if not isinstance(func, Function):
        raise ValueError("func parameter has to be of type Function")

    x0 = copy.deepcopy(point)
    xb = copy.deepcopy(point)
    xp = copy.deepcopy(point)
    xn = None

    func.reset_iterations()

    while True:
        if dx <= e:
            break
        xn = _hooke_jeeves_search(xp, func, dx)
        f_xn = func.calc(*xn)
        f_xb = func.calc(*xb)
        if f_xn < f_xb:
            for i in range(0, len(xn)):
                xp[i] = 2*xn[i] - xb[i]
                xb[i] = xn[i]
        else:
            dx /= 2
            xp = copy.deepcopy(xb)

        if print_stats == True:
            print("Xb: {0} Xp: {1} Xn: {2}".format(xb, xp, xn))

    return xb


def _hooke_jeeves_search(xp, func, dx):
    x = copy.deepcopy(xp)
    for i in range(0, len(xp)):
        p = func.calc(*x)
        x[i] += dx
        n = func.calc(*x)
        if n > p:
            x[i] -= 2*dx
            n = func.calc(*x)
            if n > p:
                x[i] += dx
    return x