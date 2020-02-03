###############################################################################
# Version: 1.1
# Last modified on: 3 April, 2016 
# Developers: Michael G. Epitropakis
#      email: m_(DOT)_epitropakis_(AT)_lancaster_(DOT)_ac_(DOT)_uk 
###############################################################################
import numpy as np
import math
from cfunction import *
###############################################################################
# Basic Benchmark functions 
###############################################################################

###############################################################################
# F1: Five-Uneven-Peak Trap
# Variable ranges: x in [0, 30
# No. of global peaks: 2
# No. of local peaks:  3.
def five_uneven_peak_trap(x = None):
    if x==None:
        return None

    result = None
    if (x[0]>=0 and x[0]<2.50):
        result = 80*(2.5-x[0])
    elif (x[0]>=2.5 and x[0]<5):
        result = 64*(x[0]-2.5)
    elif (x[0] >= 5.0 and x[0] < 7.5):
        result = 64*(7.5-x[0])
    elif (x[0] >= 7.5 and x[0] < 12.5):
        result = 28*(x[0]-7.5)
    elif (x[0] >= 12.5 and x[0] < 17.5):
        result = 28*(17.5-x[0])
    elif (x[0] >= 17.5 and x[0] < 22.5):
        result = 32*(x[0]-17.5)
    elif (x[0] >= 22.5 and x[0] < 27.5):
        result = 32*(27.5-x[0])
    elif (x[0] >= 27.5 and x[0] <= 30):
        result = 80*(x[0]-27.5)
    return result

###############################################################################
# F2: Equal Maxima
# Variable ranges: x in [0, 1]
# No. of global peaks: 5
# No. of local peaks:  0. 
def equal_maxima(x = None):

    if x==None:
        return None

    return np.sin(5.0 * np.pi * x[0])**6

###############################################################################
# F3: Uneven Decreasing Maxima
# Variable ranges: x in [0, 1]
# No. of global peaks: 1
# No. of local peaks:  4. 
def uneven_decreasing_maxima(x = None):
    
    if x == None:
        return None

    return np.exp(-2.0*np.log(2)*((x[0]-0.08)/0.854)**2)*(np.sin(5*np.pi*(x[0]**0.75-0.05)))**6

###############################################################################
# F4: Himmelblau
# Variable ranges: x, y in [-6, 6
# No. of global peaks: 4
# No. of local peaks:  0.
def himmelblau(x = None):
    
    if x==None:
        return None

    result = 200 - (x[0]**2 + x[1] - 11)**2 - (x[0] + x[1]**2 - 7)**2
    return result

###############################################################################
# F5: Six-Hump Camel Back
# Variable ranges: x in [-1.9, 1.9]; y in [-1.1, 1.1]
# No. of global peaks: 2
# No. of local peaks:  2.
def six_hump_camel_back(x = None):

    if x==None:
        return None

    x2 = x[0]**2
    x4 = x[0]**4
    y2 = x[1]**2
    expr1 = (4.0 - 2.1*x2 + x4/3.0)*x2
    expr2 = x[0]*x[1]
    expr3 = (4.0*y2 - 4.0)*y2
    return -1.0*(expr1+expr2+expr3)
    #result = (-4)*((4 - 2.1*(x[0]**2) + (x[0]**4)/3.0)*(x[0]**2) + x[0]*x[1] + (4*(x[1]**2) - 4)*(x[1]**2))
    #return result

###############################################################################
# F6: Shubert
# Variable ranges: x_i in  [-10, 10]^n, i=1,2,...,n
# No. of global peaks: n*3^n
# No. of local peaks: many
def shubert(x = None):

    if x==None:
        return None

    i = 0
    result = 1
    soma = [0]*len(x)
    D = len(x)


    while i < D:
        for j in range (1, 6):
            soma[i] = soma[i] + (j*math.cos((j+1)*x[i]+j))
        result = result*soma[i]
        i = i + 1
    return -result

###############################################################################
# F7: Vincent
# Variable range: x_i in [0.25, 10]^n, i=1,2,...,n
# No. of global optima: 6^n
# No. of local optima:  0.
def vincent(x = None):
    #print(x)
    if x==None:
        return None

    result = 0
    D = len(x)
    for i in range(0, D):
        result += (math.sin(10*math.log(x[i])))/D
    return result

###############################################################################
# F8: Modified Rastrigin - All Global Optima
# Variable ranges: x_i in [0, 1]^n, i=1,2,...,n
# No. of global peaks: \prod_{i=1}^n k_i
# No. of local peaks:  0.
def modified_rastrigin_all(x = None):

    if x==None:
        return None

    result = 0
    D = len(x)    
    if D==2:
        k = [3, 4]
    elif D==8:
        k = [1, 2, 1, 2, 1, 3, 1, 4]
    elif D==16:
        k = [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 4]

    for i in range (0, D):
        result += (10 + 9*math.cos(2*math.pi*k[i]*x[i]))        
    return -result

def protein(individual = None):
    #print(individual)


    size = len(individual)
    a_ab = b_ab = c_ab = d_ab = v1 = v2 = 0
    if size == 38:
        AB_21 = "AAAABABABABABAABAABBAAABBABAABBBABABAB"
    elif size == 64:
        AB_21 = "ABBABAABBABABBBAABBABABBBABBABABBABABBABABABAABABBAABBABBBAAABAB"
    elif size == 98:
        AB_21 = "AABABAAAAAAABBBAAAAAABAABAABBAABABAAABBBAAAABABAAABABBAAABAAABAAABAABBAABAAAAABAAABABBBABBAAABAABA"
    elif size == 120:
        AB_21 = "ABBABBAABABABAABBAAAABAABABBABABBAAABBBAABBBABAAABABBABBABBBBABBBBAABBBBBBBABABBAAAABBBBBBABBBBAAAABBBABABBBBAAAABBABABB"

    i = 0
    j = 0

    x = [0] * size
    y = [0] * size

    x[0] = y[0] = 0
    x[1] = 1
    y[1] = 0
    # amino_pos[0].x = amino_pos[0].y = 0;
    # amino_pos[1].x = 1;
    # amino_pos[1].y = 0;

    for i in range(1, size-1):
        a_ab = x[i] - x[i-1]
        b_ab = y[i] - y[i-1]
        x[i+1] = x[i] + a_ab * math.cos(individual[i-1]) - b_ab * math.sin(individual[i-1])
        y[i+1] = y[i] + b_ab * math.cos(individual[i-1]) + a_ab * math.sin(individual[i-1])
    
    # for (i = 1; i < (size-1); i++)
    #     a_ab = amino_pos[i].x-amino_pos[i-1].x;
    #     b_ab = amino_pos[i].y-amino_pos[i-1].y;
    #     amino_pos[i+1].x = amino_pos[i].x+a_ab*cos(individual[i-1])-b_ab*sin(individual[i-1]);
    #     amino_pos[i+1].y = amino_pos[i].y+b_ab*cos(individual[i-1])+a_ab*sin(individual[i-1]);

    v1 = 0
    for i in range(1, size-1):
        v1 += (1.0 - math.cos(individual[i-1])) / 4.0

    # for( i = 1; i < ( size - 1 ); i++ )
    #     v1 += (1.0 - cos(individual[i-1]) ) / 4.0;
    

    v2 = 0
    for i in range(0, size-2):
        for j in range(i+2, size):
            if AB_21[i] == 'A' and AB_21[j] == 'A':
                c_ab = 1
            elif AB_21[i] == 'B' and AB_21[j] == 'B':
                c_ab = 0.5
            else:
                c_ab = -0.5

            d_ab = math.sqrt(((x[i] - x[j]) * (x[i]- x[j])) + ((y[i] - y[j]) * (y[i] - y[j])))
            v2 += 4.0 * (1 / math.pow(d_ab, 12) - c_ab / math.pow(d_ab, 6))

    return -(v1 + v2)



def protein3D(individual = None):

    points.append(0.0, 0.0, 0.0)
    points.append(0.0, 1.0, 0.0)
    points.append(math.cos(individual[0]), 1.0 + math.sin(individual[0]), 0.0)

    x = points[0][2]
    y = points[1][2]
    z = points[2][2]

    theta = individual[0]
    beta = individual[PL-2]

    for i in range(3, PL):
        x += math.cos(theta[i-2]) * math.cos(beta[i-3])
        y += math.sin(theta[i-2]) * math.cos(beta[i-3])
        z += math.sin(beta[i-3])

        points.append(x, y, z)

    v1 = 0.0
    v2 = 0.0

    for i in range(0, PL-2):
        v1 += 1 - math.cos(theta[i])
        for j in range(i+2, PL):
            if(AB_SQ[i] == 'A' and AB_SQ[j] == 'A'):
                c_ab = 1
            elif(AB_SQ[i] == 'B' and AB_SQ[j] == 'B'):
                c_ab = 0.5
            else:
                c_ab = -0.5

            xi = points[i][0]
            xj = points[j][0]

            yi = points[i][1]
            yj = points[j][1]

            zi = points[i][2]
            zj = points[i][2]

            dx = (xi - xj)
            dx = dx*dx

            dy = (yi - yj)
            dy = dy*dy

            dz = (zi - zj)
            dz = dz*dz

            D = math.sqrt(dx + dy + dz)

            v2 += ( 1 / math.pow(D, 12) - c_ab / math.pow(D,6) )

    print("v1 = %f v2 = %f \n", v1/4.0, 4*v2)
    print("final energy value: %f \n", (v1/4.0 + 4*v2))

    return ((v1/4.0) + (4*v2))

   

# double evaluate(double * S){
#   points.clear();

#     points.push_back( std::make_tuple(0.0, 0.0, 0.0) ); // [0]
#     points.push_back( std::make_tuple(0.0, 1.0, 0.0) ); // [1]
#   points.push_back( std::make_tuple(cos(S[0]), 1.0 + sin(S[0]), 0.0) ); // [2]

#     double _x, _y, _z;
#     _x = std::get<0>(points[2]);
#     _y = std::get<1>(points[2]);
#     _z = std::get<2>(points[2]);

#     double * theta = &S[0];
#     double * beta  = &S[PL-2];

#     for( uint16_t i = 3; i < PL; i++ ){
#         _x += cos(theta[i-2])*cos(beta[i-3]);
#         _y += sin(theta[i-2])*cos(beta[i-3]);
#         _z += sin(beta[i-3]);

#         points.push_back(std::make_tuple(_x, _y, _z));
#     }

#     // printf("Pontos: \n");
#     // for( uint16_t i = 0; i < PL; i++ ){
#     //  printf("%.3f %.3f %.3f\n", std::get<0>(points[i]), std::get<1>(points[i]), std::get<2>(points[i]));
#     // }

#     double v1 = 0.0, v2 = 0.0;
#     double xi, xj, yi, yj, zi, zj, dx, dy, dz, D;
#     double c_ab;
#     for( uint16_t i = 0; i < PL-2; i++ ){
#         v1 += 1 - cos(theta[i]);
#         for( uint16_t j = i + 2; j < PL; j++ ){
#             if (AB_SQ[i] == 'A' && AB_SQ[j] == 'A') //AA bond
#                 c_ab = 1;
#             else if (AB_SQ[i] == 'B' && AB_SQ[j] == 'B') //BB bond
#                 c_ab = 0.5;
#             else
#                 c_ab = -0.5; //AB or BA bond

#             xi = std::get<0>(points[i]);
#             xj = std::get<0>(points[j]);

#             yi = std::get<1>(points[i]);
#             yj = std::get<1>(points[j]);

#             zi = std::get<2>(points[i]);
#             zj = std::get<2>(points[j]);

#             dx = (xi - xj);
#             dx *= dx;

#             dy = (yi - yj);
#             dy *= dy;

#             dz = (zi - zj);
#             dz *= dz;

#             D = sqrt(dx + dy + dz);

#             v2 += ( 1 / pow(D, 12) - c_ab / pow(D, 6) );
#         }
#     }
#     // printf("v1: %.4lf v2: %.4lf\n", v1/4, 4*v2);
#     // printf("Final energy value: %.8lf\n", v1/4 + 4*v2);
#     return(v1/4 + 4*v2);
# }



# def protein64(individual = None):
#     #print(individual)
#     size = len(individual)
#     a_ab = b_ab = c_ab = d_ab = v1 = v2 = 0
#     AB_21 = "ABBABAABBABABBBAABBABABBBABBABABBABABBABABABAABABBAABBABBBAAABAB"
    
#     i = 0
#     j = 0

#     x = [0] * size
#     y = [0] * size

#     x[0] = y[0] = 0
#     x[1] = 1
#     y[1] = 0
#     # amino_pos[0].x = amino_pos[0].y = 0;
#     # amino_pos[1].x = 1;
#     # amino_pos[1].y = 0;

#     for i in range(1, size-1):
#         a_ab = x[i] - x[i-1]
#         b_ab = y[i] - y[i-1]
#         x[i+1] = x[i] + a_ab * math.cos(individual[i-1]) - b_ab * math.sin(individual[i-1])
#         y[i+1] = y[i] + b_ab * math.cos(individual[i-1]) + a_ab * math.sin(individual[i-1])
    
#     # for (i = 1; i < (size-1); i++)
#     #     a_ab = amino_pos[i].x-amino_pos[i-1].x;
#     #     b_ab = amino_pos[i].y-amino_pos[i-1].y;
#     #     amino_pos[i+1].x = amino_pos[i].x+a_ab*cos(individual[i-1])-b_ab*sin(individual[i-1]);
#     #     amino_pos[i+1].y = amino_pos[i].y+b_ab*cos(individual[i-1])+a_ab*sin(individual[i-1]);

#     v1 = 0
#     for i in range(1, size-1):
#         v1 += (1.0 - math.cos(individual[i-1])) / 4.0

#     # for( i = 1; i < ( size - 1 ); i++ )
#     #     v1 += (1.0 - cos(individual[i-1]) ) / 4.0;
    

#     v2 = 0
#     for i in range(0, size-2):
#         for j in range(i+2, size):
#             if AB_21[i] == 'A' and AB_21[j] == 'A':
#                 c_ab = 1
#             elif AB_21[i] == 'B' and AB_21[j] == 'B':
#                 c_ab = 0.5
#             else:
#                 c_ab = -0.5

#             d_ab = math.sqrt(((x[i] - x[j]) * (x[i]- x[j])) + ((y[i] - y[j]) * (y[i] - y[j])))
#             v2 += 4.0 * (1 / math.pow(d_ab, 12) - c_ab / math.pow(d_ab, 6))

#     return -(v1 + v2)

# def protein98(individual = None):
#     #print(individual)
#     size = len(individual)
#     a_ab = b_ab = c_ab = d_ab = v1 = v2 = 0
#     AB_21 = "AABABAAAAAAABBBAAAAAABAABAABBAABABAAABBBAAAABABAAABABBAAABAAABAAABAABBAABAAAAABAAABABBBABBAAABAABA"
    
#     i = 0
#     j = 0

#     x = [0] * size
#     y = [0] * size

#     x[0] = y[0] = 0
#     x[1] = 1
#     y[1] = 0
#     # amino_pos[0].x = amino_pos[0].y = 0;
#     # amino_pos[1].x = 1;
#     # amino_pos[1].y = 0;

#     for i in range(1, size-1):
#         a_ab = x[i] - x[i-1]
#         b_ab = y[i] - y[i-1]
#         x[i+1] = x[i] + a_ab * math.cos(individual[i-1]) - b_ab * math.sin(individual[i-1])
#         y[i+1] = y[i] + b_ab * math.cos(individual[i-1]) + a_ab * math.sin(individual[i-1])
    
#     # for (i = 1; i < (size-1); i++)
#     #     a_ab = amino_pos[i].x-amino_pos[i-1].x;
#     #     b_ab = amino_pos[i].y-amino_pos[i-1].y;
#     #     amino_pos[i+1].x = amino_pos[i].x+a_ab*cos(individual[i-1])-b_ab*sin(individual[i-1]);
#     #     amino_pos[i+1].y = amino_pos[i].y+b_ab*cos(individual[i-1])+a_ab*sin(individual[i-1]);

#     v1 = 0
#     for i in range(1, size-1):
#         v1 += (1.0 - math.cos(individual[i-1])) / 4.0

#     # for( i = 1; i < ( size - 1 ); i++ )
#     #     v1 += (1.0 - cos(individual[i-1]) ) / 4.0;
    

#     v2 = 0
#     for i in range(0, size-2):
#         for j in range(i+2, size):
#             if AB_21[i] == 'A' and AB_21[j] == 'A':
#                 c_ab = 1
#             elif AB_21[i] == 'B' and AB_21[j] == 'B':
#                 c_ab = 0.5
#             else:
#                 c_ab = -0.5

#             d_ab = math.sqrt(((x[i] - x[j]) * (x[i]- x[j])) + ((y[i] - y[j]) * (y[i] - y[j])))
#             v2 += 4.0 * (1 / math.pow(d_ab, 12) - c_ab / math.pow(d_ab, 6))

#     return -(v1 + v2)


# def protein120(individual = None):
#     #print(individual)
#     size = len(individual)
#     a_ab = b_ab = c_ab = d_ab = v1 = v2 = 0
#     AB_21 = "ABBABBAABABABAABBAAAABAABABBABABBAAABBBAABBBABAAABABBABBABBBBABBBBAABBBBBBBABABBAAAABBBBBBABBBBAAAABBBABABBBBAAAABBABABB"
    
#     i = 0
#     j = 0

#     x = [0] * size
#     y = [0] * size

#     x[0] = y[0] = 0
#     x[1] = 1
#     y[1] = 0
#     # amino_pos[0].x = amino_pos[0].y = 0;
#     # amino_pos[1].x = 1;
#     # amino_pos[1].y = 0;

#     for i in range(1, size-1):
#         a_ab = x[i] - x[i-1]
#         b_ab = y[i] - y[i-1]
#         x[i+1] = x[i] + a_ab * math.cos(individual[i-1]) - b_ab * math.sin(individual[i-1])
#         y[i+1] = y[i] + b_ab * math.cos(individual[i-1]) + a_ab * math.sin(individual[i-1])
    
#     # for (i = 1; i < (size-1); i++)
#     #     a_ab = amino_pos[i].x-amino_pos[i-1].x;
#     #     b_ab = amino_pos[i].y-amino_pos[i-1].y;
#     #     amino_pos[i+1].x = amino_pos[i].x+a_ab*cos(individual[i-1])-b_ab*sin(individual[i-1]);
#     #     amino_pos[i+1].y = amino_pos[i].y+b_ab*cos(individual[i-1])+a_ab*sin(individual[i-1]);

#     v1 = 0
#     for i in range(1, size-1):
#         v1 += (1.0 - math.cos(individual[i-1])) / 4.0

#     # for( i = 1; i < ( size - 1 ); i++ )
#     #     v1 += (1.0 - cos(individual[i-1]) ) / 4.0;
    

#     v2 = 0
#     for i in range(0, size-2):
#         for j in range(i+2, size):
#             if AB_21[i] == 'A' and AB_21[j] == 'A':
#                 c_ab = 1
#             elif AB_21[i] == 'B' and AB_21[j] == 'B':
#                 c_ab = 0.5
#             else:
#                 c_ab = -0.5

#             d_ab = math.sqrt(((x[i] - x[j]) * (x[i]- x[j])) + ((y[i] - y[j]) * (y[i] - y[j])))
#             v2 += 4.0 * (1 / math.pow(d_ab, 12) - c_ab / math.pow(d_ab, 6))

#     return -(v1 + v2)