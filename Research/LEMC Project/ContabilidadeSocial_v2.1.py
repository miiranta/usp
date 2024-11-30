#DEPENDÊNCIAS ===========================================

import matplotlib.pyplot as plt
import numpy as np

#CONFIG =================================================

np.set_printoptions(precision = 3)

#VARIÁVEIS ==============================================

#Globais

b = 3
s = 1 #Assumimos como = 1 sempre.

#Ponto Fixo

e =  10**-16
MAXit = 10000

#Outras

P = np.full((b), 4)
K = np.full((b, b), 1/9)
W = np.full((s, b), 1)
R = np.ones((s, b))
A = np.full((b), 2)

#------

alpha = np.full((b, b), 1/9)
beta  = 0.98
zeta  = np.full((b), 0.1)
delta = 0.98

csi = np.full((b), 0.3)
v = 0.8

theta = np.full((b), 0.2)
ro    = np.full((b), 0.2)

nEst = np.full((b), 0.2)
oEst = np.full((b), 0.6)

w = np.full((b), 1/3)

#-----

phic = np.full((b), 1.5)
phil = np.full((s), 0.5)

#-----

gammac = 0.1
gammal = 0.1
gamma  = 0.1
gammah = 0.1

#-----

Az = np.full((b), 1.5)
Ak = np.full((b), 1.0)
Ah = np.full((b), 3)
Af = np.full((b), 0.1)

#-----

az = np.full((b, b), 2)
ak = np.full((b), 2)
ah = np.full((b), 1)
aR = np.full((b), 100)
ac = np.full((b), 100)
al = np.full((s), 2)

#-----

lamk = np.full((b), 0.9)
lamh = np.full((b), 0.8)

#-----

pTil = np.ones(b)
rTil = np.ones(b)
wTil = np.ones(b)
qTil = np.ones(b)
cTil = np.ones(b)

#-----

wBar = np.full((b), 1000.0)
hBar  = np.full((s), 1000.0)

#ASSERTS =================================================

#Na verdade, não é só o primeiro elemento que precisa ser testado, e sim todos. Mas assim fica mais bonito ;).
assert (0 < ro[0] and ro[0] < 1),           "0 < ro < 1"
assert (0 < oEst[0] and oEst[0] < 1),       "0 < oEst < 1"
assert (0 < csi[0] and csi[0] < 1),         "0 < csi < 1"

assert (0 < lamk[0] and lamk[0] < 1),       "0 < lamk < 1"
assert (0 < lamh[0] and lamh[0] < 1),       "0 < lamh < 1"

assert (0 <= theta[0] and theta[0] <= 1),   "0 <= theta <= 1"
assert (0 <= nEst[0] and nEst[0] <= 1),     "0 <= nEst <= 1"

assert (alpha[0][0] > 0),                   "alpha > 0"
assert (np.sum(alpha) == 1),                "alpha11 + alpha12... = 1"

assert (K[0][0] > 0),                       "K > 0"
assert (np.sum(K) == 1),                    "K11 + K12... = 1"

assert (w[0] > 0),                          "w > 0"
assert (np.sum(w) == 1),                    "w11 + w12... = 1"


#FUNÇÕES =================================================

#Globais

#-----

def Rp(P, i, beta, delta, aR):
    return (1/beta - 1 + delta) * P[i] * aR[i]

#----- (Til's)

def PTil(P, j, alpha, az, zeta, Az):
    sum = 0

    for n in range(len(P)):
        sum = (alpha[n][j] / (az[n][j] * P[n]) ** zeta[j]) ** (1 / (1 - zeta[j])) + sum

    sum = sum ** ((zeta[j] - 1) / zeta[j])
    sum = sum * Az[j]

    return sum

def RTil(R, j, K, ak, lamk, Ak):
    sum = 0

    for n in range(len(R)):
        #ak[n] deveria ser ak[n][j]
        #R[n] deveria ser R[n][j]
        #K[n] deveria ser K[n][j]
        sum = (K[n] / (ak[n] * R[n]) ** lamk[j]) ** (1 / (1 - lamk[j])) + sum

    sum = sum ** ((lamk[j] - 1) / lamk[j])
    sum = sum * Ak[j]

    return sum

def WTil(W, j, w, ah, lamh, Ah):
    sum = 0
    
    for n in range(len(W)):
        #ah[n] deveria ser ah[n][j]
        #W[n] deveria ser W[n][j]
        #w[n] deveria ser w[n][j]
        sum = (w[n] / (ah[n] * W[n]) ** lamh[j]) ** (1 / (1 - lamh[j])) + sum

    sum = sum ** ((lamh[j] - 1) / lamh[j])
    sum = sum * Ah[j]

    return sum

def QTil(rTil, wTil, j, theta, ro, Af):
    res = (theta[j] / rTil[j] ** ro[j] ) ** (1 / (1 - ro[j]))
    res = ((1 - theta[j]) / wTil[j] ** ro[j]) ** (1 / (1 - ro[j])) + res
    res = res ** ((ro[j] - 1) / ro[j]) * Af[j]
    return res

def CTil(pTil, qTil, j, oEst, nEst, A):
    res = (nEst[j] / pTil[j] ** oEst[j]) ** (1 / (1 - oEst[j]))
    res = ((1 - nEst[j]) / qTil[j] ** oEst[j]) ** (1 / (1 - oEst[j])) + res
    res = res ** ((oEst[j] - 1) / oEst[j]) * A[j]
    return res

#----- (Circ's)
    
def PCirc(P, gammac, phic, ac):
    
    for n in range(b):
        sum = (phic[n] / (ac[n] * P[n]) ** gammac) ** (1 / (1 - gammac))
        sum = sum ** ((gammac - 1) / gammac)
    
    return sum

def WCirc(wBar, gammal, phil, al):
    
    for n in range(s):
        sum = (phil[n] / (al[n] * wBar[n]) ** gammal) ** (1 / (1 - gammal))
        sum = sum ** ((gammal - 1) / gammal)

    return sum

#----- (Bar's)

def WBar(W):
    return W.max(axis=0)

#Ponto Fixo

def nextP(P, R, W, pTil, rTil, wTil, qTil, cTil):

    #Calcula R
    for j in range(b):
        R[:, j] = Rp(P, j, beta, delta, aR)

    #pTil
    for j in range(b):
        pTil[j] = PTil(P, j, alpha, az, zeta, Az)

    #rTil
    for j in range(b):
        rTil[j] = RTil(R[:, j], j, K[0], ak, lamk, Ak)

    #wTil
    for j in range(b):
        wTil[j] = WTil(W[:, j], j, w, ah, lamh, Ah)

    #qTil
    for j in range(b):
        qTil[j] = QTil(rTil, wTil, j, theta, ro, Af)

    #cTil (Novo P)
    cTil = np.ones(b)
    for j in range(b):
        cTil[j] = CTil(pTil, qTil, j, oEst, nEst, A)

    # print("P:", P)
    # print("R:", R)
    # print("W:", W)
    # print("pTil:", pTil)
    # print("rTil:", rTil)    
    # print("wTil:", wTil)
    # print("qTil:", qTil)
    # print("cTil (novo P):", cTil)
    # print("===============================")
    
    return cTil

def calcPontoFixo(P, R, W, pTil, rTil, wTil, qTil, cTil):
    it = 0
    P1 = np.full((b), 1)

    while(it < MAXit):
        P1 = nextP(P, R, W, pTil, rTil, wTil, qTil, cTil)

        if (max(abs(P1 - P)) < e):
            break
        
        P = P1
        
        #print("Iteração " + str(it))
        it = it + 1

    return P

#Sistema Linear

def eq1_c_coef(i, phil, wCirc, wBar, al, gammal, pCirc, v, gamma):
    
    first   = ( ((phil[i] * wCirc) / wBar[i]) / al[i] ** gammal ) ** (1 / (1 - gammal))
    second  = ((pCirc / wCirc) * ((1 - v) / v)) ** (1 / (1 - gamma)) 

    return first * second

def eq1_y_coef(j, w, wTil, W, Ah, ah, lamh, gammah, theta, qTil, Af, ro, nEst, cTil, A, oEst):
    
    first   = ( ( ( w[j] * wTil[j] ) / (W[j]) ) / ( Ah[j] * ah[j] ) ** lamh[j] ) ** (1 / (1 - lamh[j]))
    second  = ( ( ( 1 - theta[j] ) * qTil[j] ) / ( Af[j] ** ro[j] * wTil[j] ) ) ** (1 / (1 - ro[j]))
    third   = ( ( ( 1 - nEst[j] ) * cTil[j] ) / ( A[j] ** oEst[j] * qTil[j] ) ) ** (1 / (1 - oEst[j]))
    
    return first * second * third
      
def eq2_c_coef(i, phic, pCirc, P, ac, gammac):
    
    first = ( ( (phic[i] * pCirc) / P[i] ) / ( ac[i] ** gammac ) ) ** (1 / (1 - gammac))

    return first

def eq2_k_coef(i, ak, delta):
    
    first = ak[i] * delta
    
    return first

def eq2_y_coef(i, j, alpha, pTil, P, Az, az, csi, nEst, cTil, A, oEst):

    first   = ( ( ( alpha[i][j] * pTil[j] ) / P[j] ) / ( Az[j] * az[i][j] ) ** csi[j] ) ** (1 / (1 - csi[j]))
    second  = ( (nEst[j] * cTil[j]) / ( A[j] ** oEst[j] * pTil[j] ) ) ** (1 / (1 - oEst[j]))

    return first * second

def eq3_y_coef(i, j, K, rTil, R, Ak, ak, lamh, theta, qTil, Af, ro, nEst, cTil, A, oEst):
    
    first   = ( ( ( K[i][j] * rTil[j] ) / ( R[j] ) ) / ( Ak[j] * ak[j] ) ** lamh[j] ) ** (1 / (1 - lamk[j]))
    second  = ( ( theta[j] * qTil[j] ) / ( Af[j] ** ro[j] * rTil[j] ) ) ** (1 / (1 - ro[j]))
    third   = ( ( ( 1 - nEst[j] ) * cTil[j] ) / ( A[j] ** oEst[j] * qTil[j] ) ) ** (1 / ( 1 - oEst[j] ))
    
    return first * second * third

def calcularSistemaLinear(K, P, R, A, W, Af, Ak, Ah, Az, ac, ak, az, ah, al, lamh, alpha, csi, delta, v, w, theta, ro, nEst, oEst, pTil, qTil, rTil, wTil, cTil, wCirc, pCirc, wBar, hBar,  gammal, gammac, gamma, gammah, phic, phil):
    
    #recalcula o ponto fixo
    #é necessário para recalcular as dependencias do zeta na analise gráfica, mas pode ser substituido por chamadas individuais para recalcular o R e os Til's
    P = calcPontoFixo(P, R, W, pTil, rTil, wTil, qTil, cTil)
    
    mA = np.full((2*b+1, 2*b+1), 0.0) 
    mB = np.full((2*b+1, 1), 0.0)

    for i in range(s):

        for j in range(s):
            mA[i, j] = eq1_c_coef(j, phil, wCirc, wBar, al, gammal, pCirc, v, gamma)

        for j in range(b):
            mA[i, j+b+1] = eq1_y_coef(j, w, wTil, W[0], Ah, ah, lamh, gammah, theta, qTil, Af, ro, nEst, cTil, A, oEst)

        for j in range(s):
            mB[j] = hBar[j]

    for i in range(b):

        for j in range(s):
            mA[i+s, j] = eq2_c_coef(i, phic, pCirc, P, ac, gammac)

        mA[i+s, s+i] = eq2_k_coef(i, ak, delta)

        for j in range(b):
            mA[i + s, j + b + 1] = eq2_y_coef(i, j, alpha, pTil, P, Az, az, csi, nEst, cTil, A, oEst)
        
        mA[i+s, i+b+1] = mA[i+s, i+b+1] - 1

    for i in range(b):
        
        mA[i+s+b, s+i] = -1
        
        for j in range(b):
            mA[i+s+b, j+b+1] = eq3_y_coef(i, j, K, rTil, R[0], Ak, ak, lamh, theta, qTil, Af, ro, nEst, cTil, A, oEst)

    #print("\nA\n", mA, "\n\n", "B\n", mB, "\n")

    return np.linalg.solve(mA, mB)

#RUN =====================================================

#Ponto Fixo

P = calcPontoFixo(P, R, W, pTil, rTil, wTil, qTil, cTil)
print("Peq\n", P)

#Sistema Linear

wBar = WBar(W)
pCirc = PCirc(P, gammac, phic, ac)
wCirc = WCirc(wBar, gammal, phil, al)

newQ = calcularSistemaLinear(K, P, R, A, W, Af, Ak, Ah, Az, ac, ak, az, ah, al, lamh, alpha, csi, delta, v, w, theta, ro, nEst, oEst, pTil, qTil, rTil, wTil, cTil, wCirc, pCirc, wBar, hBar, gammal, gammac, gamma, gammah, phic, phil)
print("\nQeq\n", newQ)

#PLOT ====================================================

# gamma x consumo
#x é o gamma
#y é o consumo

x_gamma = np.linspace(0.2, 0.8, 100)
y_consumo = []

for x in x_gamma:   
    newQ = calcularSistemaLinear(K, P, R, A, W, Af, Ak, Ah, Az, ac, ak, az, ah, al, lamh, alpha, csi, delta, v, w, theta, ro, nEst, oEst, pTil, qTil, rTil, wTil, cTil, wCirc, pCirc, wBar, hBar, gammal, gammac, x, gammah, phic, phil)
    y_consumo.append(newQ[0][0])

plt.plot(x_gamma, y_consumo)
plt.show()

# zeta x produção
#x é o zeta
#y é o produção (y)

x_zeta = np.linspace(0.2, 0.8, 100)
y_producao = []

for x in x_zeta:
    zeta = np.full((b), x) #neste caso, é necessário alterar o valor da global zeta  
    
    newQ = calcularSistemaLinear(K, P, R, A, W, Af, Ak, Ah, Az, ac, ak, az, ah, al, lamh, alpha, csi, delta, v, w, theta, ro, nEst, oEst, pTil, qTil, rTil, wTil, cTil, wCirc, pCirc, wBar, hBar, gammal, gammac, gamma, gammah, phic, phil)  
    y_producao.append(newQ[4][0])

plt.plot(x_zeta, y_producao)
plt.show()




