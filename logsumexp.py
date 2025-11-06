import matplotlib.pyplot as plt
import math

def a(x):
    return x**2*(1+2*x**2)/(1+x**2)

def da(x):
    return (4*x**5+8*x**3+2*x)/(1+x**2)**2
def b(x):
    return x**2

def db(x):
    return 2*x
def c(x):
    return math.floor(x)**2+(x-math.floor(x))*(2*math.floor(x)+1)

def dc(x):
    return 2*math.floor(x)+1

def m(x):
    return -math.cos(x)

def dm(x):
    return math.sin(x)

def l(x):
    return math.log(math.exp(x)+1)

def dl(x):
    return math.exp(x)/(math.exp(x)+1)

def m_legendre(x):
    return x*math.asin(x)-math.sqrt(1-x**2)

E=[0.001, 0.005, 0.01, 0.1, 1]
objectivt_list = [[l, dl, l, dl], [a, da, a, da], [b, db, b, db], [c, dc, c, dc], [m, dm, m, dm], [a, da, b, db], [b, db, c, dc]]
title_list = ['f=g=log(1+e^x)', 'f=g=(x^2)(1+2x^2)/(1+x^2)', 'f=g=x^2', 'f=g=floor(x)^2+[x-floor(x)]*[2*floor(x)-1]', 'f=g=cos(x)', 'f=(x^2)(1+2x^2)/(1+x^2), g=x^2', 'f=x^2, g=floor(x)^2+[x-floor(x)]*[2*floor(x)-1]']
for i in range(1):
    current_objective = objectivt_list[i]
    f1 = current_objective[0]
    df1 = current_objective[1]
    f2 = current_objective[2]
    df2 = current_objective[3]
    fig, axs = plt.subplots(2,5,figsize=(16, 8))
    fig.suptitle(title_list[i])
    fig.tight_layout()
    for j in range(5):
        eta = E[j]
        f = f1
        df = df1
        g = f2
        dg = df2
        x = 0.1
        y = 0.5
        X = [x]
        Y = [y]
        Conserved=[]
        k=0
        K = 200/eta
        while k <= K:
            x = x+eta*df(y)
            y = y-eta*dg(x)
            X.append(x)
            Y.append(y)
            k += 1
            Conserved.append(f(x)+g(y)+0.5*eta*df(x)*dg(y))
        #Mystery_Constant.append((g(X[0])+f(Y[0])-g(X[1])-f(Y[1]))/(X[1]*Y[1]-X[0]*Y[0]))
        axs[0,j].scatter(X,Y,s=0.1)
        axs[0,j].set_title('Trail when η='+str(eta), loc='right')
        axs[1,j].plot(Conserved[:500])
        axs[1,j].set_title('Magical quantity', loc='right')
    plt.show()