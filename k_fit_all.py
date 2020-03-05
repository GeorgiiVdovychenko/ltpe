import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error

def MAE(y, pred):
    return np.mean(abs(y-pred))


def k1(T, c1, c2, n):
    return c1 * c2 * (T**n) / (c2 + c1 * (T**n))

def k2(T, c3, c4):
    return c3*np.exp(-1 * c4 / T)

def f_k(T, c1, c2, n, c3, c4):
    return k1(T, c1, c2, n) + k2(T, c3, c4)

def open_file(path):
    with open(path, 'r') as f:
        data_raw = f.readlines()
    data_raw = [d.replace(' ', '\t').replace('\n', '') for d in data_raw]
    data = [d.split('\t') for d in data_raw]
    data = pd.DataFrame(data[1:], columns=['T', 'k'])
    return data

def preprocess(data):
    for c in data.columns:
        data[c] = data[c].astype('float')
    data['T_log'] = np.log(data['T'])
    data['k_log'] = np.log(data['k'])
    return data

def ploynomal_fit(data):
    z = np.polyfit(data['T_log'], data['k_log'], 7)
    p = np.poly1d(z)
    return p

def interpolate(p, min, max):
    xp = np.linspace(min, max, 200)
    ap_df = pd.DataFrame({'x': xp})
    ap_df['y'] = [p(x) for x in xp]
    return ap_df

def find_bend_points(p):
    p3 = np.polyder(p, m=3)
    roots = np.roots(p3)
    return sorted([x for x in roots if (np.exp(x)>2) and (np.exp(x)<=70)])

def refine_palteue(interp_df, bent_pts):
    plateue = interp_df[(interp_df['x'] > bent_pts[0]) & (interp_df['x'] < bent_pts[1])]
    plateue['ae'] = plateue['y'].apply(lambda x: plateue['y'].mean() - x)
    var = np.var(plateue['ae'])
    plateue = plateue[plateue['ae'].abs() < 3*var]
    return plateue[['x', 'y']].apply(lambda x: np.exp(x))


def approximate(data):
    data = preprocess(data)
    polynom = ploynomal_fit(data)

    interp_df_log = interpolate(polynom, data['T_log'].min(), np.log(120))#data['T_log'].max())

    T = np.exp(interp_df_log['x'])
    k = np.exp(interp_df_log['y'])
    k_max = k.max()
    k = k / k_max

    perc_25 = int(len(k)*0.25)
    perc_50 = int(len(k)*0.5)
    perc_75 = int(len(k)*0.75)

    def loss_k(*params):
        c1 = params[0][0]
        c2 = params[0][1]
        n = params[0][2]
        c3 = params[0][3]
        c4 = params[0][4]
        k_fit = f_k(T, c1, c2, n, c3, c4)

        return 1.5*MAE(k[:perc_25], k_fit[:perc_25])+\
               1.2*MAE(k[perc_25:perc_50], k_fit[perc_25:perc_50])+\
               MAE(k[perc_50:perc_75], k_fit[perc_50:perc_75])+\
               MAE(k[perc_75:], k_fit[perc_75:])

    params = minimize(loss_k, (0.5, 0.2, 1, 2, 15),
                         method='SLSQP',
                         tol=1e-10,
                         options={'maxiter': 10000}
                         )
    c1, c2, n, c3, c4 = tuple(params.x.tolist())
    return c1, c2, n, c3, c4, k_max

if __name__ == "__main__":
     p = '/home/quantum/Documents/iltpe/Cu10Zn2Sb4S13.dat'
     p = '/home/quantum/Documents/iltpe/PMMA.dat'
    # p = '/home/quantum/Documents/iltpe/SiO2Damon1973.txt'
     data = open_file(p)
     T = data['T'].astype('float')
     c1, c2, n, c3, c4 = approximate(data)
     print('C1 {}, C2 {}, N {}, C3 {}, C4 {}'.format(c1, c2, n, c3, c4))

     k_fit = k1(T, c1, c2, n) + k2(T, c3, c4)

     plt.scatter(data['T'], data['k'], color='g')
     plt.plot(T, k_fit)
     plt.yscale('log')
     plt.xscale('log')