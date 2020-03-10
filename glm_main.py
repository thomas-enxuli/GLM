import numpy as np
import pandas as pd
import tqdm
from data_utils import load_dataset
import matplotlib.pyplot as plt
from sklearn import neighbors
import time

__author__ = 'En Xu Li (Thomas)'
__date__ = 'March 7, 2020'

def _RMSE(x, y):
    return np.sqrt(np.average((x-y)**2))

def _pt_to_feature(x):
    omega = 2*np.pi/0.0569
    #print(x.shape)
    return np.array([1,x,x**2,x**3,np.sin(omega*x),np.cos(omega*x)])

def _construct_phi(x_train):
    M = 6 # number of basis functions
    phi = np.empty((0,M))
    for i in range(x_train.shape[0]):
        phi = np.append(phi,[_pt_to_feature(x_train[i])],axis=0)
    #print(phi.shape)

    return phi.astype(np.float64)

def _test_predict(l=0):
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
    x_total = np.vstack([x_train, x_valid])
    y_total = np.vstack([y_train, y_valid])

    phi_train = _construct_phi(x_total)
    phi_test = _construct_phi(x_test)

    U, S, Vh = np.linalg.svd(phi_train)

    # Invert Sigma
    sig = np.diag(S)
    filler = np.zeros([phi_train.shape[0]-len(S), len(S)])
    sig = np.vstack([sig, filler])

    inv = np.linalg.inv(sig.T @ sig + l*np.eye(sig.shape[1]))
    w = Vh.T @ inv @ sig.T @ (U.T @ y_total)

    prediction = phi_test @ w
    plot(xlabel='x',ylabel='y',name='mauna_loa_predict',x=x_test,y=[prediction,y_test],legend=['Predicted','GroundTruth'])
    return _RMSE(prediction,y_test)

def run_Q2(lambda_list=[]):
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
    phi_train = _construct_phi(x_train)
    phi_validation = _construct_phi(x_valid)
    U, S, Vh = np.linalg.svd(phi_train)

    # Invert Sigma
    sig = np.diag(S)
    filler = np.zeros([phi_train.shape[0]-len(S), len(S)])
    sig = np.vstack([sig, filler])
    valid_rmse = []

    for l in lambda_list:

        inv = np.linalg.inv(sig.T @ sig + l*np.eye(sig.shape[1]))
        w = Vh.T @ inv @ sig.T @ (U.T @ y_train)

        prediction = phi_validation @ w
        valid_rmse.append(_RMSE(prediction,y_valid))

    print(valid_rmse)
    print('lambda = '+str(lambda_list[np.argmin(valid_rmse)]))
    return _test_predict(l=lambda_list[np.argmin(valid_rmse)])

def _Q3_construct_K(x_train):
    K = np.zeros([x_train.shape[0],x_train.shape[0]])
    kernel_dict = {}

    for i in range(x_train.shape[0]):
        for j in range(x_train.shape[0]):
            if hash((i,j)) not in kernel_dict:
                temp = np.dot(_pt_to_feature(x_train[i]),_pt_to_feature(x_train[j]))
                kernel_dict[hash((i,j))] = temp
                kernel_dict[hash((j,i))] = temp
                K[i,j] = temp
                K[j,i] = temp

    return K

def _Q3_construct_test_K(x_total,x_test):
    K = np.zeros([x_test.shape[0],x_total.shape[0]])

    for i in range(x_test.shape[0]):
        for j in range(x_total.shape[0]):
            temp = np.dot(_pt_to_feature(x_total[j]),_pt_to_feature(x_test[i]))
            K[i,j] = temp

    return K

def _visualize_kernel(x,z,name):
    plot_y = []
    for i in range(len(z)):
        plot_y += [np.dot(_pt_to_feature(x[i]),_pt_to_feature(z[i]))]
    plot(xlabel='z',ylabel='k',name=name,x=z,y=[plot_y],legend=[])
    return 1

def run_Q3(l=0.1):
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
    x_total = np.vstack([x_train, x_valid])
    y_total = np.vstack([y_train, y_valid])

    K = _Q3_construct_K(x_total)
    R = np.linalg.cholesky((K + l*np.eye(len(K))))
    #print(K)
    #print(K.shape)

    R_inv = np.linalg.inv(R)

    alpha = R_inv.T @ R_inv @ y_total
    K_test = _Q3_construct_test_K(x_total,x_test)
    prediction = K_test @ alpha
    plot(xlabel='x',ylabel='y',name='mauna_loa_predict_CH',x=x_test,y=[prediction,y_test],legend=['Predicted','GroundTruth'])
    z = np.linspace(-0.1, 0.1, 100)
    x = [0]*len(z)
    _visualize_kernel(x,z,'k(0,z)')
    z = np.linspace(-0.1+1, 0.1+1, 100)
    x = [1]*len(z)
    _visualize_kernel(x,z,'k(1,z+1)')
    return _RMSE(prediction,y_test)

def _Q4_construct_K(x_train,theta):
    K = np.zeros([x_train.shape[0],x_train.shape[0]])
    kernel_dict = {}

    for i in range(x_train.shape[0]):
        for j in range(x_train.shape[0]):
            if hash((i,j)) not in kernel_dict:
                temp = _GKernel(x_train[i],x_train[j],theta)
                kernel_dict[hash((i,j))] = temp
                kernel_dict[hash((j,i))] = temp
                K[i,j] = temp
                K[j,i] = temp

    return K

def _Q4_construct_test_K(x_total,x_test,theta):
    K = np.zeros([x_test.shape[0],x_total.shape[0]])

    for i in range(x_test.shape[0]):
        for j in range(x_total.shape[0]):
            temp = _GKernel(x_test[i],x_total[j],theta)
            K[i,j] = temp

    return K

def _cast_TF(x):
    """
    change bool type array to one hot encoding with 1 and 0
    Inputs:
        x: (bool type np.array)
    Outputs:
        numpy array with one hot encoding
    """
    return np.where(x==True,1,0)

def run_Q4():
    theta_list = [0.05,0.1,0.5,1,2]
    lambda_list = [0.001,0.01,0.1,1]
    #regression
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
    x_total = np.vstack([x_train, x_valid])
    y_total = np.vstack([y_train, y_valid])
    val_loss = {}

    for theta in theta_list:
        val_loss['theta = '+str(theta)] = []
        print('---- Processing Theta = '+str(theta)+' ----')
        for l in lambda_list:
            print('\t---- Processing Lambda = '+str(l)+' ----')
            K = _Q4_construct_K(x_train,theta)
            R = np.linalg.cholesky((K + l*np.eye(len(K))))
            #print(K)
            #print(K.shape)

            R_inv = np.linalg.inv(R)
            alpha = R_inv.T @ R_inv @ y_train

            K_val = _Q4_construct_test_K(x_train,x_valid,theta)
            val_prediction = K_val @ alpha
            val_loss['theta = '+str(theta)] += [_RMSE(val_prediction,y_valid)]
    df = pd.DataFrame(val_loss, index =lambda_list)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
    df.to_csv ('mau.csv')
    theta = 1
    l = 0.001
    K = _Q4_construct_K(x_total,theta)
    R = np.linalg.cholesky((K + l*np.eye(len(K))))
    R_inv = np.linalg.inv(R)
    alpha = R_inv.T @ R_inv @ y_total

    K_test = _Q4_construct_test_K(x_total,x_test,theta)
    test_prediction = K_test @ alpha
    print(_RMSE(test_prediction,y_test))


    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train=1000, d=2)
    x_total = np.vstack([x_train, x_valid])
    y_total = np.vstack([y_train, y_valid])
    print('rosenbrock')
    val_loss = {}

    for theta in theta_list:
        val_loss['theta = '+str(theta)] = []
        print('---- Processing Theta = '+str(theta)+' ----')
        for l in lambda_list:
            print('\t---- Processing Lambda = '+str(l)+' ----')
            K = _Q4_construct_K(x_train,theta)
            R = np.linalg.cholesky((K + l*np.eye(len(K))))
            #print(K)
            #print(K.shape)

            R_inv = np.linalg.inv(R)
            alpha = R_inv.T @ R_inv @ y_train

            K_val = _Q4_construct_test_K(x_train,x_valid,theta)
            val_prediction = K_val @ alpha
            val_loss['theta = '+str(theta)] += [_RMSE(val_prediction,y_valid)]
    df = pd.DataFrame(val_loss, index =lambda_list)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
    df.to_csv ('rose.csv')
    theta = 2
    l = 0.001
    K = _Q4_construct_K(x_total,theta)
    R = np.linalg.cholesky((K + l*np.eye(len(K))))
    R_inv = np.linalg.inv(R)
    alpha = R_inv.T @ R_inv @ y_total

    K_test = _Q4_construct_test_K(x_total,x_test,theta)
    test_prediction = K_test @ alpha
    print(_RMSE(test_prediction,y_test))

    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    x_total = np.vstack([x_train, x_valid])
    y_total = np.vstack([y_train, y_valid])
    if y_total.dtype==np.dtype('bool'):
        y_total = _cast_TF(y_total)
        y_train = _cast_TF(y_train)
        y_valid = _cast_TF(y_valid)
        y_test = _cast_TF(y_test)

    print('iris')
    val_acc = {}

    for theta in theta_list:
        val_acc['theta = '+str(theta)] = []
        print('---- Processing Theta = '+str(theta)+' ----')
        for l in lambda_list:
            print('\t---- Processing Lambda = '+str(l)+' ----')
            K = _Q4_construct_K(x_train,theta)
            R = np.linalg.cholesky((K + l*np.eye(len(K))))
            #print(K)
            #print(K.shape)

            R_inv = np.linalg.inv(R)
            alpha = R_inv.T @ R_inv @ y_train

            K_val = _Q4_construct_test_K(x_train,x_valid,theta)
            val_prediction = K_val @ alpha

            result = np.argmax(val_prediction,axis=1)

            gt = np.where(y_valid==True,1,0)
            gt = np.argmax(gt,axis=1)

            unique, counts = np.unique(result-gt, return_counts=True)
            correct = dict(zip(unique, counts))[0]

            acc = correct/y_valid.shape[0]

            val_acc['theta = '+str(theta)] += [acc]

    df = pd.DataFrame(val_acc, index =lambda_list)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
    df.to_csv ('iris.csv')
    theta = 1
    l = 0.001
    K = _Q4_construct_K(x_total,theta)
    R = np.linalg.cholesky((K + l*np.eye(len(K))))
    R_inv = np.linalg.inv(R)
    alpha = R_inv.T @ R_inv @ y_total

    K_test = _Q4_construct_test_K(x_total,x_test,theta)
    test_prediction = K_test @ alpha

    result = np.argmax(test_prediction,axis=1)

    gt = np.where(y_test==True,1,0)
    gt = np.argmax(gt,axis=1)

    unique, counts = np.unique(result-gt, return_counts=True)
    correct = dict(zip(unique, counts))[0]

    acc = correct/y_test.shape[0]



    print(acc)


    return 1

def _greedy_alg(x_train,y_train,theta):
    I_candidate = list(range(x_train.shape[0]))
    I_selected = []
    k = 0
    r = y_train
    last_r = y_train*2
    big_K = np.empty((x_train.shape[0],0))
    weights, final_train_loss,prev_w = 0, 0, 0

    while True:
        prev_MDL = _MDL(N=x_train.shape[0],k=k-1,l2=np.linalg.norm(last_r,ord=2)**2)
        cur_MDL = _MDL(N=x_train.shape[0],k=k,l2=np.linalg.norm(r,ord=2)**2)
        k += 1
        #print(cur_MDL)
        #print(prev_MDL)
        if cur_MDL > prev_MDL:
            break
        last_r, prev_w = r, weights
        cur_J, picked_i = 0, 0

        for i in I_candidate:
            phi = _Q5_construct_K(x_train,i,theta)
            #print(np.dot(phi,phi).shape)
            #print(np.dot(phi,r).shape)
            J = np.dot(phi,r)**2 /np.dot(phi,phi)
            if J > cur_J:
                cur_J = J
                picked_i = i
        I_selected.append(picked_i)
        I_candidate.remove(picked_i)
        #print(picked_i)

        big_K = np.append(big_K,_Q5_construct_K(x_train,picked_i,theta).reshape(-1,1),axis=1)

        U, S, Vh = np.linalg.svd(big_K)
        #print ('U')

        # Invert Sigma
        sig = np.diag(S)
        filler = np.zeros([big_K.shape[0]-len(S), len(S)])
        sig_inv = np.linalg.pinv(np.vstack([sig, filler]))

        # Compute weights
        weights = Vh.T @ (sig_inv @ (U.T @ y_train))
        # print('K = ')
        # print(big_K.shape)
        # print(big_K)
        # print('alpha = ')
        # print(alpha.shape)
        # print(alpha)
        # print('----------------------')
        r = y_train - (big_K @ weights)
        final_train_loss = prev_MDL
        #break
    print('k = '+str(len(I_selected)-1))
    print('Training Loss = '+str(prev_MDL))
    return I_selected[:-1],prev_w

def _GKernel(x,z,theta):
    return np.exp(-np.linalg.norm([x-z],ord=2)**2/theta)

def _MDL(N=1,k=1,l2=1):
    return (N/2*np.log(l2))+(k/2*np.log(N))

def _Q5_construct_K(x_train,picked_i,theta):
    K = []
    for i in range(x_train.shape[0]):
        K.append(_GKernel(x_train[i],x_train[picked_i],theta))
    return np.array(K)

def _test_kernel(basis=[],x_train=None,test_pt=None,theta=0.01):
    phi = []
    for i in basis:
        phi.append(_GKernel(x_train[i],test_pt,theta))
    return np.array(phi)

def run_Q5():
    theta_list, test_loss = [0.01,0.1,1.0], []
    #theta_list, test_loss = [1.0], []
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock',n_train=200,d=2)
    for theta in theta_list:
        print('----- Processing Theta = '+ str(theta) + '-----')
        I_selected,w = _greedy_alg(x_train,y_train,theta=theta)
        #print(I_selected)
        #print(w)
        loss_total = 0
        big_K = np.empty((0,len(I_selected)))

        for i in range(x_test.shape[0]):
            build_kernel = _test_kernel(basis=I_selected,x_train=x_train,test_pt=x_test[i],theta=theta)
            #print(build_kernel)
            #break
            big_K = np.append(big_K,[build_kernel],axis=0)
        #print(big_K)
        predicted_y = np.dot(big_K,w)
        #print(predicted_y)
        loss = _RMSE(predicted_y,y_test)
        #     loss_total += loss
        # l = loss_total/x_test.shape[0]
        # test_loss.append(l)
        #break
        print('Test Loss: '+str(loss))
    return loss

def plot(xlabel='',ylabel='',name='fig',x=None,y=None,legend=None):
    """
    plot and figures

    Inputs:
        xlabel: (str) label on x axis
        ylabel: (str) label on y axis
        name: (str) title of the figure
        x: (np.array) x data
        y: (list of np.array) list of y values to plot against x
        legend: (list of str) label on y values

    Outputs:
        None
    """
    fig = plt.figure()
    for i in range(len(y)):
        if legend: plt.plot(x,y[i],label=legend[i])
        else: plt.plot(x,y[i])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    fig.savefig(name+'.png')

if __name__ == '__main__':
    run_Q2()
    run_Q3()
    run_Q4()
    run_Q5()
