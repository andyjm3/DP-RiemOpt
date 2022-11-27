"""
    Differntially private hyperbolic structured prediction
"""

import torch
import numpy as np
from sklearn.manifold import SpectralEmbedding
from structuredlearning import Loss, Alpha

import plotly.graph_objs as go
import plotly.offline as plotlyplot

import matplotlib.pyplot as plt



def poin2lor(embPoin):
    """Converts a set of poiints from Poincaré to Lorentz representation"""
    sqnorms = np.sum(embPoin ** 2, axis=1)[:, np.newaxis]
    embLor = np.hstack((1+sqnorms, 2*embPoin)) / (1-sqnorms)
    return embLor



def lor2poin(embPoin):
    return embPoin[:,1:] / (embPoin[:, 0, np.newaxis] + 1)



def lorentzGeodesic(Y0, Y1):
    """Returns the set of geodesic distances between two set of points Y0 and Y1
    :param Y0: A set of P0 points in H_n seen as a set of row vectors in R^(n+1)
    :param Y1:  A set of P1 points in H_n seen as a set of row vectors in R^(n+1)
    :return:
    """
    return np.arccosh(-lorentzDotThresholded(Y0,Y1))

def lorentzDotThresholded(Y0, Y1):
    """Computes the Lorentz Riemannian product, thresholding values in (-1, 1)
    to the closest limit point
    :param Y0: A set of P0 points in H_n seen as a set of row vectors in R^(n+1)
    :param Y1: A set of P1 points in H_n seen as a set of row vectors in R^(n+1)
    :return: A matrix G of size P0 x P1 where each element (G)_ij = <(Y0)_i, (Y1)_j>_L
    """
    G = np.dot(Y0[:,1:], Y1[:,1:].T) - Y0[:,0][:,np.newaxis]*Y1[:,0][:,np.newaxis]
    G[ (G > 0) & (G < 1) ] =  1
    G[ (G >-1) & (G <=0) ] = -1
    return G

def lorentzDot(Y0, Y1):
    """Computes the Lorentz Riemannian product
    :param Y0: A set of P0 points in H_n seen as a set of row vectors in R^(n+1)
    :param Y1: A set of P1 points in H_n seen as a set of row vectors in R^(n+1)
    :return: A matrix G of size P0 x P1 where each element (G)_ij = <(Y0)_i, (Y1)_j>_L
    """
    return np.dot(Y0[:,1:], Y1[:,1:].T) - Y0[:,0][:,np.newaxis]*Y1[:,0][:,np.newaxis]

def lorentzSqNorm(y):
    """Returns the Lorentz squared norm evalueted with the Lorentz Riemannian product
    :param y: A spoint in H_n seen as a set of row vectors in R^(n+1)
    :return: the Lorentz norm of y
    """
    return lorentzDotThresholded(y,y)

def lorentzNorm(y):
    """Returns the Lorentz norm evaluated with the Lorentz Riemannian product
    :param y: A spoint in H_n seen as a set of row vectors in R^(n+1)
    :return: the Lorentz norm of y
    """
    return np.sqrt(lorentzDotThresholded(y,y))

def steepestAscent(Ytr, Y):
    """Computes the steepest ascent direction of the squared geodesic loss for the Lorentz hyperbolic model.
    h = g_L^{-1} grad f(Y)
    Note: H_n = Lorentz model of hyperbolic manifold
    :param Y: A point in H_n seen as a row vector in R^(n+1)
    :param Ytr: A set of P points in H_n seen as a set of row vector in R^(n+1)
    :return: A matrix (n+1) x P, where each column is the steepest ascent direction w.r.t to Y for the
    squared geodesic loss in H_n: d(Y, (Ytr)_i)^2
    """
    inners = lorentzDotThresholded(Ytr, Y) # col vector
    coefficients =  -2 * np.arccosh(-inners) / np.sqrt(inners**2 - 1)
    coefficients[inners == -1] = -2
    # step_control = lorentzGeodesic(Ytr, Y)**2
    step_control = 1
    return (step_control * coefficients * Ytr).T

def tangentSpaceProj(v, p):
    """Computes the projection of vector v in the space tangent to p
    :param v: A set of V vectors in the tangent space of p seen as row vectors in R^{n+1}
    :param p: A point in H_n seen as a row vector in R^{n+1}
    :return: A matrix V x (n+1) where each row is the steepest ascent direction of (v)_i"""
    p_tile = np.tile(p, (v.shape[0],1))
    return v - (lorentzDot(v, p)/lorentzDot(p,p)) * p_tile

def expMap(v, p):
    """Evaluates the exponential map of a vector v in the space tangent to p on the Lorentz model of
    hyperbolic manifold
    :param v: A row vector on the tangent space of p
    :param p: A point in H_n seen as a row vector in R^(n+1)
    :return: The point closest to p with geodesics having as an acceleration vector v
    """
    v_norm = np.sqrt(lorentzDot(v,v))
    if v_norm == 0:
        return p
    else:
        return np.cosh(v_norm) * p + np.sinh(v_norm) * (v / v_norm)

def gradLorentz(Ytr, Y):
    """Returns the Riemannian gradient of the squared geodesic distance for the Lorentz model of hyperbolic
    manifold of a point y in H_n with respect to a set of points Ytr in H_n.
    Squared geodesic distance in Lorentz model: d(y0,y1)^2 = arcosh( -<y0,y1>_L )^2
    <y0,y1>_L is the Lorentz Riemannian scalar product.
    :param Y: A point in H_n seen as a row vector in R^(n+1)
    :param Ytr: A set of P points in H_n seen as a set of row vector in R^(n+1)
    :return: A matrix (n+1),where each row i is the steepest gradient w.r.t the i-th point in Ytr
    """
    H = steepestAscent(Ytr, Y).T # row indexes training points, col indexes dim
    return tangentSpaceProj(H, Y)




def loretzn_addnoise(base_points, sigma=0.01):
    """
        Sample at base points 
    """
    def lorentz_inp(a,b):
        prod = a*b
        return prod[1:].sum()-prod[0]

    num, k = base_points.shape 

    samples = np.random.normal(0,sigma,size=(num,k-1))
    zero_padded = np.hstack([np.zeros((num,1)),samples ])
    tg_samples = []
    for i in range(num):
        base_temp =  base_points[i]
        base_temp_1 = np.zeros_like(base_temp)
        base_temp_1[0] = 1 + base_temp[0]
        base_temp_1[1:] = base_temp[1:]
        sample = zero_padded[i] + (lorentz_inp(zero_padded[i],base_temp)/(1+base_temp[0]) * base_temp_1)
        tg_samples.append(sample)
    tg_samples = np.array(tg_samples)
    return tg_samples



def mAP(node, node_embed, graph, embed, embed_labels):
    neighs = [n for n in graph.neighbors(node)]
    D = lorentzGeodesic(embed, node_embed)
    embed_neighs =[embed_labels[idx] for idx in np.squeeze(np.argsort(D, axis=0)[:len(neighs)])] # labels of neighbours in the embedding

    accuracy = []
    for idx in range(1, len(neighs)+1):
        c = sum(neigh in embed_neighs[:idx] for neigh in neighs)
        accuracy.append(c/idx)












def plot_results():
    
    assert len(ctest) == 1
    assert len(ypred_dp.shape) == len(ypred_rgd.shape) == len(ytest.shape) == 2
    assert ytest.shape[0] == 1
    
    edge_trace = go.Scatter(
        x=[],
        y=[],
        #line=dict(width=0.5,color='#888'),
        line = dict(width=0.9,color='#76a0ad'), # light green
        hoverinfo='none',
        mode='lines',
        showlegend =False
        )
  
    xs = []
    ys = []
    for s0,s1 in edgelist:
        x0, y0 = vis[data.item2id[s0]]
        x1, y1 = vis[data.item2id[s1]]
  
        xs.extend(tuple([x0, x1, None]))
        ys.extend(tuple([y0, y1, None]))
  
    edge_trace['x'] = xs
    edge_trace['y'] = ys
    
    
    # plot train nodes 
    node_trace_train = go.Scatter(
        x= ytrain[:,0],
        y= ytrain[:,1],
        text=ctrain,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            reversescale=True,
            #color='#8b9dc3',
            line=dict(width=0.5,color='DarkSlateGrey'),
            color='#76a0ad',
            size=5),
        showlegend =False
        )
    
    # display the major categories
    display_list = [#'placental.n.01',
     'primate.n.02',
     'mammal.n.01',
     'carnivore.n.01',
     'canine.n.02',
     'dog.n.01',
     'pug.n.01',
     #'homo_erectus.n.01',
     'homo_sapiens.n.01',
     'terrier.n.01',
     'rodent.n.01',
     'ungulate.n.01',
     #'odd-toed_ungulate.n.01',
     'even-toed_ungulate.n.01',
     'monkey.n.01',
     'cow.n.01',
     'welsh_pony.n.01',
     'feline.n.01',
     'cheetah.n.01',
     'mouse.n.01']
    
    label_trace_train = go.Scatter(
        x=[],
        y=[],
        mode='text',
        text=[],
        textposition='top center',
        textfont=dict(
            family='times',
            size=22,
            color = "#000000"
        ),
        showlegend =False
    )
    
    for name in display_list:
        x,y = vis[data.item2id[name]]
        label_trace_train['x'] += tuple([x])
        label_trace_train['y'] += tuple([y])
        label_trace_train['text'] += tuple([name.split('.')[0]])
        
        
    
    # plot test nodes
    node_trace_test = go.Scatter(
        x= ytest[:,0],
        y= ytest[:,1],
        text=ctest,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            reversescale=True,
            #color='#8b9dc3',
            color='#76a0ad',
            line=dict(width=0.5,color='DarkSlateGrey'),
            size=18),
        showlegend =False
        )
    
    display_list = [
     'workhorse.n.02',
     'carthorse.n.01',
     'horse.n.01',
     'dark_horse.n.02',
     'warhorse.n.03',
     'dun.n.01'
    ]
    
    label_trace_test = go.Scatter(
        x=[],
        y=[],
        mode='text',
        text=[],
        textposition='top center',
        textfont=dict(
            family='times',
            size=22,
            color = "#000000"
        ),
        showlegend =False
    )
    
    for name in display_list:
        x,y = vis[data.item2id[name]]
        label_trace_test['x'] += tuple([x])
        label_trace_test['y'] += tuple([y])
        label_trace_test['text'] += tuple([name.split('.')[0]])
    
    
    
    # plot y_pred_rgd (plot the last one)
    node_ypred_rgd = go.Scatter(
        x=ypred_rgd[-2:-1, 0],
        y=ypred_rgd[-2:-1, 1],
        mode='markers',
        marker=dict(showscale=False,
        reversescale=True, symbol='diamond', color='#7685ad',
        line=dict(width=0.5,color='DarkSlateGrey'),
        size=18),
        name = 'RGD'
    )
    
    
    # plot y_pred_dp (plot many)
    plot_num = 3
    permute_idx = np.random.permutation(ypred_dp.shape[0])    
    
    node_ypred_dp = go.Scatter(
        x=ypred_dp[permute_idx[:plot_num], 0],
        y=ypred_dp[permute_idx[:plot_num], 1],
        mode='markers',
        marker=dict(showscale=False,
        reversescale=True, symbol='star', color='#ad7685',
        line=dict(width=0.5,color='DarkSlateGrey'),
        size=18),
        name = 'DP-RGD'
    )
    
            
    
    
    
    fig = go.Figure(data=[edge_trace, 
                          node_trace_train,
                          label_trace_train, 
                          node_trace_test, 
                          label_trace_test,
                          node_ypred_rgd,
                          node_ypred_dp],
                 layout=go.Layout(
                    width=700,
                    height=700,
                    #showlegend=False,
                    legend=dict(
                        yanchor="top",
                        y=0.98,
                        xanchor="left",
                        x=0.8,
                        bgcolor='rgba(255,255,255,0.4)',
                        font=dict(
                                    family="Times",
                                    size=24,
                                    color="black"
                                ),
                    ),
                    hovermode='closest',
                    margin=dict(b=0,l=0,r=0,t=0),
                    xaxis=dict(range=[0.25, 0.75],
                               showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(range=[-0.92, -0.6],
                               showgrid=False, zeroline=False, showticklabels=False)))
    
    
    
    
    
    fig.show()
    #fig.write_image("hsp_embed_full.pdf")
    fig.write_image("hsp_embed_zoom.pdf") # change range, display_test!!!


    #plotlyplot.plot(fig)





#%%

if __name__ == '__main__':
    
    np.random.seed(123)
    
    
    # load trained model and data
    model = torch.load('demo/model.pt')
    data = torch.load('demo/data.pt')
    
    with open('data/wordnet/mammal_hierarchy.tsv','r') as f:
        edgelist = [line.strip().split('\t') for line in f.readlines()]
    
    
    # get adj matrix
    N = data.n_items
    adj = np.zeros((N, N))
    adj += np.eye(N)
    for s0,s1 in edgelist: 
        s0idx = data.item2id[s0]
        s1idx = data.item2id[s1]
        adj[s0idx, s1idx] = 1.
        adj[s1idx, s0idx] = 1.
    
    # use adj as features
    #X = adj
    
    # use laplacian eigenmap as features
    embedding = SpectralEmbedding(n_components=3, affinity='precomputed')
    X = embedding.fit_transform(adj)
    
    # get the embedding 
    vis = model.embedding.weight.data.numpy()
    
    # get hyperbolic embeddings
    y = np.zeros((N, 2))
    names = [None] * N
    for name in data.items:
        idx = data.item2id[name]
        y[idx, :] = vis[idx]
        names[idx] = name.split('.')[0]
    
    sigmaKRLS = 0.2
    lambdaKRLS = 1E-5
    
    # convert to Lorentz model
    y = poin2lor(y)
    print("Embedding matrix shape:", y.shape)
    
    # Split training and test data
    testList = [data.item2id['horse.n.01']]

    xtest =[]; ytest = []; ctest =[]
    xtrain =[]; ytrain = []; ctrain=[]
    for idx, embPoint in enumerate(y):
        if idx in testList:
            xtest.append(X[idx])
            ytest.append(embPoint)
            ctest.append(names[idx])
        else:
            xtrain.append(X[idx])
            ytrain.append(embPoint)
            ctrain.append(names[idx])

    xtrain = np.array(xtrain)
    xtest  = np.array(xtest)
    ytrain = np.array(ytrain)
    ytest  = np.array(ytest)
    
    
    valIdx = np.random.choice(xtrain.shape[0], int(xtrain.shape[0]*0.2))
    X = {'xtr':xtrain, 'xval':xtrain[valIdx]}
    Y = {'ytr':ytrain, 'yval':ytrain[valIdx]}
    
    
    loss = Loss('squaredLorentzGeodesic')
    
    Alphax = Alpha(kernel='gausskernel')
    
    
    repeats = 20
    
    
    losses_rgd = []
    gradnorm_rgd = []
    ypred_rgd = []
    losses_dp = []
    gradnorm_dp = []
    ypred_dp = []
    for irep in range(repeats):
        
        assert xtest.shape[0] == 1 # make sure we only have 1 test point
        assert ytest.shape[0] == 1
    
        # set init 
        y0 = ytrain[np.random.randint(0, ytrain.shape[0])][np.newaxis,:]
        
        # non-private
        epochs = 1500
        lr = 0.003
        # Computes the estimator to be minimized
        alpha = Alphax.eval(X=xtrain, x=xtest, lam=lambdaKRLS, sigma=sigmaKRLS)
        print("Positive alphas: ", np.sum(alpha > 0), "/", alpha.size,
              "\t Average alpha magnitude:", np.mean(alpha),
              "\t Max alpha magnitude:", np.max(np.abs(alpha)))
        #y_hist = np.full((epochs, ytrain.shape[0]), np.inf)
        yt = y0
        eta = np.ones(epochs) * lr
        
        losses_epoch = []
        gradnorm_epoch = []
        for it in range(epochs):
            pointwise_grad = gradLorentz(ytrain, yt)
            loss_grad =np.dot(alpha, pointwise_grad)[np.newaxis,:]
            yt = expMap(-eta[it]*loss_grad, yt)
            
            loss_this_epoch = loss.eval(ytest, yt)
            losses_epoch.append(loss_this_epoch[0][0])
            gradnorm_this_epoch = np.linalg.norm(loss_grad)
            #gradnorm_this_epoch = lorentzNorm(loss_grad)
            gradnorm_epoch.append(gradnorm_this_epoch)
            if (it % 50 == 0) or (it == (epochs-1)):
                print("Iteration %d, loss = %f, gradnorm = %f" % (it,loss_this_epoch, gradnorm_this_epoch))
            #if np.linalg.norm(loss_grad) < 1E-4:
            #    break
        
        gradnorm_rgd.append(gradnorm_epoch)
        losses_rgd.append(losses_epoch)
        ypred_rgd.append(yt)
        
        
        # private
        alpha = Alphax.eval(X=xtrain, x=xtest, lam=lambdaKRLS, sigma=sigmaKRLS)
        print("Positive alphas: ", np.sum(alpha > 0), "/", alpha.size,
              "\t Average alpha magnitude:", np.mean(alpha),
              "\t Max alpha magnitude:", np.max(np.abs(alpha)))
        max_alpha = np.max(np.abs(alpha))
        
        eps = 0.1
        delta = 1e-3
        Dw = 10
        L0 = Dw * max_alpha
        L1 = Dw / np.tanh(Dw)
        n = xtrain.shape[0]
        T = np.sqrt(L1) * n * eps / (np.sqrt(2 * np.log(1/delta)) * L0)
        T = int(T)
        sigma = T*np.log(1/delta) * L0**2/(n**2 * eps**2)
        sigma = np.sqrt(sigma)
        lr = 0.001
            
        
        #y_hist = np.full((epochs, ytrain.shape[0]), np.inf)
        yt = y0
        eta = np.ones(T) * lr
        
        losses_epoch = []
        gradnorm_epoch = []
        for it in range(T):
            pointwise_grad = gradLorentz(ytrain, yt)
            loss_grad =np.dot(alpha, pointwise_grad)[np.newaxis,:]
            noise = loretzn_addnoise(yt, sigma = sigma)
            
            yt = expMap(-eta[it]*(loss_grad+noise), yt)
            
            loss_this_epoch = loss.eval(ytest, yt)
            losses_epoch.append(loss_this_epoch[0][0])
            gradnorm_this_epoch = np.linalg.norm(loss_grad)
            #gradnorm_this_epoch = lorentzNorm(loss_grad)
            gradnorm_epoch.append(gradnorm_this_epoch)
            if (it % 50 == 0) or (it == (T-1)):
                print("Iteration %d, loss = %f, gradnorm = %f" % (it,loss_this_epoch, gradnorm_this_epoch))
            if np.linalg.norm(loss_grad) < 1E-4:
                break
        
        gradnorm_dp.append(gradnorm_epoch)
        losses_dp.append(losses_epoch)
        ypred_dp.append(yt)
        
        
        
    # convert back to poincare model
    ytrain = lor2poin(ytrain)
    ytest = lor2poin(ytest)
    ypred_rgd = lor2poin(np.concatenate(ypred_rgd))
    ypred_dp = lor2poin(np.concatenate(ypred_dp))
    
    
    #%%
    ####### plot losses ########
    col = ['dodgerblue', "tab:orange", "mediumaquamarine"]
    marker = ['*', 'x', 'o', '--']
    
    losses_rgd = np.array(losses_rgd)
    losses_dp = np.array(losses_dp)
    
    mean_rgd = np.mean(losses_rgd,0)
    std_rgd = np.std(losses_rgd,0)
    mean_log_rgd = np.mean(np.log10(losses_rgd),0)
    std_log_rgd = np.std(np.log10(losses_rgd), 0)
    # 
    mean_dp = np.mean(losses_dp,0)
    std_dp = np.std(losses_dp,0)
    mean_log_dp = np.mean(np.log10(losses_dp),0)
    std_log_dp = np.std(np.log10(losses_dp), 0)
    
    scale = 1
    plt.rcParams["figure.figsize"] = (7,6)
    plt.figure()
    plt.yscale("log")
    
    # # rgd
    plt.plot(range(1,len(mean_log_rgd)+1), np.power(10,mean_log_rgd), label='RGD', color=col[0], marker=marker[0],markersize=1, linewidth=2.5)
    plt.fill_between(range(1,len(mean_log_rgd)+1), np.power(10,mean_log_rgd-scale*std_log_rgd) , np.power(10,mean_log_rgd+scale*std_log_rgd), alpha=0.3, fc=col[0])
    # # prgd
    plt.plot(range(1,len(mean_log_dp)+1), np.power(10,mean_log_dp), label='DP-RGD', color=col[1], marker=marker[1],markersize=1, linewidth=2.5)
    plt.fill_between(range(1,len(mean_log_dp)+1), np.power(10,mean_log_dp-scale*std_log_dp) , np.power(10,mean_log_dp+scale*std_log_dp), alpha=0.3, fc=col[1])
    # 
    plt.legend(loc='upper right', fontsize=21)
    plt.xlabel("Iteration", fontsize=25) 
    plt.xticks(fontsize=15)
    plt.ylabel("Distance to optimal", fontsize=25)
    plt.yticks(fontsize=15)
    
    plt.savefig('hsp_loss.pdf', bbox_inches='tight')
    
    
    
    ######### plot gradnorm #############
    losses_rgd = np.array(gradnorm_rgd)
    losses_dp = np.array(gradnorm_dp)
    
    mean_rgd = np.mean(losses_rgd,0)
    std_rgd = np.std(losses_rgd,0)
    mean_log_rgd = np.mean(np.log10(losses_rgd),0)
    std_log_rgd = np.std(np.log10(losses_rgd), 0)
    # 
    mean_dp = np.mean(losses_dp,0)
    std_dp = np.std(losses_dp,0)
    mean_log_dp = np.mean(np.log10(losses_dp),0)
    std_log_dp = np.std(np.log10(losses_dp), 0)
    
    scale = 1
    plt.rcParams["figure.figsize"] = (7,6)
    plt.figure()
    plt.yscale("log")
    
    # # rgd
    plt.plot(range(1,len(mean_log_rgd)+1), np.power(10,mean_log_rgd), label='RGD', color=col[0], marker=marker[0],markersize=1, linewidth=2.5)
    plt.fill_between(range(1,len(mean_log_rgd)+1), np.power(10,mean_log_rgd-scale*std_log_rgd) , np.power(10,mean_log_rgd+scale*std_log_rgd), alpha=0.3, fc=col[0])
    # # prgd
    plt.plot(range(1,len(mean_log_dp)+1), np.power(10,mean_log_dp), label='DP-RGD', color=col[1], marker=marker[1],markersize=1, linewidth=2.5)
    plt.fill_between(range(1,len(mean_log_dp)+1), np.power(10,mean_log_dp-scale*std_log_dp) , np.power(10,mean_log_dp+scale*std_log_dp), alpha=0.3, fc=col[1])
    # 
    plt.legend(loc='upper right', fontsize=21)
    plt.xlabel("Iteration", fontsize=25) 
    plt.xticks(fontsize=15)
    plt.ylabel("Gradient norm", fontsize=25)
    plt.yticks(fontsize=15)
    
    plt.savefig('hsp_gradnorm.pdf', bbox_inches='tight')
    
    plot_results()
