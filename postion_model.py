'''
Author:Chen Yaoxin
Southeast University
'''

import numpy as np
class HMM():
    def fit(self,X,y):#model.rss_proba,model.rss_coor
        self.u=np.mean(X,axis=2)
        self.sigma=np.std(X,axis=2)
        print self.u.shape,self.sigma.shape
        
    def cal_proba(self,RSS_vec):
        RSS_vec=np.array(RSS_vec).reshape(1,1,4)
        proba=np.product(np.exp(-(((RSS_vec-self.u)/self.sigma)**2)/2)/((2*np.pi)**0.5)/self.sigma,axis=2)
        return proba
    
    def predict(self,traj):
        traj=np.array(traj)
        observe_proba=np.zeros((traj.shape[0],self.u.shape[0],self.u.shape[1]))
        for i in xrange(traj.shape[0]):
            observe_proba[i,:,:]=self.cal_proba(traj[i])
        
        #trans_proba
        pi=np.ones((self.u.shape[0],self.u.shape[1]))
        pi/=float(self.u.shape[0]*self.u.shape[1])
        newpi=pi.copy()
        traceback=np.zeros((traj.shape[0],self.u.shape[0],self.u.shape[1],2),dtype=int)
        traj_long=traj.shape[0]
        for t in xrange(traj_long):#xrange(traj.shape[0]):
            #trans from [i1j1] to [i2,j2]
            for i2 in xrange(self.u.shape[0]):
                for j2 in xrange(self.u.shape[1]):
                    tmp=observe_proba[t,i2,j2]
                    maxls=0
                    mark=(0,0)
                    for i1 in xrange(i2-2,i2+2):
                        if i1<0 or i1>=self.u.shape[0]:
                            continue
                        for j1 in xrange(j2-2,j2+2):
                            if j1<0 or j1>=self.u.shape[1]:
                                continue
                            ls=tmp*pi[i1,j1]
                            if ls>maxls:
                                maxls=ls
                                mark=(i1,j1)
                    traceback[t,i2,j2,:]=list(mark)
                    newpi[i2,j2]=maxls
            pi=newpi.copy()
            pi/=pi.max()
        statemax=(np.argmax(np.max(pi,axis=1)),np.argmax(np.max(pi,axis=0)))
        print "state argmax",statemax
    
        #traceback
        preds=[[statemax[0],statemax[1]]]
        for i in xrange(traj_long-1,-1,-1):
            preds.append(traceback[i,preds[-1][0],preds[-1][1]])
        return np.array(preds)

def least_square(model,X,n_iter,lr=0.1):
    X=np.array(X).reshape(-1,model.ap_num)
    pred=np.zeros((X.shape[0],2))
    for i in xrange(X.shape[0]):
        rss2dist=np.zeros(X.shape[1])
        for j in xrange(X.shape[1]):
            rss2dist[j]=model.cal_dist(X[i,j])
            
        #minmize loss=(dist(c,ap1)-dist1)**2+(dist(c,ap2)-dist2)**2+(dist(c,ap3)-dist3)**2
        #using gradient decsent
        x=y=loss=ep=0
        lastloss=-10000
        d1x=np.zeros(model.ap_num)
        d1y=np.zeros(model.ap_num)
        d2=np.zeros(model.ap_num)
        d3=np.zeros(model.ap_num)
        while abs(lastloss-loss)>0.1 and ep<n_iter:
            lastloss=loss
            ep+=1
            loss=dx=dy=0
            for ap in xrange(model.ap_num):
                d1x[ap]=(x-model.ap_coor_list[ap][0])
                d1y[ap]=(y-model.ap_coor_list[ap][1])
                d2[ap]=(d1x[ap]**2)+(d1y[ap]**2)+(model.height**2)
                d3[ap]=d2[ap]**0.5-rss2dist[ap]
                loss+=0.5*d3[ap]**2
                ls=d3[ap]*(d2[ap]**-.5)
                dx+=ls*d1x[ap]
                dy+=ls*d1y[ap]
            x-=lr*dx
            y-=lr*dy
        pred[i]=[x,y]
    return pred
    
    
class knn():
    def __init__(self,n_neighbors=1):
        self.neb=n_neighbors
    
    def fit(self,X,Y):
        self.x=np.array(X)
        self.y=np.array(Y)
        
    def predict(self,X):
        X=np.array(X).reshape(-1,self.x.shape[1])
        pred=np.zeros((X.shape[0],self.y.shape[1]))
        for i in xrange(X.shape[0]):
            dist=np.sqrt(np.sum((X[i,:]-self.x)**2,axis=1))
            cc=np.argsort(dist)[:self.neb]
#             w=np.zeros(self.neb)
#             for j in xrange(self.neb):
#                 w[j]=(1/dist[cc[j]])
#             w/=sum(w)
#             w=w.reshape(-1,1)
            w=1.0/self.neb
            pred[i]=np.sum(w*self.y[cc,:],axis=0)
        return pred     