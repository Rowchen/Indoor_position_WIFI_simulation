'''
Author:Chen Yaoxin
Southeast University
'''

import numpy as np
import matplotlib.pyplot as plt
from WIFI_RSS_model import *
from postion_model import *


model1=WIFI_net(ap_coor_list=[[300,300],[1700,1700],[300,1700],[1700,300]],constA=10,constn=5.5,room_size=[2000,2000],height=300)
model1.generate_rss(grid_size=50,n_sample=10)
model1.generate_trajectory()
plt.subplot(2,2,1)
plt.title("origin")
plt.plot(model1.trajectory_x,model1.trajectory_y,linewidth ='3')


#HMM
hmm=HMM()
hmm.fit(model1.RSS_proba,model1.RSS_coor)
print hmm.cal_proba(model1.RSS[0,0,:]).shape
hmm_pred=hmm.predict(model1.rss_track)
hmm_pred_coor=[]
for i in hmm_pred:
    hmm_pred_coor.append(model1.RSS_coor[i[0],i[1]])
hmm_pred_coor.reverse() 
hmm_pred_coor=np.array(hmm_pred_coor)

plt.subplot(2,2,2)
plt.title("HMM")
plt.plot(model1.trajectory_x,model1.trajectory_y,color='r',linewidth ='3') 
plt.plot(hmm_pred_coor[:,0],hmm_pred_coor[:,1],color='b') 
error=np.sqrt((hmm_pred_coor[:-1,0]-model1.trajectory_x)**2+(hmm_pred_coor[:-1,1]-model1.trajectory_y)**2)
print "HMM"
print 'mean error is:',np.mean(error),'cm'
print 'max error is:',error.max(),'cm'



#least_square
pred_traj_ls=least_square(model1,model1.rss_track,500) 
plt.subplot(2,2,3)
plt.title("least_square")
plt.plot(model1.trajectory_x,model1.trajectory_y,color='r',linewidth ='3') 
plt.plot(pred_traj_ls[:,0],pred_traj_ls[:,1],color='b') 
error=np.sqrt((pred_traj_ls[:,0]-model1.trajectory_x)**2+(pred_traj_ls[:,1]-model1.trajectory_y)**2)
print "least_square"
print 'mean error is:',np.mean(error),'cm'
print 'max error is:',error.max(),'cm'



#knn
s=knn(8)
s.fit(model1.RSS2,model1.RSS_coor2) 
pred_traj=s.predict(model1.rss_track)  
plt.subplot(2,2,4)
plt.title("KNN")
plt.plot(model1.trajectory_x,model1.trajectory_y,color='r',linewidth ='3') 
plt.plot(pred_traj[:,0],pred_traj[:,1],color='b') 
plt.show()
error=np.sqrt((pred_traj[:,0]-model1.trajectory_x)**2+(pred_traj[:,1]-model1.trajectory_y)**2)
print "KNN"
print 'mean error is:',np.mean(error),'cm'
print 'max error is:',error.max(),'cm'

