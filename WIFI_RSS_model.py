'''
Author:Chen Yaoxin
Southeast University
'''

import numpy as np
class WIFI_net():
    def __init__(self,ap_coor_list,constA,constn,room_size,height):
        #list of coordinate of ap,etc,[[1,2],[3,4]]
        self.ap_coor_list=np.array(ap_coor_list)
        #propagation parameter
        self.constA=constA
        self.constn=constn
        self.room_size=np.array(room_size)#centimeter
        self.ap_num=self.ap_coor_list.shape[0]
        self.height=height#centimeter
        np.random.seed(2018)
        
    def cal_rss(self,dist):#dist is centimeter ,calculate the rss given dist
        rss=np.floor(-self.constA-10*self.constn*np.log10(dist/100.0))+np.random.randint(-6,7)#add random_vary
        return rss
    
    def cal_dist(self,rss):#calculate the dist given rss
        dist=10**(-(rss+self.constA)/10.0/self.constn)*100
        return dist
    
    def cal_rss_vec(self,coor):#calculate the rss vector  given coordinate
        rss_vec=np.zeros(self.ap_num,dtype=int)
        for ap in xrange(self.ap_num):
            distance=np.sqrt((coor[0]-self.ap_coor_list[ap][0])**2+(coor[1]-self.ap_coor_list[ap][1])**2+self.height**2)
            rss_vec[ap]=self.cal_rss(distance)
        return rss_vec
    
    def query_rss_coor(self,coor):#query the rss in the database given coordinate
        return self.RSS[coor[0]/self.grid_size,coor[1]/self.grid_size,:]
    
    def generate_rss(self,grid_size,n_sample=1):#generate the rss database given gridsize
        self.grid_size=grid_size
        x_num=self.room_size[0]/grid_size
        y_num=self.room_size[1]/grid_size
        self.RSS=np.zeros((x_num,y_num,self.ap_num),dtype=int)
        self.RSS2=np.zeros((x_num*y_num,self.ap_num),dtype=int)
        self.RSS_coor=np.zeros((x_num,y_num,2),dtype=int)
        self.RSS_coor2=np.zeros((x_num*y_num,2),dtype=int)
        self.RSS_proba=np.zeros((x_num,y_num,n_sample,self.ap_num),dtype=int)
        for x in xrange(x_num):
            coor_x=x*grid_size
            for y in xrange(y_num):
                coor_y=y*grid_size
                self.RSS_coor[x,y,:]=[coor_x,coor_y]
                self.RSS[x,y,:]=self.cal_rss_vec([coor_x,coor_y])
                
                self.RSS_coor2[x*y_num+y,:]=[coor_x,coor_y]
                self.RSS2[x*y_num+y,:]=self.cal_rss_vec([coor_x,coor_y])
                
                for n in xrange(n_sample):
                    self.RSS_proba[x,y,n,:]=self.cal_rss_vec([coor_x,coor_y])

    def generate_trajectory(self):#random ly generate a trajectory
        self.trajectory_x,self.trajectory_y,self.rss_track=[],[],[]
        np.random.seed(2018)
        x,y=0,0
        while (x<self.room_size[0])and(y>=0)and(y<self.room_size[1]):
            this_dir=np.random.randint(-3,4)
            step_random=np.random.randint(-10,10)
            step=20+step_random
            if this_dir>0:
                x+=step
            else:
                if x>self.room_size[0]/2:
                    y-=step
                else:
                    y+=step
            self.trajectory_x.append(x)
            self.trajectory_y.append(y)
            self.rss_track.append(self.cal_rss_vec([x,y]))
        self.rss_track=np.array(self.rss_track)
         
        
    def dist(self,a,b):
        return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+self.height**2)