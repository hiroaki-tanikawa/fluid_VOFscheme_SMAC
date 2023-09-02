import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as animation
from matplotlib.animation import PillowWriter
import time

imax=51#x_index
jmax=51#y_index
dx=1/imax
dy=1/jmax

dt=dx**2/4
mu=0.01
g=9.8;

i_m=np.zeros([imax],dtype=int);
i_p=np.zeros([imax],dtype=int);
j_m=np.zeros([jmax],dtype=int);
j_p=np.zeros([jmax],dtype=int);

i=np.linspace(0,imax-1,imax,dtype=int);
i_m[1:imax]  =i[0:imax-1];i_m[0]=i[0]; #壁 勾配0
i_p[0:imax-1]=i[1:imax]  ;i_p[imax-1]=i[imax-1];#壁 勾配0

j=np.linspace(0,jmax-1,jmax,dtype=int);
j_m[1:jmax]  =j[0:jmax-1];j_m[0]=j[0];#壁 勾配0
j_p[0:jmax-1]=j[1:jmax]  ;j_p[jmax-1]=j[jmax-1];#壁 勾配0

x= np.zeros([jmax,imax]);
y= np.zeros([jmax,imax]);
for i_n in range(0,imax): 
 for j_n in range(0,jmax):
   x[j_n,i_n]=i_n#プロット用
   y[j_n,i_n]=j_n#プロット用


U= np.zeros([jmax,imax]);
V= np.zeros([jmax,imax]);
u= np.zeros([jmax,imax]);
v= np.zeros([jmax,imax]);
p= np.zeros([jmax,imax]);
lo= np.zeros([jmax,imax]);
dp= np.zeros([jmax,imax]);
dp_old= np.zeros([jmax,imax]);

u_p=np.zeros([jmax,imax]);
u_m=np.zeros([jmax,imax]);
v_p=np.zeros([jmax,imax]);
v_m=np.zeros([jmax,imax]);
F_x_p=np.zeros([jmax,imax]);
F_x_m=np.zeros([jmax,imax]);
F_y_p=np.zeros([jmax,imax]);
F_y_m=np.zeros([jmax,imax]);

F= np.zeros([jmax,imax]);
F_old= np.zeros([jmax,imax]);
dF=np.zeros([jmax,imax]);#F値変化量
FF=np.zeros([jmax,imax]);#計算範囲判定

DFU= np.zeros([jmax,imax]);
DFV= np.zeros([jmax,imax]);



def F_ini():#F値初期設定
  for i in range(imax):
   F[0:int(jmax/2),i]=1
  return F

def u_v_lo_p_ini(F):
  p[:,:]=1
  lo[:,:]=F[:,:]  
  #速度
  for j_n in range(jmax):
    for i_n in range(imax):
      U[j_n,i_n]=2*np.sin(i_n/imax*2*3.14)
      #V[j_n,i_n]=2*np.cos(j_n/jmax*2*3.14)
  #運動量
  u=np.multiply(lo,U)
  v=np.multiply(lo,V)

  #境界条件---------------
  v[0,:]=0#床
  u[:,0]=0#左壁
  u[:,-1]=0#右壁
  
  return u,v,p    


def F_cal(F,u,v):#F値計算
  F_old[:,:]=F[:,:]
 
  u_p=u[:,i_p]
  u_m=u[:,i_m]
  v_m=v[j_m,:]
  F_x_p=F_old[:,i_p]
  F_x_m=F_old[:,i_m]  
  F_y_p=F_old[j_p,:]
  F_y_m=F_old[j_m,:]  

  for I in range(imax):
     for J in range(jmax): 
        if F_old[J,I]>0 and F_old[J+1,I]<=0:##上界面--------------------  
           DFU_p=0.5*(u[J,I]+u_p[J,I])*dt*dy
           DFU_m=0.5*(u[J,I]+u_m[J,I])*dt*dy         
           DFV_m=0.5*(v[J,I]+v_m[J,I])*dt*dx
           dF[J,I]=(-DFU_p+DFU_m+DFV_m)/(dx*dy)
           if dF[J,I]>=0:
             F[J,I]=min(1,F_old[J,I]+dF[J,I]) #満タン=1
             if F[J,I]==1: F[J+1,I]=F_old[J,I]+dF[J,I]-1#下セルの余りを上セルに入れる
           else: 
             F[J,I]=max(0,F_old[J,I]+dF[J,I]) #なし=0
             if F[J,I]==0: F[J-1,I]=F_old[J,I]+dF[J,I]+1#上の借金を下セルに入れる

         
        elif F_old[J,I]>0 and F_old[J,I-1]<=0:##左界面(上に界面なし)--------------------  
           DFU_p=0.5*(u[J,I]+u_p[J,I])*dt*dy    
           DFV_p=0.5*(v[J,I]+v_p[J,I])*dt*dx             
           DFV_m=0.5*(v[J,I]+v_m[J,I])*dt*dx
           dF[J,I]=(-DFU_p-DFV_p+DFV_m)/(dx*dy)
           if dF[J,I]>=0: 
             F[J,I]=min(1,F_old[J,I]+dF[J,I]) #満タン=1
             if F[J,I]==1: F[J,I-1]=F_old[J,I]+dF[J,I]-1#右セルの余りを左セルに入れる
           else: 
             F[J,I]=max(0,F_old[J,I]+dF[J,I]) #なし=0
             if F[J,I]==0: F[J,I+1]=F_old[J,I]+dF[J,I]+1#左の借金を右セルに入れる


        elif F_old[J,I]>0 and F_old[J,I-1]<=0:##右界面(上に界面なし)--------------------  
           DFU_m=0.5*(u[J,I]+u_p[J,I])*dt*dy    
           DFV_p=0.5*(v[J,I]+v_p[J,I])*dt*dx             
           DFV_m=0.5*(v[J,I]+v_m[J,I])*dt*dx
           dF[J,I]=(-DFU_p-DFV_p+DFV_m)/(dx*dy)
           if dF[J,I]>=0: 
             F[J,I]=min(1,F_old[J,I]+dF[J,I]) #満タン=1
             if F[J,I]==1: F[J,I+1]=F_old[J,I]+dF[J,I]-1#左セルの余りを右セルに入れる
           else: 
             F[J,I]=max(0,F_old[J,I]+dF[J,I]) #なし=0
             if F[J,I]==0: F[J,I-1]=F_old[J,I]+dF[J,I]+1#右の借金を左セルに入れる           




  #print(I,J,F[24,:])  
  for I in range(imax):
     for J in range(jmax):     
        if F[J,I]>1:
          F[J,I]=1

  return F

F=F_ini()
u,v,p=u_v_lo_p_ini(F)
F_cal(F,u,v)
#breakpoint()
  
def predictor(u,v,p):
 u_pre= u-dt*(p[:,i_p]-p[:,i_m])/(2*dx)\
          +mu*dt*((u[:,i_p]-2*u[:,i]+u[:,i_m])/(dx**2)+(u[j_p,:]-2*u[j,:]+u[j_m,:])/(dy**2))\
       -(np.multiply(u[:,i], (u[:,i_p]-u[:,i_m])/(2*dx))-np.multiply(v[:,i],(u[:,i_p]-u[:,i_m])/(2*dy)))*dt

 v_pre= v-dt*(p[j_p,:]-p[j_m,:])/(2*dy)-dt*g\
          +mu*dt*((v[:,i_p]-2*v[:,i]+v[:,i_m])/(dx**2)+(v[j_p,:]-2*v[:,i]+v[j_m,:])/(dy**2))\
       -(np.multiply(u[j,:], (v[j_p,:]-v[j_m,:])/(2*dx))-np.multiply(v[j,:],(v[j_p,:]-v[j_m,:])/(2*dy)))*dt
 return u_pre,v_pre

def renzoku_siki_gosa(u_pre,v_pre):
 Dx= (u_pre[:,i_p]-u_pre[:,i_m])/(2*dx)
 Dy= (v_pre[j_p,:]-v_pre[j_m,:])/(2*dy)
 D =Dx+Dy
 return D

def pressure_corect(D,dp):
 for tt in range(100):
     dp_old[:,:]=dp[:,:]      
     dp[:,:]=(dp_old[:,i_m]+dp_old[:,i_p]\
             +dp_old[j_m,:]+dp_old[j_p,:])/4-D[:,:]*dx**2/4 #dx=dy
     #print(dp[10,10]-dp_old[10,10])    
 return dp

def corrector(u_pre,v_pre,dp): 
 u= u_pre-dt*(dp[:,i_p]-dp[:,i_m])/(2*dx)
 v= v_pre-dt*(dp[j_p,:]-dp[j_m,:])/(2*dy)
 return u,v


#グラフ描画---------------------------------------------------------------------------
fig,ax = plt.subplots(figsize=(5,5),dpi=120)
ims=[]#アニメーション用
ax.set_xlim(-2,52)
ax.set_ylim(-2,52)

tend=10000
F=F_ini()
u,v,p=u_v_lo_p_ini(F)
for t in range(tend):#時間発展---------------------------------------------------------
  F=F_cal(F,u,v)
  u,v=predictor(u,v,p)
  u_pre,v_pre=u,v
  D=renzoku_siki_gosa(u_pre,v_pre) 
  dp=pressure_corect(D,dp)
  u,v=corrector(u_pre,v_pre,dp)
  p[:,:]=p[:,:]+dp[:,:]

  #境界条件---------------
  v[0,:]=0#床
  u[:,0]=0#左壁
  u[:,-1]=0#右壁

  #流体がない領域は運動量0"""
  for J in range(jmax):
    for I in range(jmax):
     if F[J,I]<=0:
       u[J,I]=0
       v[J,I]=0
  #"""
  

  """
  #VOF_total確認----------
  summ=0
  for ii in range(imax):
    for jj in range(jmax):
      summ=summ+F[jj,ii]
  print(summ)
  """

  if np.mod(t,100)==0:
    vector=ax.quiver(x[::2,::2],y[::2,::2],u[::2,::2],v[::2,::2],color="blue", scale = 40)
    ims.append([vector])#アニメーション用  

  #plt.draw()  
  #plt.pause(0.0001)
  #if t!=tend-1:ax.clear()
    
ani = animation.ArtistAnimation(fig, ims)
ani.save('fluid_vof.gif',writer="pillow",dpi=150,fps=5)    
#plt.show()    

