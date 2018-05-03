import numpy as np
import math as math

#constants
time_step=1
sigma_w=1
sigma_r=1
cov_sunsensor=np.zeros(3)

# time t=k declarations
state_k0=np.zeros((1,7))
cov_k0=np.zeros(6)
w_k0=np.zeros((1,3))
error_k0=np.zeros((1,6))

# time t=k+1 declarations
state_k1=np.zeros((1,7))
cov_k1=np.zeros(6)
w_k1=np.zeros((1,3))
error_k1=np.zeros((1,6))

#measured quantities
w_m_k1=np.zeros((1,3))
z_sunsensor=np.zeros((1,3))

#vectors from models
z_sunsensor_mdl=np.zeros((1,3)) #in orbit frame

def quatRotate(q,x):
	
	#rotates vecctor x by quaternion q
	#M = np.array([[q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2,2*]])
	qi = quatInv(q)
	y = np.hstack(([0.],x.copy()))
	y = quatMultiply(q,y)
	y = quatMultiply(y,qi)

	x2 = y[1:4]
	return x2

def quatMultiply(q1,q2):

	#quaternion is scalar, vector. function multiplies 2 quaternions

	a1 = q1[0:1].copy()
	a2 = q2[0:1].copy()
	
	b1 = (q1[1:4].copy())
	b2 = (q2[1:4].copy())
	
	a = a1*a2 - np.dot(b1,b2)
	b = a1*b2 + a2*b1 + np.cross(b1,b2)

	#b = b.reshape((1,3))

	q = np.hstack((a,b))
	q = q/np.linalg.norm(q)
	return q

def quat_propogation(q_initial,w_0,w_1):
	w_avg=(w_0.copy()+w_1.copy())/2
	w_avg_norm=np.linalg.norm(w_avg)
	del_theta=(w_avg_norm*time_step)/2 #assuming w units is rad/sec
	q_w_delt_scalar=math.cos(del_theta)    #theta input to cos has to be in radians
	q_w_delt_vector=(w_avg*math.sin(del_theta))/w_avg_norm
	q_w_delt=np.hstack((q_w_delt_scalar,q_w_delt_vector))
	qw_n_n1=np.hstack(0,np.cross(w_0.copy(),w_1.copy()))
	q_propogate=np.array(q_w_delt)+np.array((((time_step*time_step)/24)*qw_n_n1)) #np.array needed??
	q_final=quatMultiply(q_initial.copy(),q_propogate)
	return q_final/np.linalg.norm(q_final)

def cross_matrix(w):
	cross_w_matrix=np.zeros(3)
	w_x=w[0:1].copy()
	w_y=w[1:2].copy()
	w_z=w[2:3].copy()
	cross_w_matrix[1,2]=(-w_z)
	cross_w_matrix[1,3]=w_y
	cross_w_matrix[2,1]=w_z
	cross_w_matrix[2,3]=(-w_x)
	cross_w_matrix[3,1]=(-w_y)
	cross_w_matrix[3,2]=w_x
	return cross_w_matrix	

def calculate_phi(w_0):
	w_0_norm=np.linalg.norm(w_0.copy())
	del_theta=(w_0_norm*time_step)
	cross_w_0=cross_matrix(w_0.copy())
	theta_1=math.cos(del_theta)*np.identity(3)
	theta_2=(math.sin(del_theta)/w_0_norm)*cross_w_0
	theta_3=((1-math.cos(del_theta))/(w_0_norm*w_0_norm))*np.matmul(np.transpose(w_0.copy()),w_0.copy())
	theta=theta_1+theta_2+theta_3
	psi_1=-np.identity(3)*time_step
	psi_2=-((1-math.cos(del_theta))/np.power(w_0_norm,2))*cross_w_0
	psi_3=-((del_theta-math.sin(del_theta))/np.power(w_0_norm,3))*np.matmul(cross_w_0,cross_w_0) #np.dot instead of np.matmul
	psi=psi_1+psi_2+psi_3
	phi=np.block([theta,psi],[np.zeros(3),np.identity(3)])
	return phi

def calculate_Q_d(w_0):
	w_0_norm=np.linalg.norm(w_0.copy())
	del_theta=(w_0_norm*time_step)
	cross_w_0=cross_matrix(w_0.copy())
	Q_11_1=np.power(sigma_r,2)*time_step*np.identity(3) 
	Q_11_22_numrtr= (np.power(del_theta,3)/3) + (2*math.sin(del_theta)) - (2*del_theta)
	Q_11_2=((np.power(time_step,3)/3)*np.identity(3)) +(Q_11_22_numrtr/np.power(w_0_norm,5))*np.matmul(cross_w_0,cross_w_0) #np.dot
	Q_11= Q_11_1 + (np.power(sigma_w,2)*Q_11_2)
	Q_12_1=(np.identity(3)*np.power(time_step,2))/2
	Q_12_2=((del_theta-math.sin(del_theta))/np.power(w_0_norm,3))*cross_w_0
	Q_12_3_numrtr=(np.power(del_theta,2)/2)+ math.cos(del_theta)- 1
	Q_12_3= (Q_12_3_numrtr/np.power(w_0_norm,4))*np.matmul(cross_w_0,cross_w_0)
	Q_12=-np.power(sigma_w,2)*(Q_12_1+Q_12_2+Q_12_3)
	Q_22= np.power(sigma_w,2)*time_step*np.identity(3)
	Q=np.block([Q_11,Q_12],[np.transpose(Q_12),Q_22])
	return Q

def calculate_H(z_mdl,q):
	temp=quatRotate(q.copy(),z_mdl.copy())
	cross_temp= cross_matrix(temp)
	H=np.block([-cross_temp,zeros(3)])
	return H

#propogation
state_k1[4:7]=state_k0[4:7] #b
w_k1=w_m_k1 - state_k1[4:7]  #w k+1 given k
state_k1[0:4]=quat_propogation(state_k0[0:4],w_k0,w_k1) #q 
phi=calculate_phi(w_k0)
Q_d=calculate_Q_d(w_k0)
cov_k1=np.matmul(np.matmul(phi,cov_k0),np.transpose(phi)) + Q_d

#Update
r=z_sunsensor-quatRotate(state_k1[0:4],z_sunsensor_mdl)
H=calculate_H(z_sunsensor_mdl,state_k1[0:4])
cov_r= np.matmul(np.matmul(H,cov_k1),np.transpose(H)) + cov_sunsensor
Kalman_gain= np.matmul(np.matmul(cov_k1,np.transpose(H)), np.linalg.inv(cov_r))
state_correction= np.matmul(r,np.transpose(Kalman_gain)) #use np.dot...
q_correction=np.hstack(1,(state_correction[0:3]/2))   #careful! quaternion convention is scalar, vector; state correction has 6 elts not 7
q_correction_norm=np.linalg.norm(q_correction)
q_correction=q_correction/q_correction_norm
state_k0[0:4]= quatMultiply(q_correction,state_k1[0:4]) #q update
state_k0[4:7]= state_k1[4:7]+ state_correction[3:6] #b update
w_k0= w_m_k1 - state_k0[4:7] #w update w k+1 given k+1
factor= np.identity(6)- np.matmult(Kalman_gain,H)
cov_k0= np.matmult(np.matmult(factor,cov_k1),np.transpose(factor)) + (np.matmult(np.matmult(Kalman_gain,cov_sunsensor),np.transpose(Kalman_gain)))
