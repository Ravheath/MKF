import numpy as np
import math as math

# just included to avoid errors from division operations because initial values have got all zeros; comment this later
np.seterr(divide='ignore', invalid='ignore') 

#constants
time_step=1  #decide
sigma_w=0   #covariance from rate of change in gyro bias (n_w), set to zero
sigma_r=0   #covariance from gyro noise(n_r), set to zero
cov_sunsensor=np.zeros((3,3))  #from sensor noise (n_m), set to zero

# time t=k declarations
state_k0=np.zeros(7)  #initialise to initial quat and gyro bias(0?)
cov_k0=np.zeros((6,6)) #we need initial covariance of state_error vector(del_theta, del_b) (but how?)
w_k0=np.zeros(3) #initial angular rate
#error_k0=np.zeros(6) #this isnt used anywhere in the code

# time t=k+1 declarations #initial declarations dont matter
state_k1=np.zeros(7)
cov_k1=np.zeros((6,6))
w_k1=np.zeros(3)
#error_k1=np.zeros(6) #this isnt used anywhere in the code

#measured quantities
w_m_k1=np.zeros(3) # gyro rate measurements
z_sunsensor=np.zeros(3) #sensor measurements

#vectors from models
z_sunsensor_mdl=np.zeros(3) #in orbit frame

def quatRotate(q,x):
	
	#rotates vecctor x by quaternion q
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

	q = np.hstack((a,b))
	q = q/np.linalg.norm(q)
	return q

def quatInv(q):
	#to get inverse of a quaternion
	qi = np.hstack((q[0:1],-1*q[1:4]))
	
	return qi

def quat_propogation(q_initial,w_0,w_1):
	# propagates quaternion using first order integration
	w_avg=(w_0.copy()+w_1.copy())/2
	w_avg_norm=np.linalg.norm(w_avg)
	del_theta=(w_avg_norm*time_step)/2 #assuming w units is rad/sec
	q_w_delt_scalar=math.cos(del_theta)    #theta input to cos has to be in radians
	q_w_delt_vector=(w_avg*math.sin(del_theta))/w_avg_norm
	q_w_delt=np.hstack((q_w_delt_scalar,q_w_delt_vector))
	qw_n_n1=np.hstack((0,np.cross(w_0.copy(),w_1.copy())))
	q_propogate=q_w_delt+((((time_step*time_step)/24)*qw_n_n1)) 
	q_final=quatMultiply(q_initial.copy(),q_propogate)
	return q_final/np.linalg.norm(q_final)

def cross_matrix(w):
	# represents crossproduct operation of a vector in matrix form
	cross_w_matrix=np.zeros((3,3))
	w_x=w[0:1].copy()
	w_y=w[1:2].copy()
	w_z=w[2:3].copy()
	cross_w_matrix[0,1]=(-w_z)
	cross_w_matrix[0,2]=w_y
	cross_w_matrix[1,0]=w_z
	cross_w_matrix[1,2]=(-w_x)
	cross_w_matrix[2,0]=(-w_y)
	cross_w_matrix[2,1]=w_x
	return cross_w_matrix

def calculate_phi(w_0):
	# calculates state matrix
	w_0_norm=np.linalg.norm(w_0.copy())
	del_theta=(w_0_norm*time_step)
	cross_w_0=cross_matrix(w_0.copy())
	theta_1=math.cos(del_theta)*np.identity(3)
	theta_2=(math.sin(del_theta)/w_0_norm)*cross_w_0
	w_0_rowmatrix=np.reshape(w_0.copy(),(1,3))
	theta_3=((1-math.cos(del_theta))/(w_0_norm*w_0_norm))*np.dot(np.transpose(w_0_rowmatrix),w_0_rowmatrix)
	theta=theta_1+theta_2+theta_3
	psi_1=-np.identity(3)*time_step
	psi_2=-((1-math.cos(del_theta))/np.power(w_0_norm,2))*cross_w_0
	psi_3=-((del_theta-math.sin(del_theta))/np.power(w_0_norm,3))*np.dot(cross_w_0,cross_w_0) 
	psi=psi_1+psi_2+psi_3
	phi_1=np.hstack((theta,psi))
	phi_2=np.hstack((np.zeros((3,3)),np.identity(3)))
	phi=np.vstack((phi_1,phi_2))
	return phi

def calculate_Q_d(w_0):
	# Q_d is the term added while propagating state covariance matrix
	w_0_norm=np.linalg.norm(w_0.copy())
	del_theta=(w_0_norm*time_step)
	cross_w_0=cross_matrix(w_0.copy())
	Q_11_1=np.power(sigma_r,2)*time_step*np.identity(3) 
	Q_11_22_numrtr= (np.power(del_theta,3)/3) + (2*math.sin(del_theta)) - (2*del_theta)
	Q_11_2=((np.power(time_step,3)/3)*np.identity(3)) +(Q_11_22_numrtr/np.power(w_0_norm,5))*np.dot(cross_w_0,cross_w_0) 
	Q_11= Q_11_1 + (np.power(sigma_w,2)*Q_11_2)
	Q_12_1=(np.identity(3)*np.power(time_step,2))/2
	Q_12_2=((del_theta-math.sin(del_theta))/np.power(w_0_norm,3))*cross_w_0
	Q_12_3_numrtr=(np.power(del_theta,2)/2)+ math.cos(del_theta)- 1
	Q_12_3= (Q_12_3_numrtr/np.power(w_0_norm,4))*np.dot(cross_w_0,cross_w_0)
	Q_12=-np.power(sigma_w,2)*(Q_12_1+Q_12_2+Q_12_3)
	Q_22= np.power(sigma_w,2)*time_step*np.identity(3)
	Q1=np.hstack((Q_11,Q_12))
	Q2=np.hstack((Q_12,Q_22))
	Q=np.vstack((Q1,Q2))
	return Q

def calculate_H(z_mdl,q):
	#matrix that relates sensor deviation from model with error in state vector
	temp=quatRotate(q.copy(),z_mdl.copy())
	cross_temp= cross_matrix(temp)
	H=np.hstack((-cross_temp,np.zeros((3,3))))
	return H #3X6

#propogation
state_k1[4:7]=state_k0[4:7] #b
w_k1=w_m_k1 -state_k1[4:7]  #w k+1 given k
state_k1[0:4]=quat_propogation(state_k0[0:4],w_k0,w_k1) #q 
phi=calculate_phi(w_k0)
Q_d=calculate_Q_d(w_k0)
cov_k1=np.dot(np.dot(phi,cov_k0),np.transpose(phi)) + Q_d #6X6

#Update
r=z_sunsensor-quatRotate(state_k1[0:4],z_sunsensor_mdl)
H=calculate_H(z_sunsensor_mdl,state_k1[0:4])
cov_r= np.dot(np.dot(H,cov_k1),np.transpose(H)) + cov_sunsensor #3X3
Kalman_gain= np.dot(np.dot(cov_k1,np.transpose(H)), np.linalg.inv(cov_r)) #6X3
state_correction= np.dot(r,np.transpose(Kalman_gain))  #1X6
q_correction=np.hstack((1,(state_correction[0:3]/2)))   #careful! quaternion convention is scalar, vector; state correction has 6 elts not 7
q_correction_norm=np.linalg.norm(q_correction)
q_correction=q_correction/q_correction_norm
state_k0[0:4]= quatMultiply(q_correction,state_k1[0:4]) #q update
state_k0[4:7]= state_k1[4:7]+ state_correction[3:6] #b update
w_k0= w_m_k1 - state_k0[4:7] #w update w k+1 given k+1
factor= np.identity(6)- np.dot(Kalman_gain,H) # factor dependent on kalman gain for calculation of new covariance #6X6
cov_k0= np.dot(np.dot(factor,cov_k1),np.transpose(factor)) + (np.dot(np.dot(Kalman_gain,cov_sunsensor),np.transpose(Kalman_gain))) # covariance update #6X6
