from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
import sys
import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#local modules, no name conflicts
from ia_geometry_func import *
from execution_func import *

###############################################################################################################
########                     This block contains Setups for this simulation                            ########
########                     Scroll to the end of file to code executions                              ########
########                     Check for Rank-Dependence before running                                  ########
###############################################################################################################
#
def baR_func(x,y,z,global_r,global_theta,global_phi): 
    return baR_0

def theta_func(x,y,z,global_r,global_theta,global_phi,std): 
    #### GAUSSIAN NOISE
    return global_theta+np.random.normal(0,std) #(center, std)

def phi_func(x,y,z,global_r,global_theta,global_phi,std): 
    #### GAUSSIAN NOISE
    return global_phi+np.random.normal(0,std)

def density_func(x,y,z,global_r,global_theta,global_phi): #NEED TO BE NORMALIZED
    # this gives the Fraction NUMBER of gals expected in a box of size dV, approximated at (x,y,z)
    # should normalize to 1, but given the finiteness of n,  this only holds as n gets large
    if (global_r > 1): return 0
    else: 
        #return 1/(4*np.pi/3)
        return 1/((global_r)*(1+global_r)**2)

###############################################################################################################
########                     This block contains the core functions for this simulation                ########
########                               Check for Rank-Dependence before running                        ########
###############################################################################################################

def get_RA_data(save_2D,iteration_tracker,outputpath,mode,n,baR_0,phistd,thetastd): 
    import sys
    #n=64 ~ 1min30sec
    
    #x,y,z are of size 2n+1, {+1,-1, and 0 always included to prevent singularities in functions}
    if mode == 'linear':
        xs = get_axis_lin(n)
        ys = get_axis_lin(n)
        zs = get_axis_lin(n)
    if mode == 'linear_cut': #To be implemented
        xs = None
        ys = None
        zs = None
        sys.exit('code not ready')
    if mode == 'log':
        xs = get_axis_log(n)
        ys = get_axis_log(n)
        zs = get_axis_log(n)
    
    #get a list of mid points and cell widths for later computing values in each simulated cell
    midx = get_mid_points(xs)
    midy = get_mid_points(ys)
    midz = get_mid_points(zs)
    diffx = get_diff_list(xs)
    diffy = get_diff_list(ys)
    diffz = get_diff_list(zs)
    #print(xs)
    #print(midx)
    #print(diffx)
    
    #this prevents the density function later used become to small for numeric purposes
    #this will get divided out later. doesnt really matter
    total_number_of_sats = (2*n)**3
    
    #[[x_cell_midpt, y_cell_midpt, r , sum_eps+(midpt)_at_xy, sum_numberfraction(midpt)_at_xy]]
    #This will be saved in file. # x, y are only used to plot the 2D density profile. Otherwise, r is enough
    Proj_Data=np.empty((0,5))
    
    for x_index in range(2*n):
        #looping in range2*n ensures never out of bound
        #midpoint_cell_coordinate will be referred to as = x,y,z.
        x = midx[x_index]
        dx = diffx[x_index]
        
        for y_index in range(2*n):
            y = midy[y_index]
            dy = diffy[y_index]
            
            [[global_r,global_phi]] = Cart_to_Polar(np.array([[x,y]]))
            weighted_orients_same_r = np.zeros(2) #stores np.array: [sum_of_weighted_epsilon+_at_r, sum_of_rho_at_r]
            #print(x,y,global_r)
            for z_index in range(2*n):
                z = midz[z_index]
                dV=dx*dy*diffz[z_index]
                #print(dV)
                
                #global here refers to global coordinates
                #local meanning local to the cell, evaluated at cell midpt
                #[[global_r,global_theta,global_phi]] = Cart_to_Sph(np.array([[x,y,z]]))   #get sat's position in global sph coord
                global_theta = RZ_to_Theta(np.array([[global_r,z]]))
                #each function will decide if (xyz) or (r,theta, phi) is faster
                #other parameters are defined globally before this function is run
                local_num_fraction = density_func(x,y,z,global_r,global_theta,global_phi)*dV*total_number_of_sats    #density * volume
                local_baR          = baR_func(x,y,z,global_r,global_theta,global_phi)
                local_theta        = theta_func(x,y,z,global_r,global_theta,global_phi,thetastd)
                local_phi          = phi_func(x,y,z,global_r,global_theta,global_phi,phistd)
                
                #get sat's orientation (see nnote below!!!) in local sph and cartesian coord. #not very useful, deactivated.
                #But IMPORTANT to note that
                #This vector IS NOT WHAT SATELLITEs PHYSICALLY LOOKLIKE. R direction encodes b/a Ratio. 
                #But this vector physically parallels the satellite's shape
                #local_baR_vector = np.array([[local_baR,local_theta,local_phi]]) 
                #[[local_x,local_y,local_z]] = Sph_to_Cart(local_baR_vector)

                #Deprecated #build 2D projected catalogue with average baR
                proj_mag = Ellip_proj_mag(local_baR,local_theta)    #calculate the projected b/a Ratio first
                #print('ptoeps',Polar_to_eps(global_phi,proj_mag,local_phi))
                #print(local_num_fraction)
                weighted_orients_same_r[0]+=Polar_to_eps(global_phi,proj_mag,local_phi)*local_num_fraction #sum over z
                weighted_orients_same_r[1]+=local_num_fraction
                #print(weighted_orients_same_r)
            Proj_Data = np.vstack((Proj_Data,np.array([
                                   x,
                                   y,
                                   global_r,
                                   weighted_orients_same_r[0],
                                   weighted_orients_same_r[1]])))
                
                 #This whole section is removed in favor of a the above more efficient code (and the algorithm has changed to reflect the logic of true observations)
                
#                #Deprecated #[[proj_local_x, proj_local_y]]=Polar_to_Cart(np.array([[proj_mag,local_phi]]))    #convert the projected vector to cartesian
#                #Deprecated #weighted_orients_same_xy = np.append(weighted_orients_same_xy,[[proj_local_x*local_num_fraction,proj_local_y*local_num_fraction,local_num_fraction]],axis=0)
#
#            #Deprecated #compute the average baR vector. add all orient vectors of the same x,y coord element-wise
#            #Deprecated #sum_temp = np.sum(weighted_orients_same_xy,axis=0)
#
#            if (np.abs(sum_temp[2])) >= 10.**(-13):# if the density is 0, can just set the mean vector to 0, this also avoids 0 division
#               [mean_temp_x,mean_temp_y] = sum_temp[0:2]/sum_temp[2]
#               sum_temp[2]=sum_temp[2]/total_number_of_sats #this gets rid of the run-dependent factor that helped with the previous calculations
#                
#                #Deprecated! All_Sats_2D = np.append(All_Sats_2D,[[x,y,mean_temp_x,mean_temp_y,sum_temp[2]]],axis=0)
#                
#                #from the average baR vector, compute the expected gamma+ value at (x,y)
#                Proj_Data = np.append(Proj_Data,np.array([[
#                                                                        x,
#                                                                        y,
#                                                                        Cart_to_eps(np.array([[x,y,mean_temp_x,mean_temp_y]])),
#                                                                        sum_temp[2],
#                                                                        mean_temp_x,
#                                                                        mean_temp_y,
#                                                                        ]]),axis=0)
    
    #Now reduce the 2D data to 1D: perform radial averaging and get the gamma+ function
    #get the radial axis from 0 to 1, this has size n+1. Will later take the number of intervals, 
    #and will thus reduce the size of radial axis to n
    if mode == 'linear':
        rs = get_axis_lin(n)[n:2*n+1]
    if mode == 'linear_cut': #To be implemented
        rs = None
        sys.exit('code not ready')
    if mode == 'log':
        rs = get_axis_log(n)[n:2*n+1]
    
    gamma_plus = np.zeros(n) #size=n
    densities = np.zeros(n)
    #print(Proj_Data[:,3])
    #print(Proj_Data[:,4])
    #Will store densities and rs seperately #R_density_profile = np.empty((0,n)) #stores[[Radial coord],densities]
    for i in range(n): # n = len(rs)-1. i.e. there is nothing beyond R_cut
        #each i is a radial interval
        #print(rs[i])
        count=0
        #print(Proj_Data)
        for elmt in Proj_Data:
            #print(elmt)
            # if in the first, second, third, radial interval(s)... and so on.
            xyplane_dist = elmt[2]
            if (xyplane_dist >= rs[i] and xyplane_dist < rs[i+1]):
                gamma_plus[i] += elmt[3]
                count += elmt[4] #adds up the total number density, in case it is 0, and cannot be used to average Gamma+
        #print(gamma_plus[i],count)
        if (count == 0):
            gamma_plus[i]=0
        else: gamma_plus[i]=gamma_plus[i]/count
        densities[i] = count/((rs[i+1]-rs[i])*2*np.pi*((rs[i+1]+rs[i])/2))
    rs = get_mid_points(rs)
    
    densities2D = np.vstack((Proj_Data[:,0],Proj_Data[:,1],Proj_Data[:,4]))
#    This section is killed in favor of the above more efficient one: In General_Condition_IA_Sim_v5_3DDirectComputation    
#    for i in range(n): # n = len(rs)-1. i.e. there is nothing beyond R_cut
#        #each i is a radial interval
#        count=0
#        for elmt in Proj_Data:
#            # if in the first, second, third, radial interval(s)... and so on.
#            xyplane_dist = np.sqrt(elmt[0]**2+elmt[1]**2)
#            if (xyplane_dist >= rs[i] and xyplane_dist < rs[i+1]):
#                gamma_plus[i] += elmt[2]*elmt[3]
#                count += elmt[3] #adds up the total number density, in case it is 0, and cannot be used to average Gamma+
#        if (count == 0):
#            gamma_plus[i]=0
#        else: gamma_plus[i]=gamma_plus[i]/count
#        densities[i] = count
                

    
    #R_density_profile = np.vstack((R_density_profile,densities))
    #R_density_profile = np.vstack((R_density_profile,rs))
                
    if save_2D == True:
        if iteration_tracker == 0: #only save rs once for a setting
            write_file_at_path(outputpath, 'NA', rs,'rs'+mode+'mod')
        #write_file_at_path(outputpath, 'proj_data', Proj_Data,str(iteration_tracker)+mode+'mod')
        write_file_at_path(outputpath, 'gamma_plus', gamma_plus,str(iteration_tracker)+mode+'mod')
        write_file_at_path(outputpath, 'densities', densities,str(iteration_tracker)+mode+'mod')
        write_file_at_path(outputpath, 'densities2D', densities2D,str(iteration_tracker)+mode+'mod')

baR_0 = 0.2
n=64
smr_run = np.pi/4
outputpath = '/home/azhou/IA-Sim-AZhou00/IA_Numeric_Output/NFW log n %i baR_0 %1.2f smr %1.2f'%(n,0.2,smr_run)
offset = 0
size_per_rank = 3
def run_rank_sing_cond(node_index,batch_num,offset): 
    if rank == node_index:
        if batch_num == 0:
            for i in range(size_per_rank):
                get_RA_data(True,i+(size_per_rank*(rank-1)),outputpath,'linear',n,baR_0,smr_run,smr_run)
        if batch_num == 1:
            for i in range(size_per_rank):
                get_RA_data(True,i+(size_per_rank*(rank-7)),outputpath,'log',n,baR_0,smr_run,smr_run)

#how ever many slot you want to run:
run_rank_sing_cond(0,0,offset)
run_rank_sing_cond(1,0,offset)
run_rank_sing_cond(2,0,offset)
run_rank_sing_cond(3,0,offset)
run_rank_sing_cond(4,0,offset)
run_rank_sing_cond(5,0,offset)
run_rank_sing_cond(6,0,offset)
time.sleep(5) #this gives time for the previous threads to create directories, etcs.
run_rank_sing_cond(7,1,offset)
run_rank_sing_cond(8,1,offset)
run_rank_sing_cond(9,1,offset)
run_rank_sing_cond(10,1,offset)
run_rank_sing_cond(11,1,offset)
run_rank_sing_cond(12,1,offset)
run_rank_sing_cond(13,1,offset)
run_rank_sing_cond(14,1,offset)
