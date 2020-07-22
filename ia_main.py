from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
import sys
import os
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

def baR_func(x,y,z,global_r,global_theta,global_phi): 
    return baR_0

def theta_func(x,y,z,global_r,global_theta,global_phi,std): 
    #### GAUSSIAN NOISE
    return global_theta+np.random.normal(0,std) #center, std = pi/4 = 0.7853981633974483

def phi_func(x,y,z,global_r,global_theta,global_phi,std): 
    #### GAUSSIAN NOISE
    return global_phi+np.random.normal(0,std)

def density_func(x,y,z,global_r,global_theta,global_phi): #NEED TO BE NORMALIZED
    # this gives the Fraction NUMBER of gals expected in a box of size dV, approximated at (x,y,z)
    # should normalize to 1, but given the finiteness of n,  this only holds as n gets large
    if (x**2+y**2+z**2 > R_cut**2): return 0
    else: return dV/(4*np.pi/3)

def set_condition(n,R_cut,baR_0):
    global dV 
    dV = (2*R_cut)**3/n**3

###############################################################################################################
########                     This block contains the core functions for this simulation                ########
########                               Check for Rank-Dependence before running                        ########
###############################################################################################################

def get_RA_data(rank,save_2D,iteration_tracker,outputpath,n,R_cut,baR_0,phistd,thetastd): 
    #n=50 ~ 6.18 sec without getting All_Sat (best trial)
    #n=100 ~  48.6 sec without getting All_Sat
    #n=200 ~ 7-10 min

    xs = np.linspace(- R_cut, R_cut, n)
    ys = np.linspace(- R_cut, R_cut, n)
    zs = np.linspace(- R_cut, R_cut, n)

    #All_Sats=np.empty((0,7)) #[[x,y,z,xlocal,ylocal,zlocal,numberfraction]]
    All_Sats_2D=np.empty((0,5))  #[[x,y,average_xlocal, average_ylocal,numberfraction]]
    All_Sats_2D_eps=np.empty((0,4))  #[[x,y,eps+,numberfraction]]
    for x in xs:
        #print(x)
        for y in ys:
            temp_baR_num = np.empty((0,3)) 
            # this stores the data point with the same x,y but different z coord, and store [[localx*num,localy*num,num(i.e. frac density)],...]
            for z in zs:
                [[global_r,global_theta,global_phi]] = Cart_to_Sph(np.array([[x,y,z]]))   #get sat's position in global sph coord

                local_density = density_func(x,y,z,global_r,global_theta,global_phi)
                local_baR = baR_func(x,y,z,global_r,global_theta,global_phi)
                local_theta = theta_func(x,y,z,global_r,global_theta,global_phi,thetastd)
                local_phi = phi_func(x,y,z,global_r,global_theta,global_phi,phistd)

                local_baR_vector = np.array([[local_baR,local_theta,local_phi]]) #get sat's orientation in local sph coord
                [[local_x,local_y,local_z]] = Sph_to_Cart(local_baR_vector) #get sat's orientation in local Cart coord

                #build 3D catalogue
    #            All_Sats = np.append(All_Sats,[[x, #sat's location
    #                                            y,  #sat's location
    #                                            z,  #sat's location
    #                                            local_x,  #sat's baR mag and orient
    #                                            local_y,  #sat's baR mag and orient
    #                                            local_z,  #sat's baR mag and orient
    #                                            local_density #approximated frac of sats in dV around (x,y,z)
    #                                            ]],axis=0)

                #build 2D projected catalogue with average baR
                proj_mag = Ellip_proj_mag(local_baR,local_theta)    #calculate the projected b/a Ratio first
                [[proj_local_x, proj_local_y]]=Polar_to_Cart(np.array([[proj_mag,local_phi]]))    #convert the projected vector to cartesian

                temp_baR_num = np.append(temp_baR_num,[[proj_local_x*local_density,proj_local_y*local_density,local_density]],axis=0)

            #compute the average baR vector
            sum_temp = np.sum(temp_baR_num,axis=0)

            if (np.abs(sum_temp[2])) >= 10.**(-13):# if the density is 0, can just set the mean vector to 0, this also avoids 0 division
                [mean_temp_x,mean_temp_y] = sum_temp[0:2]/sum_temp[2]
                All_Sats_2D = np.append(All_Sats_2D,[[x,y,mean_temp_x,mean_temp_y,sum_temp[2]]],axis=0)

                #from the average baR vector, compute the expected eps+ value at (x,y)
                All_Sats_2D_eps = np.append(All_Sats_2D_eps,np.array([[
                                                                        x,
                                                                        y,
                                                                        Cart_to_eps(np.array([[x,y,mean_temp_x,mean_temp_y]])),
                                                                        sum_temp[2]
                                                                        ]]),axis=0)
    #Now perform radial averaging and get the gamma+ function
    n_radial = int(n+1) #partition the radial direction  into n intervals = n+1 points 
    rs = np.linspace(0, R_cut, n_radial)#size=n+1, will discard the last data point
    gamma_plus = np.zeros(n)#size=n
    for i in range(len(rs)-1): #there is nothing beyond R_cut
        count=0
        for elmt in All_Sats_2D_eps:
            # if in the radial interval of 0 and 0.001(lets say e.g.).., and so on
            if (np.sqrt(elmt[0]**2+elmt[1]**2) >= rs[i] and np.sqrt(elmt[0]**2+elmt[1]**2) < rs[i]+R_cut/(n_radial-1)):
                count += elmt[3]
                gamma_plus[i] += elmt[2]*elmt[3]
        if (count == 0):
            #print(count)
            #print(gamma_plus[i])
            gamma_plus[i]=0
        else: gamma_plus[i]=gamma_plus[i]/count
    rs = rs[0:-1]+(rs[0]+rs[1])/2 #so that the corresponding computed gamma_plus is at the center for each radial direction interval. Discard the last interval
    #rint(len(rs),len(gamma_plus))
    #rs=np.linspace(0, 1, 11)
    #print(rs)
    #rs = rs[0:-1]+(rs[0]+rs[1])/2
    #print(rs)
    if save_2D == True:
        write_file_at_path(outputpath, 'All_Sats_2D smr=%1.2f'%rad_to_deg(phistd), All_Sats_2D,iteration_tracker)
        write_file_at_path(outputpath, 'All_Sats_2D_eps smr=%1.2f'%rad_to_deg(phistd), All_Sats_2D_eps,iteration_tracker)
        write_file_at_path(outputpath, 'Gamma_plus smr=%1.2f'%rad_to_deg(phistd), gamma_plus,iteration_tracker)
        print('rank ',rank,': finished saving 2D, 2D_EPS, and Gamma_plus')

def Plot_Gamma_Plus(rank,n,smoothing_len,baR_0, outputpath,searchpath,imagename):
    #smoothing length (multiples of 2): plotting one datapoint for every smoothing length worth of eps data (taking arithmetic average ), >=2
    #output path is where the figure folder is saved
    #searchpath is where all the gamma_plus data are stored
    rs_presmooth = np.linspace(0, R_cut, n+1) #size = n+1
    rs_presmooth = rs_presmooth[0:-1]+(rs_presmooth[0]+rs_presmooth[1])/2 #size = n
    rs = np.array([sum(rs_presmooth[i:i+smoothing_len])/smoothing_len for i in range(0,len(rs_presmooth),smoothing_len)])
        
    #read all the file names in the searchpath folder
    filenames = [f for f in os.listdir(searchpath) if os.path.isfile(os.path.join(searchpath, f))]
    print('rank ',rank,': file names discovered',filenames)
    
    fig= plt.figure(figsize=(9,6))
    #Plotting gamma curves
    for filename in filenames:
        filepath = os.path.join(searchpath, filename)
        file = open(filepath, "rb")
        gammapls_temp = np.load(file)
        file.close
        
        smoothed_gamma = [sum(gammapls_temp[i:i+smoothing_len])/smoothing_len for i in range(0,len(gammapls_temp),smoothing_len)]
        #print(smoothed_gamma)
        #print(gammapls_temp)
        #print(len(smoothed_gamma))
        plt.plot(rs,smoothed_gamma,label=filename)
    
    #Plotting reference curves
    asymp = np.array([])
    y_0 = np.zeros(rs.shape)
    for r in rs:
        asymp = np.append(asymp,(1-(baR_0)**2)/(1+(baR_0)**2))

    plt.plot(rs,y_0,label='y=0')
    plt.plot(rs,asymp,"--",label=(
        "asymptotic value %1.3f, b/a_0 = %1.2f" %(
            float((1-(baR_0)**2)/(1+(baR_0)**2)), baR_0)))

    plt.xlabel('distance')
    plt.ylabel('gamma +')
    plt.title('gamma + vs. normalized distance from cluster center, smoothing=%i'%smoothing_len)
    plt.xlim(0,1)
    plt.ylim(-1,1)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    #plt.show()
    imagesavepath = os.path.join(outputpath, 'figures')
    if not os.path.exists(imagesavepath):
        os.makedirs(imagesavepath)
    fig.savefig(os.path.join(imagesavepath, imagename), bbox_inches = 'tight')

###############################################################################################################
########                     Execution code happens below                                              ########
###############################################################################################################
    
n=50
R_cut = 1.0
baR_0 = 0.2
set_condition(n,R_cut,baR_0)
sim_step = 30

def run_rank(node_index):
    if rank == node_index:
        smr=(np.pi)/8*(rank)
        outputpath = '/home/azhou/IA-Sim-AZhou00/IA_Numeric_Output/n%i-baR_0%1.1f-phiSTD%1.2f-thetaSTD%1.2f' %(n,baR_0,rad_to_deg(smr),rad_to_deg(smr))
        for index in range(sim_step):
            print('rank ',rank,': on task ',index+1)
            get_RA_data(rank,True,index,outputpath,n,R_cut,baR_0,smr,smr)
            print('rank ',rank,': tasks ',index+1,' out of ',sim_step,' done')
    
        searchpath = os.path.join(outputpath,'Gamma_plus smr=%1.2f'%rad_to_deg(smr))
        smoothing = 1
        Plot_Gamma_Plus(rank,n,smoothing,baR_0,outputpath,searchpath,'GammaPlus_Smoothing=%i'%smoothing)
        smoothing = 1
        Plot_Gamma_Plus(rank,n,smoothing,baR_0,outputpath,searchpath,'GammaPlus_Smoothing=%i'%smoothing)
        #Plot_Gamma_Plus(rank,n,smoothing+3,baR_0,outputpath,searchpath,'GammaPlus_Smoothing=%i'%smoothing+3)

#how ever many slot you want to run:
run_rank(0)
run_rank(1)
run_rank(2)
run_rank(3)
run_rank(4)
run_rank(5)
run_rank(6)
run_rank(7)
