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
        write_file_at_path(outputpath, 'All_Sats_2D', All_Sats_2D,iteration_tracker)
        write_file_at_path(outputpath, 'All_Sats_2D_eps', All_Sats_2D_eps,iteration_tracker)
        write_file_at_path(outputpath, 'Gamma_plus', gamma_plus,iteration_tracker)
        print('rank ',rank,': finished saving 2D, 2D_EPS, and Gamma_plus')

def Plot_Gamma_Plus(rank,n,smoothing_len,baR_0, outputpath,searchpath):
    #smoothing length (multiples of 2): plotting one datapoint for every smoothing length worth of eps data (taking arithmetic average ), >=2
    #output path is where the figure folder is saved
    #searchpath is where all the gamma_plus data are stored
    rs_presmooth = np.linspace(0, R_cut, n+1) #size = n+1
    rs_presmooth = rs_presmooth[0:-1]+(rs_presmooth[0]+rs_presmooth[1])/2 #size = n
    rs = np.array([sum(rs_presmooth[i:i+smoothing_len])/smoothing_len for i in range(0,len(rs_presmooth),smoothing_len)])
        
    #read all the file names in the searchpath folder
    filenames = [f for f in os.listdir(searchpath) if os.path.isfile(os.path.join(searchpath, f))]
    print('rank ',rank,': ',len(filenames),' files discovered')
    
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
        plt.plot(rs,smoothed_gamma)#,label=filename)
    
    #Plotting reference curves
    asymp = np.array([])
    y_0 = np.zeros(rs.shape)
    for r in rs:
        asymp = np.append(asymp,(1-(baR_0)**2)/(1+(baR_0)**2))
    plt.plot(rs,y_0)#,label='y=0')
    plt.plot(rs,asymp,"--")#,label=(
        #"asymptotic value %1.3f, b/a_0 = %1.2f" %(
        #    float((1-(baR_0)**2)/(1+(baR_0)**2)), baR_0)))
    
    if PLOT_LOG_FLAG == True: plt.xscale('log')
    plt.xlabel('distance')
    plt.ylabel('gamma +')
    plt.title('gamma + vs. normalized distance from cluster center, smoothing=%i'%smoothing_len)
    if PLOT_LOG_FLAG == True: 
        plt.xlim(rs[0],1)
    else: plt.xlim(0,1)
    plt.ylim(-1,1)
    #plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    #plt.show()
    imagesavepath = os.path.join(outputpath, 'figures')
    if not os.path.exists(imagesavepath):
        os.makedirs(imagesavepath)
    if PLOT_LOG_FLAG == True:
        fig.savefig(os.path.join(imagesavepath, 'GammaPlus LOG smoothing = %i'%smoothing_len), bbox_inches = 'tight')
    else: fig.savefig(os.path.join(imagesavepath, 'GammaPlus smoothing = %i'%smoothing_len), bbox_inches = 'tight')
    
#The error bar function
#in each folder of Gamma_plus_some_qualifiers, there are data of multiple runs on the same settinf
#so first we want to navigate to that folder via search_path, and read all the Gamma_plus_files
#This is analogous of the ploting functions above
#This function searches for all the files in a folder. The files should all contain 1-dim np.array. 
#It applies smoothing first, then
#It takes the two lines that signifies the 1 and 2 th STD upper&lower bounds
#and return these 4 lines in one 2-D file that contain both lists.

def Get_Error_Bar(n,smoothing_len,outputpath,searchpath): 
    #outputpath is where the figure folder is saved
    #searchpath is where all the gamma_plus data are stored
    #will save a file, and a plot     
    #[
    #[y value of  std 1 upperbound],
    #[y value of  std 1 lowerbound],
    #[y value of  std 2 upperbound],
    #[y value of  std 2 lowerbound],
    #[smoothed radial coordinate]
    #]

    rs_presmooth = np.linspace(0, R_cut, n+1) #size = n+1
    rs_presmooth = rs_presmooth[0:-1]+(rs_presmooth[0]+rs_presmooth[1])/2 #size = n
    rs = np.array([sum(rs_presmooth[i:i+smoothing_len])/smoothing_len for i in range(0,len(rs_presmooth),smoothing_len)])
        
    #read all the file names in the searchpath folder
    filenames = [f for f in os.listdir(searchpath) if os.path.isfile(os.path.join(searchpath, f))]
    print('rank ',rank,': ',len(filenames),' files discovered')
    
    complete_GamPls = np.empty((0,int(n/smoothing_len)))
    
   
    #get all the gamma data, and smooth them accordingly
    for filename in filenames:
        filepath = os.path.join(searchpath, filename)
        file = open(filepath, "rb")
        gammapls_temp = np.load(file)
        file.close
        
        smoothed_gamma = [sum(gammapls_temp[i:i+smoothing_len])/smoothing_len for i in range(0,len(gammapls_temp),smoothing_len)]
        #now sort each column of the matrix and take the standard deviations
        complete_GamPls = np.vstack((complete_GamPls,smoothed_gamma))
    
    error_GamPls = sort_matrix_columns(complete_GamPls)
    #print(complete_GamPls)
    error_GamPls = get_1_2_std(error_GamPls)
    #print(complete_GamPls)
    error_GamPls = np.vstack((error_GamPls,rs)) #attach the smoothed x coordinate
    #save in the figures folder
    write_namedfile_at_path(outputpath, 'figures',error_GamPls,'STD_With_Radial_Coord_smooth=%i'%smoothing_len)
    
    
    ### NOW PLOT OVERLAYED SCATTER PLOT + STD ###
    #############################################
    fig = plt.figure(figsize=(9,6))
    for gam in complete_GamPls:
        plt.scatter(rs,gam,c='r',s=0.25,marker='o')
    
    plt.fill_between(rs,error_GamPls[2],error_GamPls[3],alpha = 0.4,label='2nd STD')    
    plt.fill_between(rs,error_GamPls[0],error_GamPls[1],alpha = 0.4,label='1st STD')

    #plt.plot(rs,complete_GamPls[0],label='1st STD')
    #plt.plot(rs,complete_GamPls[1],label='1st STD')
    #plt.plot(rs,complete_GamPls[2],label='2nd STD')
    #plt.plot(rs,complete_GamPls[3],label='2nd STD')
    
    #Plotting reference curves
    asymp = np.array([])
    y_0 = np.zeros(rs.shape)
    for r in rs:
        asymp = np.append(asymp,(1-(baR_0)**2)/(1+(baR_0)**2))

    plt.plot(rs,y_0,label='y=0')
    plt.plot(rs,asymp,"--",label=(
        "asymptotic value %1.3f, b/a_0 = %1.2f" %(
            float((1-(baR_0)**2)/(1+(baR_0)**2)), baR_0)))
    
    if PLOT_LOG_FLAG == True: plt.xscale('log')         
    plt.xlabel('distance')
    plt.ylabel('gamma +')
    plt.title('gamma + vs. normalized R, smoothing=%i'%smoothing_len)
    if PLOT_LOG_FLAG == True: 
        plt.xlim(rs[0],1)
    else: plt.xlim(0,1)
    plt.ylim(-1,1)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    #plt.show()
    imagesavepath = os.path.join(outputpath, 'figures')
    if not os.path.exists(imagesavepath):
        os.makedirs(imagesavepath)
    if PLOT_LOG_FLAG == True: 
        fig.savefig(os.path.join(imagesavepath, 'OVERLAY LOG smoothing = %i'%smoothing_len), bbox_inches = 'tight')  
    else: fig.savefig(os.path.join(imagesavepath, 'OVERLAY smoothing = %i'%smoothing_len), bbox_inches = 'tight')       
    
    
    ######  NOW PLOT JUST STD ########
    ##################################
    fig2 = plt.figure(figsize=(9,6))
    plt.fill_between(rs,error_GamPls[2],error_GamPls[3],alpha = 0.4,label='2nd STD')    
    plt.fill_between(rs,error_GamPls[0],error_GamPls[1],alpha = 0.4,label='1st STD')

    #plt.plot(rs,complete_GamPls[0],label='1st STD')
    #plt.plot(rs,complete_GamPls[1],label='1st STD')
    #plt.plot(rs,complete_GamPls[2],label='2nd STD')
    #plt.plot(rs,complete_GamPls[3],label='2nd STD')
    
    #Plotting reference curves
    plt.plot(rs,y_0,label='y=0')
    plt.plot(rs,asymp,"--",label=(
        "asymptotic value %1.3f, b/a_0 = %1.2f" %(
            float((1-(baR_0)**2)/(1+(baR_0)**2)), baR_0)))
    
    if PLOT_LOG_FLAG == True: plt.xscale('log')         
    plt.xlabel('distance')
    plt.ylabel('gamma +')
    plt.title('gamma + vs. normalized R, smoothing=%i'%smoothing_len)
    if PLOT_LOG_FLAG == True: 
        plt.xlim(rs[0],1)
    else: plt.xlim(0,1)
    plt.ylim(-1,1)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    #plt.show()
    imagesavepath = os.path.join(outputpath, 'figures')
    if not os.path.exists(imagesavepath):
        os.makedirs(imagesavepath)
    if PLOT_LOG_FLAG == True: 
        fig2.savefig(os.path.join(imagesavepath, 'STD LOG smoothing = %i'%smoothing_len), bbox_inches = 'tight')  
    else: fig2.savefig(os.path.join(imagesavepath, 'STD smoothing = %i'%smoothing_len), bbox_inches = 'tight')       
    
###############################################################################################################
########                     Execution code happens below                                              ########
###############################################################################################################
    
n=256
R_cut = 1.0
baR_0 = 0.2
set_condition(n,R_cut,baR_0)
sim_step = 25
offset = 0
PLOT_LOG_FLAG = True

def run_rank(node_index,batch_num,offset): #starting accumulate data at filename = offset.
    #batch number  = 0 or 1. This let the 16 threads to run instead of only 8 cores. 
    #The batch number 2 runs the second copy of the sim steps
    if rank == node_index:
        smr=(np.pi)/8*(rank-size*batch_num/2)
        outputpath = '/home/azhou/IA-Sim-AZhou00/IA_Numeric_Output/n%i-baR_0%1.1f-phiSTD%1.2f-thetaSTD%1.2f' %(n,baR_0,rad_to_deg(smr),rad_to_deg(smr))
        searchpath = os.path.join(outputpath,'Gamma_plus')
        
        index_list = np.array(list(range(sim_step)))+(batch_num*sim_step)
        index_list = index_list.astype(int)
        
        for index in index_list:
            print('rank ',rank,': on task ',index+1)
            true_index = index+offset
            get_RA_data(rank,True,true_index,outputpath,n,R_cut,baR_0,smr,smr)
            print('rank ',rank,': tasks ',index+1,' out of ',sim_step*(batch_num+1),' done')
    
        smoothing = 2
        Plot_Gamma_Plus(rank,n,smoothing,baR_0,outputpath,searchpath)
        smoothing = 4
        Plot_Gamma_Plus(rank,n,smoothing,baR_0,outputpath,searchpath)
        
        if batch_num == 1: #the later batch go get error bar
            #Get_Error_Bar(n,1,outputpath,searchpath)
            Get_Error_Bar(n,2,outputpath,searchpath)
            Get_Error_Bar(n,4,outputpath,searchpath)
            #Get_Error_Bar(n,8,outputpath,searchpath)

def run_rank_plots(node_index,batch_num): 
    if rank == node_index:
        if batch_num == 0:
            
            smr=(np.pi)/8*(rank-size*batch_num/2)
            outputpath = '/home/azhou/IA-Sim-AZhou00/IA_Numeric_Output/n%i-baR_0%1.1f-phiSTD%1.2f-thetaSTD%1.2f' %(n,baR_0,rad_to_deg(smr),rad_to_deg(smr))
            searchpath = os.path.join(outputpath,'Gamma_plus')
            
            smoothing = 2
            Plot_Gamma_Plus(rank,n,smoothing,baR_0,outputpath,searchpath)
            Get_Error_Bar(n,4,outputpath,searchpath)
            smoothing = 4
            Plot_Gamma_Plus(rank,n,smoothing,baR_0,outputpath,searchpath)
            Get_Error_Bar(n,8,outputpath,searchpath)
            #Get_Error_Bar(n,8,outputpath,searchpath)
"""            
#how ever many slot you want to run:
run_rank(0,0,offset)
run_rank(1,0,offset)
run_rank(2,0,offset)
run_rank(3,0,offset)
run_rank(4,0,offset)
run_rank(5,0,offset)
run_rank(6,0,offset)
time.sleep(5) #this gives time for the previous threads to create directories, etcs.
run_rank(7,1,offset)
run_rank(8,1,offset)
run_rank(9,1,offset)
run_rank(10,1,offset)
run_rank(11,1,offset)
run_rank(12,1,offset)
run_rank(13,1,offset)
run_rank(14,1,offset)"""

run_rank_plots(0,0)
run_rank_plots(1,0)
run_rank_plots(2,0)
run_rank_plots(3,0)
run_rank_plots(4,0)
run_rank_plots(5,0)
run_rank_plots(6,0)
#14 threads gets to ~100%cpu
#n=50 is about 0.9sec/single_run including graphing
#8 cores ~60%cpu
#n=50 is about 1.23sec/single_run including graphing
