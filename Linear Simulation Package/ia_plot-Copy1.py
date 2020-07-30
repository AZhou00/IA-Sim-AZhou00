#This module has 
#gamma_base_plot(smoothing_len,outputpath,mode):
#error_bar_plot(smoothing_len,outputpath) <- THIS IS PREFERRED
    
#The error bar function
#in each folder of Gamma_plus_some_qualifiers, there are data of multiple runs on the same setting
#so first we want to navigate to that folder via search_path, and read all the Gamma_plus_files
#This function searches for all the files in a folder. The files should all contain version two data structure. 
#It applies smoothing first, then
#It takes the two lines that signifies the 1 and 2 th STD upper&lower bounds
#and return these 4 lines in one 2-D file that contain both lists.

def error_bar_plot(smoothing_len,run_folder,SCATFLAG): 
    "scatflag decides if you want the scatter plot overlaying the error bar plot"
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
    
    import sys
    import os
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import execution_func as ef
    import re
    
    

    outputpath = '/home/azhou/IA-Sim-AZhou00/IA_Numeric_Output/'
    outputpath = os.path.join(outputpath,run_folder)
    searchpath = os.path.join(outputpath,'gamma_plus/')
    
    path_bar_search = re.search('baR_0 ([+-]?([0-9]*[.])?[0-9]+)', outputpath, re.IGNORECASE)
    baR_0 = float(path_bar_search.group(1))
    
    #consistency checks:
    #1. the n in file name should  = the stored array length in each dimension
    #2. the smoothing length needs to be a factor of n
    path_n_search = re.search('n (\d+)', outputpath, re.IGNORECASE)
    n = int(path_n_search.group(1))
    rs_check=ef.read_file_from_name(searchpath,'0linearmod')
    if (n != len(rs_check[1,:])):
        sys.exit(['error(s) in log radial vector(s)!!!'])
    #rs_check=ef.read_file_from_name(searchpath,'0logmod')
    #if (n != len(rs_check[1,:])):
    #    sys.exit(['error(s) in log radial vector(s)!!!'])

    
    
    

    
    #LOG CURVES STD
    searchpath = os.path.join(outputpath,'gamma_plus/')
    gamma_log = np.empty((0,int(n/smoothing_len)))
    for i in os.listdir(searchpath):
        if i.endswith("logmod"):
            if os.path.isfile(os.path.join(searchpath, i)): #check if it is file and not directory
                [temp_r,temp_gam] = ef.read_file_from_name(searchpath,i)
                temp_gam_smoothed = np.array([sum(temp_gam[i:i+smoothing_len])/smoothing_len for i in range(0,len(temp_gam),smoothing_len)])
                temp_r_smoothed = np.array([sum(temp_r[i:i+smoothing_len])/smoothing_len for i in range(0,len(temp_r),smoothing_len)])
                gamma_log = np.vstack((gamma_log,temp_gam_smoothed))
    error_GamPls_log = ef.sort_matrix_columns(gamma_log)
    #print(complete_GamPls)
    error_GamPls_log = ef.get_1_2_std(error_GamPls_log)
    #print(complete_GamPls)
    error_GamPls_log = np.vstack((error_GamPls_log,temp_r_smoothed)) #attach the smoothed x coordinate
    #save in the figures folder
    ef.write_file_at_path(outputpath, 'STD',error_GamPls_log,'gamma smth %i log'%smoothing_len)
    
    searchpath = os.path.join(outputpath,'densities/')
    density_log = np.empty((0,int(n/smoothing_len)))
    for i in os.listdir(searchpath):
        if i.endswith("logmod"):
            if os.path.isfile(os.path.join(searchpath, i)): #check if it is file and not directory
                [temp_r,temp_gam] = ef.read_file_from_name(searchpath,i)
                temp_gam_smoothed = np.array([sum(temp_gam[i:i+smoothing_len])/smoothing_len for i in range(0,len(temp_gam),smoothing_len)])
                temp_r_smoothed = np.array([sum(temp_r[i:i+smoothing_len])/smoothing_len for i in range(0,len(temp_r),smoothing_len)])
                density_log = np.vstack((density_log,temp_gam_smoothed))
    error_density_log = ef.sort_matrix_columns(density_log)
    #print(complete_GamPls)
    error_density_log = ef.get_1_2_std(error_density_log)
    #print(complete_GamPls)
    error_density_log = np.vstack((error_density_log,temp_r_smoothed)) #attach the smoothed x coordinate
    #save in the figures folder
    ef.write_file_at_path(outputpath, 'STD',error_density_log,'density smth %i log'%smoothing_len)
    
    #LINEAR CURVES STD
    searchpath = os.path.join(outputpath,'gamma_plus/')
    gamma_linear = np.empty((0,int(n/smoothing_len)))
    for i in os.listdir(searchpath):
        if i.endswith("linearmod"):
            if os.path.isfile(os.path.join(searchpath, i)): #check if it is file and not directory
                [temp_r,temp_gam] = ef.read_file_from_name(searchpath,i)
                temp_gam_smoothed = np.array([sum(temp_gam[i:i+smoothing_len])/smoothing_len for i in range(0,len(temp_gam),smoothing_len)])
                temp_r_smoothed = np.array([sum(temp_r[i:i+smoothing_len])/smoothing_len for i in range(0,len(temp_r),smoothing_len)])
                gamma_linear = np.vstack((gamma_linear,temp_gam_smoothed))
    error_GamPls_linear = ef.sort_matrix_columns(gamma_linear)
    #print(complete_GamPls)
    error_GamPls_linear = ef.get_1_2_std(error_GamPls_linear)
    #print(complete_GamPls)
    error_GamPls_linear = np.vstack((error_GamPls_linear,temp_r_smoothed)) #attach the smoothed x coordinate
    #save in the figures folder
    ef.write_file_at_path(outputpath, 'STD',error_GamPls_linear,'gamma smth %i linear'%smoothing_len)
    
    searchpath = os.path.join(outputpath,'densities/')
    density_linear = np.empty((0,int(n/smoothing_len)))
    for i in os.listdir(searchpath):
        if i.endswith("linearmod"):
            if os.path.isfile(os.path.join(searchpath, i)): #check if it is file and not directory
                [temp_r,temp_gam] = ef.read_file_from_name(searchpath,i)
                temp_gam_smoothed = np.array([sum(temp_gam[i:i+smoothing_len])/smoothing_len for i in range(0,len(temp_gam),smoothing_len)])
                temp_r_smoothed = np.array([sum(temp_r[i:i+smoothing_len])/smoothing_len for i in range(0,len(temp_r),smoothing_len)])
                density_linear = np.vstack((density_linear,temp_gam_smoothed))
    error_density_linear = ef.sort_matrix_columns(density_linear)
    #print(complete_GamPls)
    error_density_linear = ef.get_1_2_std(error_density_linear)
    #print(complete_GamPls)
    error_density_linear = np.vstack((error_density_linear,temp_r_smoothed)) #attach the smoothed x coordinate
    #save in the figures folder
    ef.write_file_at_path(outputpath, 'STD',error_density_linear,'density smth %i linear'%smoothing_len)

    
    
    
    
    #reference curves setups
    def integrateNFW(r,c,N): #spherical cutoff, z direction integration
        a=np.sqrt(1-r**2)
        z = np.linspace(-a, a, N)
        tempv = np.sqrt(r**2+z**2)
        fz = 1/(tempv*c*(1+c*tempv)**2)
        area = np.sum(fz)*2*a/N
        return area
    def get_ref_density(conc,ref_r):
        # ref_r can be in either log or linear
        arr_c = np.array([]) #plotted only in linear for simplicity, since this is just a reference curve
        totalmass = 4.0*np.pi*(np.log(1+conc)-1+1/(1+conc))/(conc**3) #with max radial coordinate normalized to 1
        print(conc,totalmass)
        for i in range(int(n/smoothing_len-1)):
            radius = ref_r[i]
            temp_=2*np.pi*radius*integrateNFW(radius,conc,200)/(totalmass) #2*np.pi*radius*
            arr_c=np.append(arr_c,temp_)
        return arr_c
#    def integrateNFW_R(ri,rf,c,N_z,N_r=10):
#        r_interval = np.linspace(ri, rf, N_r)
#        evaluatedlist = np.empty((0))
#        for rad in r_interval:
#            evaluatedlist = np.append(evaluatedlist,rad*integrateNFW(rad,c,N_z))
#        area = np.sum(evaluatedlist)*(rf-ri)/N_r
#        return area
    c5 = get_ref_density(5.,error_GamPls_log[4])
    c10 = get_ref_density(10.,error_GamPls_log[4])
    c20 = get_ref_density(20.,error_GamPls_log[4])
    c30 = get_ref_density(30.,error_GamPls_log[4])
    c40 = get_ref_density(40.,error_GamPls_log[4])
    

    def integrateCONST(r,N): #spherical cutoff, z direction integration
        a=np.sqrt(1-r**2)
        z = np.linspace(-a, a, N)
        fz = z*0+3/(4*np.pi)
        area = np.sum(fz)*2*a/N
        return area
    def get_ref_density_CONST(ref_r):
        # ref_r can be in either log or linear
        arr_c = np.array([]) #plotted only in linear for simplicity, since this is just a reference curve
        for i in range(int(n/smoothing_len-1)):
            radius = ref_r[i]
            temp_=2*np.pi*radius*integrateCONST(radius,200)
            arr_c=np.append(arr_c,temp_)
        return arr_c
    constant_rho = get_ref_density_CONST(error_GamPls_log[4])

    asymp = np.array([])
    asymp_value = float((1-(baR_0)**2)/(1+(baR_0)**2)) #asymptotic value assuming perfect alignment. Used as a reference curve
    for i in range(int(n/smoothing_len)):
        asymp = np.append(asymp,asymp_value)
    y_0 = np.zeros(int(n/smoothing_len))    
    
    
    
    
    flog, (ax1,ax2) = plt.subplots(2,figsize=(12,8)) #vertically gamma, density
    #HybridLOG PLOT:
    if SCATFLAG == True:
        for g in gamma_log:
            ax1.scatter(error_GamPls_log[4],g,c='r',s=0.75,marker='o')
        for g in gamma_linear:
            ax1.scatter(error_GamPls_linear[4],g,c='b',s=0.75,marker='o')
    ax1.fill_between(error_GamPls_log[4],error_GamPls_log[2],error_GamPls_log[3],alpha = 0.3,label='log_2nd STD')    
    ax1.fill_between(error_GamPls_log[4],error_GamPls_log[0],error_GamPls_log[1],alpha = 0.4,label='log_1st STD')
    ax1.fill_between(error_GamPls_linear[4],error_GamPls_linear[2],error_GamPls_linear[3],alpha = 0.3,label='linear_2nd STD')    
    ax1.fill_between(error_GamPls_linear[4],error_GamPls_linear[0],error_GamPls_linear[1],alpha = 0.4,label='linear_1st STD')
    ax1.plot(error_GamPls_log[4],asymp,"--",label=("asymptotic value %1.3f, b/a_0 = %1.2f" %(asymp_value, baR_0)))
    ax1.plot(error_GamPls_log[4],y_0,label='y=0')
    ax1.set_title('mean epsilon + at r from BCG')
    ax1.set_xscale('log')
    ax1.set_xlabel('r')
    ax1.set_ylabel('gamma +')
    ax1.grid(b=True,which='both')
    ax1.set_xlim(0.01,1)
    #ax1.set_ylim(0,0.4)
    ax1.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    
    if SCATFLAG == True:
        for g in density_log:
            ax2.scatter(error_density_log[4],g,c='r',s=0.75,marker='o')
        for g in density_linear:
            ax2.scatter(error_density_linear[4],g,c='b',s=0.75,marker='o')
    ax2.fill_between(error_density_log[4],error_density_log[2],error_density_log[3],alpha = 0.3,label='log_2nd STD')    
    ax2.fill_between(error_density_log[4],error_density_log[0],error_density_log[1],alpha = 0.6,label='log_1st STD')
    ax2.fill_between(error_density_linear[4],error_density_linear[2],error_density_linear[3],alpha = 0.3,label='linear_2nd STD')    
    ax2.fill_between(error_density_linear[4],error_density_linear[0],error_density_linear[1],alpha = 0.6,label='linear_1st STD')
    ax2.plot(error_density_log[4][0:-1],constant_rho,label='constant density')
    ax2.plot(error_density_log[4][0:-1],c5,label='c5')
    ax2.plot(error_density_log[4][0:-1],c10,label='c10')
    ax2.plot(error_density_log[4][0:-1],c20,label='c20')
    ax2.plot(error_density_log[4][0:-1],c30,label='c30')
    ax2.plot(error_density_log[4][0:-1],c40,label='c40')
    ax2.set_title('linear density distribution')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('r')
    ax2.set_ylabel('density')
    ax2.grid(b=True,which='both')
    ax2.set_xlim(0.01,1)
    #ax1.set_ylim(0,0.4)
    ax2.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    flog.suptitle(run_folder+'\n'+'red:log sim; blue:linear sim. smoothing=%i \n'% int(smoothing_len))
    
    imagesavepath = os.path.join(outputpath, 'figures')
    if not os.path.exists(imagesavepath):
        os.makedirs(imagesavepath)
    flog.savefig(os.path.join(imagesavepath, 'hybrid STD g+density smth %i '%smoothing_len), bbox_inches = 'tight')  
