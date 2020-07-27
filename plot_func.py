#This module has 
#gamma_base_plot(smoothing_len,outputpath,mode):
#error_bar_plot(smoothing_len,outputpath) <- THIS IS PREFERRED

def gamma_base_plot(smoothing_len,run_folder,mode):
    #mode = 'scat' or 'line'
    #smoothing length reduces the length of the data by smoothing_len-fold. This is done by pari/triple/etc.-wise arthmetic averaging
    #will save the figures in outputpath/figures
    
    #Will deduce from filenames which simulation is log which is linear. 
    #plot all the log simulations in log, the linear simulations in linear
    #plot a hybrid graph showing both log and linear simulations on both log and linear scales
    #4 plots in total
    import sys
    import os
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import execution_func as ef
    import re
    
    outputpath = '/home/azhou/IA-Sim-AZhou00/IA_Numeric_Output'
    outputpath = os.path.join(outputpath,run_folder)
    
    searchpath = os.path.join(outputpath,'gamma_plus')
    scat = False
    if mode == 'scat':
        scat = True
    path_bar_search = re.search('baR_0 ([+-]?([0-9]*[.])?[0-9]+)', outputpath, re.IGNORECASE)
    baR_0 = float(path_bar_search.group(1))
    
    #check which plots are possible to make
    LOGSIMFLAG = False
    LINEARSIMFLAG = False
    HYBRIDFLAG = False
    if os.path.isfile(os.path.join(outputpath,'rslogmod')):
        LOGSIMFLAG = True
    if os.path.isfile(os.path.join(outputpath,'rslinearmod')):
        LINEARSIMFLAG = True
    if (LOGSIMFLAG == True and LINEARSIMFLAG == True): HYBRIDFLAG = True
    if (LOGSIMFLAG == False and LINEARSIMFLAG == False): sys.exit(['no radial coordinate file detected'])
    
    #consistency checks:
    #1. the linear and log rs should have the same length. this should agree with the n number in the output path
        #the rs file we grab from the run_folder has length n by construction
    #2. the smoothing length needs to be a factor of n
    path_n_search = re.search('n (\d+)', outputpath, re.IGNORECASE)
    n = int(path_n_search.group(1))
    if LOGSIMFLAG == True:
        rs_presmoothlog=ef.read_file_from_name(outputpath,'rslogmod')
        if (n != len(rs_presmoothlog)):
            sys.exit(['error(s) in log radial vector(s)!!!'])
    if LINEARSIMFLAG == True:
        rs_presmoothlinear=ef.read_file_from_name(outputpath,'rslinearmod')
        if (n != len(rs_presmoothlinear)):
            sys.exit(['error(s) in linear radial vector(s)!!!'])
    if HYBRIDFLAG == True:
        if ((n != len(rs_presmoothlog)) or (n != len(rs_presmoothlinear)) or (len(rs_presmoothlog)!= len(rs_presmoothlinear))):
            sys.exit(['in consistencies among radial vector(s)!!!'])
    if (n%smoothing_len != 0):
        sys.exit(['incompatible smoothing length!!!(smoothing length needs to be a factor of n)'])

    #reference curves setups
    asymp = np.array([])
    asymp_value = float((1-(baR_0)**2)/(1+(baR_0)**2)) #asymptotic value assuming perfect alignment. Used as a reference curve
    for i in range(int(n/smoothing_len)):
        asymp = np.append(asymp,asymp_value)
    y_0 = np.zeros(int(n/smoothing_len))
    
    #LOG PLOT:
    if LOGSIMFLAG == True:
        rs_log = np.array([sum(rs_presmoothlog[i:i+smoothing_len])/smoothing_len for i in range(0,len(rs_presmoothlog),smoothing_len)])   
        #get all the log files
        gamma_log = np.empty((0,int(n/smoothing_len)))
        for i in os.listdir(searchpath):
            if i.endswith("logmod"):
                if os.path.isfile(os.path.join(searchpath, i)): #check if it is file and not directory
                    temp_gam = ef.read_file_from_name(searchpath,i)
                    temp_gam_smoothed = np.array([sum(temp_gam[i:i+smoothing_len])/smoothing_len for i in range(0,len(temp_gam),smoothing_len)])   
                    gamma_log = np.vstack((gamma_log,temp_gam_smoothed))
        #plot:
        fig = plt.figure(figsize=(9,6))
        for g in gamma_log:
            if scat == True:  plt.scatter(rs_log,g,c='r',s=0.75,marker='o')
            if scat == False: plt.plot(rs_log,g)
        plt.plot(rs_log,asymp,"--",label=("asymptotic value %1.3f, b/a_0 = %1.2f" %(asymp_value, baR_0)))
        plt.plot(rs_log,y_0,label='y=0')
        plt.xscale('log')
        plt.xlabel('r')
        plt.ylabel('gamma +')
        plt.grid(b=True,which='both')
        plt.title(run_folder +' smooth=%i'% int(smoothing_len))
        plt.xlim(0.01,1)
        plt.ylim(0,1)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
        imagesavepath = os.path.join(outputpath, 'figures')
        if not os.path.exists(imagesavepath):
            os.makedirs(imagesavepath)
        if scat == True:
            fig.savefig(os.path.join(imagesavepath, 'Gamma '+str(n)+' smth %i scatplot log'%smoothing_len), bbox_inches = 'tight')
        if scat == False:
            fig.savefig(os.path.join(imagesavepath, 'Gamma '+str(n)+' smth %i lineplot log'%smoothing_len), bbox_inches = 'tight')

    #LINEAR PLOT:
    if LINEARSIMFLAG == True:
        rs_linear = np.array([sum(rs_presmoothlinear[i:i+smoothing_len])/smoothing_len for i in range(0,len(rs_presmoothlinear),smoothing_len)])   
        #get all the linear files
        gamma_linear = np.empty((0,int(n/smoothing_len)))
        for i in os.listdir(searchpath):
            if i.endswith("linearmod"):
                if os.path.isfile(os.path.join(searchpath, i)): #check if it is file and not directory
                    temp_gam = ef.read_file_from_name(searchpath,i)
                    temp_gam_smoothed = np.array([sum(temp_gam[i:i+smoothing_len])/smoothing_len for i in range(0,len(temp_gam),smoothing_len)])   
                    gamma_linear = np.vstack((gamma_linear,temp_gam_smoothed))
        #plot:
        fig2 = plt.figure(figsize=(9,6))
        for g in gamma_linear:
            if scat == True:  plt.scatter(rs_linear,g,c='r',s=0.75,marker='o')
            if scat == False: plt.plot(rs_linear,g)
        plt.plot(rs_linear,asymp,"--",label=("asymptotic value %1.3f, b/a_0 = %1.2f" %(asymp_value, baR_0)))
        plt.plot(rs_linear,y_0,label='y=0')
        plt.xlabel('r')
        plt.ylabel('gamma +')
        plt.grid(b=True,which='both')
        plt.title(run_folder + ' smooth=%i'% int(smoothing_len))
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
        imagesavepath = os.path.join(outputpath, 'figures')
        if not os.path.exists(imagesavepath):
            os.makedirs(imagesavepath)
        if scat == True:
            fig2.savefig(os.path.join(imagesavepath, 'Gamma '+str(n)+' smth %i scatplot linear'%smoothing_len), bbox_inches = 'tight')  
        if scat == False:
            fig2.savefig(os.path.join(imagesavepath, 'Gamma '+str(n)+' smth %i lineplot linear'%smoothing_len), bbox_inches = 'tight')

    #Hybrid PLOT:  
    if HYBRIDFLAG == True:
        #HybridLOG PLOT:
        fig3 = plt.figure(figsize=(9,6))
        for g in gamma_log:
            if scat == True:  plt.scatter(rs_log,g,c='r',s=0.75,marker='o')
            if scat == False: plt.plot(rs_log,g,c='r')
        for g in gamma_linear:
            if scat == True:  plt.scatter(rs_linear,g,c='b',s=0.75,marker='o')
            if scat == False: plt.plot(rs_linear,g,c='b')        
        plt.plot(rs_log,asymp,"--",label=("asymptotic value %1.3f, b/a_0 = %1.2f" %(asymp_value, baR_0)))
        plt.plot(rs_log,y_0,label='y=0')
        plt.xscale('log')
        plt.xlabel('r')
        plt.ylabel('gamma +')
        plt.grid(b=True,which='both')
        plt.title(run_folder+'\n'+'red:log sim; blue:linear sim\n'+'n='+str(n)+' smooth=%i'% int(smoothing_len))
        plt.xlim(0.01,1)
        plt.ylim(0,1)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
        imagesavepath = os.path.join(outputpath, 'figures')
        if not os.path.exists(imagesavepath):
            os.makedirs(imagesavepath)
        if scat == True:
            fig3.savefig(os.path.join(imagesavepath, 'hybrid Gamma '+str(n)+' smth %i scatplot log'%smoothing_len), bbox_inches = 'tight')
        if scat == False:
            fig3.savefig(os.path.join(imagesavepath, 'hybrid Gamma '+str(n)+' smth %i lineplot log'%smoothing_len), bbox_inches = 'tight')

        #HybridLOG PLOT:
        fig4 = plt.figure(figsize=(9,6))
        for g in gamma_log:
            if scat == True:  plt.scatter(rs_log,g,c='r',s=0.75,marker='o')
            if scat == False: plt.plot(rs_log,g,c='r')
        for g in gamma_linear:
            if scat == True:  plt.scatter(rs_linear,g,c='b',s=0.75,marker='o')
            if scat == False: plt.plot(rs_linear,g,c='b')   
        plt.plot(rs_linear,asymp,"--",label=("asymptotic value %1.3f, b/a_0 = %1.2f" %(asymp_value, baR_0)))
        plt.plot(rs_linear,y_0,label='y=0')
        plt.xlabel('r')
        plt.ylabel('gamma +')
        plt.grid(b=True,which='both')
        plt.title(run_folder+'\n'+'red:log sim; blue:linear sim\n'+'n='+str(n)+' smooth=%i'% int(smoothing_len))
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
        imagesavepath = os.path.join(outputpath, 'figures')
        if not os.path.exists(imagesavepath):
            os.makedirs(imagesavepath)
        if scat == True:
            fig4.savefig(os.path.join(imagesavepath, 'hybrid Gamma '+str(n)+' smth %i scatplot linear'%smoothing_len), bbox_inches = 'tight')  
        if scat == False:
            fig4.savefig(os.path.join(imagesavepath, 'hybrid Gamma '+str(n)+' smth %i lineplot linear'%smoothing_len), bbox_inches = 'tight')
    
    
    
    
    
    
#The error bar function
#in each folder of Gamma_plus_some_qualifiers, there are data of multiple runs on the same settinf
#so first we want to navigate to that folder via search_path, and read all the Gamma_plus_files
#This is analogous of the ploting functions above
#This function searches for all the files in a folder. The files should all contain 1-dim np.array. 
#It applies smoothing first, then
#It takes the two lines that signifies the 1 and 2 th STD upper&lower bounds
#and return these 4 lines in one 2-D file that contain both lists.

def error_bar_plot(smoothing_len,run_folder): 
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
    
    outputpath = '/home/azhou/IA-Sim-AZhou00/IA_Numeric_Output'
    outputpath = os.path.join(outputpath,run_folder)
    
    searchpath = os.path.join(outputpath,'gamma_plus')
    path_bar_search = re.search('baR_0 ([+-]?([0-9]*[.])?[0-9]+)', outputpath, re.IGNORECASE)
    baR_0 = float(path_bar_search.group(1))
    
    #check which plots are possible to make
    LOGSIMFLAG = False
    LINEARSIMFLAG = False
    HYBRIDFLAG = False
    if os.path.isfile(os.path.join(outputpath,'rslogmod')):
        LOGSIMFLAG = True
    if os.path.isfile(os.path.join(outputpath,'rslinearmod')):
        LINEARSIMFLAG = True
    if (LOGSIMFLAG == True and LINEARSIMFLAG == True): HYBRIDFLAG = True
    if (LOGSIMFLAG == False and LINEARSIMFLAG == False): sys.exit(['no radial coordinate file detected'])
    
    #consistency checks:
    #1. the linear and log rs should have the same length. this should agree with the n number in the output path
        #the rs file we grab from the run_folder has length n by construction
    #2. the smoothing length needs to be a factor of n
    path_n_search = re.search('n (\d+)', outputpath, re.IGNORECASE)
    n = int(path_n_search.group(1))
    if LOGSIMFLAG == True:
        rs_presmoothlog=ef.read_file_from_name(outputpath,'rslogmod')
        if (n != len(rs_presmoothlog)):
            sys.exit(['error(s) in log radial vector(s)!!!'])
    if LINEARSIMFLAG == True:
        rs_presmoothlinear=ef.read_file_from_name(outputpath,'rslinearmod')
        if (n != len(rs_presmoothlinear)):
            sys.exit(['error(s) in linear radial vector(s)!!!'])
    if HYBRIDFLAG == True:
        if ((n != len(rs_presmoothlog)) or (n != len(rs_presmoothlinear)) or (len(rs_presmoothlog)!= len(rs_presmoothlinear))):
            sys.exit(['in consistencies among radial vector(s)!!!'])
    if (n%smoothing_len != 0):
        sys.exit(['incompatible smoothing length!!!(smoothing length needs to be a factor of n)'])

    #reference curves setups
    asymp = np.array([])
    asymp_value = float((1-(baR_0)**2)/(1+(baR_0)**2)) #asymptotic value assuming perfect alignment. Used as a reference curve
    for i in range(int(n/smoothing_len)):
        asymp = np.append(asymp,asymp_value)
    y_0 = np.zeros(int(n/smoothing_len))
    
    #LOG PLOT:
    if LOGSIMFLAG == True:
        rs_log = np.array([sum(rs_presmoothlog[i:i+smoothing_len])/smoothing_len for i in range(0,len(rs_presmoothlog),smoothing_len)])   
        #get all the log files
        gamma_log = np.empty((0,int(n/smoothing_len)))
        for i in os.listdir(searchpath):
            if i.endswith("logmod"):
                if os.path.isfile(os.path.join(searchpath, i)): #check if it is file and not directory
                    temp_gam = ef.read_file_from_name(searchpath,i)
                    temp_gam_smoothed = np.array([sum(temp_gam[i:i+smoothing_len])/smoothing_len for i in range(0,len(temp_gam),smoothing_len)])   
                    gamma_log = np.vstack((gamma_log,temp_gam_smoothed))
        error_GamPls_log = ef.sort_matrix_columns(gamma_log)
        #print(complete_GamPls)
        error_GamPls_log = ef.get_1_2_std(error_GamPls_log)
        #print(complete_GamPls)
        error_GamPls_log = np.vstack((error_GamPls_log,rs_log)) #attach the smoothed x coordinate
        #save in the figures folder
        ef.write_file_at_path(outputpath, 'figures',error_GamPls_log,'STD smth %i log'%smoothing_len)

        #plot:
        fig1 = plt.figure(figsize=(9,6))
        plt.fill_between(rs_log,error_GamPls_log[2],error_GamPls_log[3],alpha = 0.4,label='2nd STD')    
        plt.fill_between(rs_log,error_GamPls_log[0],error_GamPls_log[1],alpha = 0.4,label='1st STD')
        for g in gamma_log:
            plt.scatter(rs_log,g,c='r',s=0.75,marker='o')        
        plt.plot(rs_log,asymp,"--",label=("asymptotic value %1.3f, b/a_0 = %1.2f" %(asymp_value, baR_0)))
        plt.plot(rs_log,y_0,label='y=0')
        plt.xscale('log')
        plt.xlabel('r')
        plt.ylabel('gamma +')
        plt.grid(b=True,which='both')
        plt.title(run_folder+' smooth=%i'% int(smoothing_len))
        plt.xlim(0.01,1)
        plt.ylim(0,1)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
        imagesavepath = os.path.join(outputpath, 'figures')
        if not os.path.exists(imagesavepath):
            os.makedirs(imagesavepath)
        fig1.savefig(os.path.join(imagesavepath, 'STD Gamma '+str(n)+' smth %i log'%smoothing_len), bbox_inches = 'tight')

    #LINEAR PLOT:
    if LINEARSIMFLAG == True:
        rs_linear = np.array([sum(rs_presmoothlinear[i:i+smoothing_len])/smoothing_len for i in range(0,len(rs_presmoothlinear),smoothing_len)])   
        #get all the log files
        gamma_linear = np.empty((0,int(n/smoothing_len)))
        for i in os.listdir(searchpath):
            if i.endswith("linearmod"):
                if os.path.isfile(os.path.join(searchpath, i)): #check if it is file and not directory
                    temp_gam = ef.read_file_from_name(searchpath,i)
                    temp_gam_smoothed = np.array([sum(temp_gam[i:i+smoothing_len])/smoothing_len for i in range(0,len(temp_gam),smoothing_len)])   
                    gamma_linear = np.vstack((gamma_linear,temp_gam_smoothed))
        error_GamPls_linear = ef.sort_matrix_columns(gamma_linear)
        #print(complete_GamPls)
        error_GamPls_linear = ef.get_1_2_std(error_GamPls_linear)
        #print(complete_GamPls)
        error_GamPls_linear = np.vstack((error_GamPls_linear,rs_linear)) #attach the smoothed x coordinate
        #save in the figures folder
        ef.write_file_at_path(outputpath, 'figures',error_GamPls_linear,'STD smth %i linear'%smoothing_len)

        #plot:
        fig2 = plt.figure(figsize=(9,6))
        plt.fill_between(rs_linear,error_GamPls_linear[2],error_GamPls_linear[3],alpha = 0.4,label='2nd STD')    
        plt.fill_between(rs_linear,error_GamPls_linear[0],error_GamPls_linear[1],alpha = 0.4,label='1st STD')
        for g in gamma_linear:
            plt.scatter(rs_linear,g,c='r',s=0.75,marker='o')        
        plt.plot(rs_linear,asymp,"--",label=("asymptotic value %1.3f, b/a_0 = %1.2f" %(asymp_value, baR_0)))
        plt.plot(rs_linear,y_0,label='y=0')
        plt.xlabel('r')
        plt.ylabel('gamma +')
        plt.grid(b=True,which='both')
        plt.title(run_folder+' smooth=%i'% int(smoothing_len))
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
        imagesavepath = os.path.join(outputpath, 'figures')
        if not os.path.exists(imagesavepath):
            os.makedirs(imagesavepath)
        fig2.savefig(os.path.join(imagesavepath, 'STD Gamma '+str(n)+' smth %i linear'%smoothing_len), bbox_inches = 'tight')            
            
    #Hybrid PLOT:  
    if HYBRIDFLAG == True:
    #HybridLOG PLOT:
        f, (ax1,ax2) = plt.subplots(1,2,figsize=(8,10))
        for g in gamma_log:
            ax1.scatter(rs_log,g,c='r',s=0.75,marker='o')
        for g in gamma_linear:
            ax1.scatter(rs_linear,g,c='b',s=0.75,marker='o')
        ax1.fill_between(rs_log,error_GamPls_log[2],error_GamPls_log[3],alpha = 0.2,label='log_2nd STD')    
        ax1.fill_between(rs_log,error_GamPls_log[0],error_GamPls_log[1],alpha = 0.6,label='log_1st STD')
        ax1.fill_between(rs_linear,error_GamPls_linear[2],error_GamPls_linear[3],alpha = 0.2,label='linear_2nd STD')    
        ax1.fill_between(rs_linear,error_GamPls_linear[0],error_GamPls_linear[1],alpha = 0.6,label='linear_1st STD')
        ax1.plot(rs_log,asymp,"--",label=("asymptotic value %1.3f, b/a_0 = %1.2f" %(asymp_value, baR_0)))
        ax1.plot(rs_log,y_0,label='y=0')
        ax1.set_xscale('log')
        ax1.set_xlabel('r')
        ax1.set_ylabel('gamma +')
        ax1.grid(b=True,which='both')
        ax1.set_xlim(0.01,1)
        ax1.set_ylim(0,0.4)
        #ax1.legend(bbox_to_anchor=(0.5, 0.1), loc='lower left', ncol=1)
        #HybridLINEAR PLOT:
        for g in gamma_log:
            ax2.scatter(rs_log,g,c='r',s=0.75,marker='o')
        for g in gamma_linear:
            ax2.scatter(rs_linear,g,c='b',s=0.75,marker='o')
        ax2.fill_between(rs_log,error_GamPls_log[2],error_GamPls_log[3],alpha = 0.2,label='log_2nd STD')    
        ax2.fill_between(rs_log,error_GamPls_log[0],error_GamPls_log[1],alpha = 0.6,label='log_1st STD')
        ax2.fill_between(rs_linear,error_GamPls_linear[2],error_GamPls_linear[3],alpha = 0.2,label='linear_2nd STD')    
        ax2.fill_between(rs_linear,error_GamPls_linear[0],error_GamPls_linear[1],alpha = 0.6,label='linear_1st STD')
        ax2.plot(rs_linear,asymp,"--",label=("asymptotic value %1.3f, b/a_0 = %1.2f" %(asymp_value, baR_0)))
        ax2.plot(rs_linear,y_0,label='y=0')
        ax2.set_xlabel('r')
        ax2.set_ylabel('gamma +')
        ax2.grid(b=True,which='both')
        ax2.set_xlim(0,1)
        ax2.set_ylim(0,0.4)
        #ax2.legend(bbox_to_anchor=(0.5, 0.1), loc='lower left', ncol=1)
        
        f.suptitle(run_folder+'\n'+'red:log sim; blue:linear sim\n'+'n='+str(n)+' smooth=%i'% int(smoothing_len))
        imagesavepath = os.path.join(outputpath, 'figures')
        if not os.path.exists(imagesavepath):
            os.makedirs(imagesavepath)
        f.savefig(os.path.join(imagesavepath, 'hybrid STD Gamma '+str(n)+' smth %i '%smoothing_len), bbox_inches = 'tight')  
