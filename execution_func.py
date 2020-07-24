def get_axis_lin(n):
    #this gives a sorted (ascending) linear partition from -1 to 1, in a list of 2n+1 elmts
    #0 always in the list, this prevent computing singular values
    import numpy as np
    return np.linspace(-1,1,n*2+1)

def get_axis_log(n):
    #this gives a sorted (ascending) log partition from -1 to 1, in a list of 2n+1 elmts
    #0 always in the list, this prevent computing singular values
    import numpy as np
    log_axis = np.append(np.logspace(-2,0,n),-1*np.logspace(-2,0,n))
    log_axis = np.append(log_axis,0)
    return np.sort(log_axis)

def get_mid_points(ls):
    import numpy as np
    return np.array([(ls[i]+ls[i+1])/2 for i in range(0,len(ls)-1,1)])

def read_file_from_name(path,filename):
    import os
    import numpy as np
    filepath = os.path.join(path, filename)
    file = open(filepath, "rb")
    var = np.load(file)
    file.close
    return var
#syntax a = read_file_from_name(path,filename)

def write_file_at_path(path_output, decorated_name, object_name,filename):
    ##write 'numpy'object_name to file as path_output/decorated_name/iteration_tracker.format
    ##if decorated_name = 'NA', no subfolder will be created, saving directly to path_output
    import numpy as np
    import os
    if decorated_name != 'NA':
        fullpath = os.path.join(path_output, decorated_name)
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)
            print('made path', fullpath)
        filename = os.path.join(fullpath, filename)
        file = open(filename, "wb")
        np.save(file, object_name)
        file.close
    if decorated_name == 'NA':
        fullpath = path_output
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)
            print('made path', fullpath)
        filename = os.path.join(fullpath, filename)
        file = open(filename, "wb")
        np.save(file, object_name)
        file.close

#deprecated. If want to use iteration as file name, just do str(iteration_tracker)
#def write_namedfile_at_path(path_output, decorated_name, object_name,name):
#    #write 'numpy' object_name to file as path_output/decorated_name/name.format
#    ##if decorated_name = 'NA', no subfolder will be created, saving directly to path_output
#    import numpy as np
#    import os
#    if decorated_name != 'NA':
#        fullpath = os.path.join(path_output, decorated_name)
#        if not os.path.exists(fullpath):
#            os.makedirs(fullpath)
#            print('made path', fullpath)
#        filename = name
#        filename = os.path.join(fullpath, filename)
#        file = open(filename, "wb")
#        np.save(file, object_name)
#        file.close
#    if decorated_name == 'NA':
#        fullpath = path_output
#        if not os.path.exists(fullpath):
#            os.makedirs(fullpath)
#            print('made path', fullpath)
#        filename = name
#        filename = os.path.join(fullpath, filename)
#        file = open(filename, "wb")
#        np.save(file, object_name)
#        file.close

def sort_matrix_columns(matrix):
    #smallest entries on top
    import numpy as np
    return np.sort(matrix,axis=0)

def get_1_2_std(sorted_matrix): 
    #the input matrix has to be sorted in each column
    #the returned object is an np.array 
    #[
    #[y value of  std 1 upperbound],
    #[y value of  std 1 lowerbound],
    #[y value of  std 2 upperbound],
    #[y value of  std 2 lowerbound]
    #]
    
    #by definition
    #_|_13.6_|__34.1__||__34.1__|_13.6_|_
    #from top row of the matrix
    #%2.3 ->lowerbound std2
    #%15.9->lowerbound std1
    #%84.1->upperbound std1
    #%97.7->upperbound std1
    #%2.3 ->bottom row of the matrix
    #IF IN THIS ORDER:
    #np.floor(sample_size*0.023)
    #np.floor(sample_size*0.159)
    #np.ceil(sample_size*0.841)
    #np.ceil(sample_size*0.977)
    #TO CHANGE THE ORDER AS DESIRED IN THE FIRST SECTION WE HAVE:
    import numpy as np
    sample_size = (sorted_matrix.shape)[0] #how many rows
    #make sure index doesnt go out of bounds, or become negative
    u1 = np.min([sample_size-1,int(np.ceil(sample_size*0.841))-1])
    l1 = np.max([0,int(np.floor(sample_size*0.159))-1])
    u2 = np.min([sample_size-1,int(np.ceil(sample_size*0.977))-1])
    l2 = np.max([0,int(np.floor(sample_size*0.023))-1])
    return np.array([
        sorted_matrix[u1],
        sorted_matrix[l1],
        sorted_matrix[u2],
        sorted_matrix[l2]
    ])
#tests:
#import numpy as np
#a=np.array(
#[  0,   4,   0,3],
#[  1,   5,   5,4],
#[  3, 199,   9,0],
#[  4,   5,   5,4],
#[  5, 199,   9,0],
#[  7,   5,   5,4],
#[  6, 199,   9,0],
#[  10,   5,   5,4],
#[  9,   5,   5,4],
#[  8, 199,   9,0]])
#a= sort_matrix_columns(a)
#print(a)
#print((a.shape)[0])
#a = np.empty((0,4))
#for i in range(100):
#    a = np.vstack((a,[i+1,i+101,(i+1)*0.1,-(i+1)]))
##print(a)
#print(get_1_2_std(a))
