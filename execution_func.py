def write_file_at_path(path_output, decorated_name, object_name,iteration_tracker):
    #write file at path/decoratedname/iteration_tracker.format
    import numpy as np
    import os
    #write 'numpy'object_name to file using interation_number as name in path_output/decorated_name/
    fullpath = os.path.join(path_output, decorated_name)
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)
        print('made path', fullpath)
    filename = str(iteration_tracker)
    filename = os.path.join(fullpath, filename)
    file = open(filename, "wb")
    np.save(file, object_name)
    file.close

def write_namedfile_at_path(path_output, decorated_name, object_name,name):
    #write file at path/decoratedname/name.format
    import numpy as np
    import os
    #write 'numpy'object_name to file using interation_number as name in path_output/decorated_name/
    fullpath = os.path.join(path_output, decorated_name)
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)
        print('made path', fullpath)
    filename = name
    filename = os.path.join(fullpath, filename)
    file = open(filename, "wb")
    np.save(file, object_name)
    file.close

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
