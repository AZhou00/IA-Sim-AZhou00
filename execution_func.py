def write_file_at_path(path_output, decorated_name, object_name,iteration_tracker):
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
