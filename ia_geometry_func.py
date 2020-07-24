def Cart_to_Sph(xyz): #input np array [[x,y,z]], output[[r,theta,phi]]
    import numpy as np
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew
#pts = np.array([[1,1,1]])
#print(Cart_to_Sph(pts))
#[[r,theta,phi]] = Cart_to_Sph(np.array([[1,1,1]]))
#print(r)#print(theta)#print(phi)

def RZ_to_Theta(xyz): #input np array [[r,z]], output theta
    import numpy as np
    return np.arctan2(xyz[:,0], xyz[:,1])[0] # for elevation angle defined from Z-axis down

def Cart_to_Polar(xy): #input np array [[x,y]], output[[r,phi]]
    import numpy as np
    ptsnew = np.zeros(xy.shape)
#    ptsnew = np.zeros((1,2))
    ptsnew[:,0] = np.sqrt(xy[:,0]**2+xy[:,1]**2)
    ptsnew[:,1] = np.arctan2(xy[:,1], xy[:,0]) #for elevation angle defined from Z-axis down
    return ptsnew

def Sph_to_Cart(xyz): #input np array [[r,theta,phi]], output[[x,y,z]]
    import numpy as np
    ptsnew = np.zeros(xyz.shape)
#    ptsnew = np.zeros((1,3))
    ptsnew[:,0] = xyz[:,0]*np.sin(xyz[:,1])*np.cos(xyz[:,2])
    ptsnew[:,1] = xyz[:,0]*np.sin(xyz[:,1])*np.sin(xyz[:,2])
    ptsnew[:,2] = xyz[:,0]*np.cos(xyz[:,1])
    return ptsnew
#Cartesian_np(np.array([[1.73205081, 0.95531662, 0.78539816]]))
#pts = np.array([[1,1,1]])
#Cartesian_np(Cart_to_Sph(pts))

def Polar_to_Cart(xy): #input np array [[r,phi]], output[[x,y]]
    import numpy as np
#    ptsnew = np.zeros((1,2))
    ptsnew = np.zeros(xy.shape)
    ptsnew[:,0] = xy[:,0]*np.cos(xy[:,1])
    ptsnew[:,1] = xy[:,0]*np.sin(xy[:,1])
    return ptsnew
#print(Cart_to_Polar(np.array([[-1,0]])))
#np.array([[1,1,1]]).shape

def Ellip_proj_mag(baR_mag,local_theta): # calculates the projected baR ratio
    import numpy as np
    #arctan2 takes (y,x)
    temp = np.arctan(baR_mag/np.tan(local_theta))
    #baR_mag/(np.sin(local_theta)*np.cos(temp)+baR_mag*np.cos(local_theta)*np.sin(temp))
    return baR_mag/(np.sin(local_theta)*np.cos(temp)+baR_mag*np.cos(local_theta)*np.sin(temp))

def Cart_to_eps(sat_state): 
    #input is a length 4 np.array, in cartesian coord, [[sat_location_x, sat_location_y,sat_orient_x,sat_orient_y]], output eps+
    import numpy as np
    sat_state_loc = sat_state[:,0:2] #[sat_location_x, sat_location_y]
    sat_state_orient = sat_state[:,2:4] #[sat_orient_x,sat_orient_y]
    temp_posangle = Cart_to_Polar(sat_state_loc)[:,1][0]
    [[temp_mag,temp_satangle]] = Cart_to_Polar(sat_state_orient)
    #temp_mag = Cart_to_Polar(sat_state_orient)[:,0][0]
    #temp_satangle = Cart_to_Polar(sat_state_orient)[:,1][0]
    return (1-temp_mag**2)/(1+temp_mag**2)*np.cos(2*(temp_satangle-temp_posangle))

def Polar_to_eps(sat_location_phi,sat_orient_r,sat_orient_phi): 
    #input is a length !!!3!!! np.array[sat_location_phi,sat_orient_r,sat_orient_phi], output eps+
    import numpy as np
    return (1-sat_orient_r**2)/(1+sat_orient_r**2)*np.cos(2*(sat_location_phi-sat_orient_phi))

def rad_to_deg(rad):
    import numpy as np
    return (rad/np.pi)*180
