import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import copy
import random
import scipy.stats as st
import shapely as sp
from shapely.ops import polygonize
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import multiprocessing as mp
import time
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
import ast
import matplotlib.pylab as pl
import json
import os
import re

def generate(shapes):
    '''Generates the desired shapes in 2D space. The shapes have no overlap and are an idealized version of the structure.
    Inputs:
        shapes: Dictionary object where first entry is the shape, second entry is patch thickness, and consecutive
            entries are the side lengths starting from the patch of the shape and rotating clockwise around the object
            example: shapes = {'shape 1':['s','A',10],'shape 2':['s','A',10]}
    Return:
        hinge_vec: Vector of hinge angles. Interdipolar angle between each shape
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
        sequence: List of strings describing the orientation of each shape
            Example: ['A','B','B','A']
        linelist: List of integers describing the number of sides in each shape
        patch_arr: 2x(2n) array of x and y points that describe the lines of the magentic patch. n is number of distinct magnetic domains'''
    
    #Separate the origin into x and y components. Note that the origin will be updated with the addition of each shape
    origin = [0,0] #the origin is at 0,0
    xi = origin[0] #Initialize the x-coordiante of the initial point for each shape
    yi = origin[1] #Initialize the y-coordinate of the initial point for each shape
    deltax = 0.005 #The spacing between shape patches is 5 nm
    patch_start = xi
    
    indexvec = ['shape ']*len(shapes) #Create vector to hold dictionary indices
    patch_arr = np.array(origin)[:,None]
    shape_arr = np.zeros((2,2)) #Initialize shape_arr
    sequence = [] #Initialize sequence
    linelist = [] #initialize linelist

    #For loop  goes through each shape in the shape dictionary
    for i in range(len(shapes)):
        indexvec[i] += str(i+1) #Generate the index for each shape
    
    #For loop  goes through each shape in the shape dictionary
    for i in range(len(shapes)):
        
        l = shapes[indexvec[i]][2] #Store the edge length of a cube as a single variable to be referenced for creating all shapes
        
        #Determine if the shape is a square.
        if shapes[indexvec[i]][0] == 's':
            
            linelist.append(4) #Append 4 sides to linelist
            #A indicates that the patch is on top
            if shapes[indexvec[i]][1] == 'A': 
                yf = yi - l
                sequence.append('A') #Append A orientation to sequence
                
            #B indicates that the patch is on the bottom
            elif shapes[indexvec[i]][1] == 'B':
                yf = yi + l
                sequence.append('B')

            xf = xi + l #Determine the final x-coordinate of the shape
            patch_end = xf

            patch = np.array(([xi,xf],[yi,yi]))
            right = np.array(([xf,xf],[yi,yf]))
            left = np.array(([xi,xi],[yi,yf]))
            bottom = np.array(([xi,xf],[yf,yf]))
            
            if i == 0:
                shape_arr[:,:] = patch #the first entry into shape_arr is a patch. This is needed for intialization
            else:
                shape_arr = np.hstack((shape_arr,patch)) #After the first round, append each patch to shape_arr
            #original_patch_arr = np.hstack((original_patch_arr,patch))
                
            shape_arr = np.hstack((shape_arr,left,bottom,right)) #Append all other sides to shape_arr

            if i < len(shapes)-1: #As long as it's before the last shape
                if shapes[indexvec[i]][1] != shapes[indexvec[i+1]][1]:
                    xi = patch_end+deltax #Update the origin position for generating the next shape
                    if shapes[indexvec[i+1]][1] == 'A': #If the sequence of the next shape is A
                        yi -= deltax #The initial y-value is below the origin
                    elif shapes[indexvec[i+1]][1] == 'B':
                        yi += deltax #This initial value is below the origin
                else:
                     xi = xf+deltax #Update the origin position for generating the next shape
        #Determine if shape is a triangle
        elif shapes[indexvec[i]][0] == 't':
            
            linelist.append(3) #Append 3 sides to the list
            
            h = np.sqrt(2*l**2)  #Determine the height of the triangle from pythagorean theorem. The height is the diagonal of the cube
            
            #Set x and y values for the parts touching the patch
            xmid = h/np.sqrt(3)+xi #Determine x midpoint
            xf = h/np.sqrt(3)*2+xi
            patch_end = xf
            
            #A indicates that the patch is on top
            if shapes[indexvec[i]][1] == 'A':
                ymid = yi-h #y at the midpoint is the negative of the height
                sequence.append('A')
            
            #B indicatesthat the patch is on bottom
            elif shapes[indexvec[i]][1] == 'B':
                ymid = yi + h #y at the midpoint is the height
                sequence.append('B')

            patch = np.array(([xi,xf],[yi,yi]))
            left = np.array(([xi,xmid],[yi,ymid]))
            right = np.array(([xmid,xf],[ymid,yi]))
            
            if i == 0:
                shape_arr[:,:] = patch #the first entry into shape_arr is a patch. This is needed for intialization
            else:
                shape_arr = np.hstack((shape_arr,patch)) #Append the patch to shape_arr
                
            shape_arr = np.hstack((shape_arr,left,right)) #Append all other sides to shape_arr
            
            if i < len(shapes)-1:
                if shapes[indexvec[i]][1] != shapes[indexvec[i+1]][1]:
                    xi = patch_end+deltax #Update the origin position for generating the next shape
                    if shapes[indexvec[i+1]][1] == 'A':
                        yi -= deltax
                    elif shapes[indexvec[i+1]][1] == 'B':
                        yi += deltax
                else:
                     xi = xf+deltax #Update the origin position for generating the next shape

        if i > 0: #Start performing this on the second shape
            if shapes[indexvec[i]][1] == shapes[indexvec[i-1]][1]: #If the current shape has the same orientation as the previous shape
                patch_arr = np.hstack((patch_arr,np.array(([patch_start],[patch[1,0]])))) #perform hstack of the starting x- and y-coordinates. A new patch has started

        if i < len(shapes)-1: #As long as we are before the last shape
            if shapes[indexvec[i]][1] == shapes[indexvec[i+1]][1]: #if the orientation of the current shape matches that of the next shape
                patch_arr = np.hstack((patch_arr,np.array(([patch_end],[patch[1,1]])))) #perform hstack of the ending x- and y-values. The patch has ended

        if i == len(shapes)-1: #If we are on the last shape
            patch_arr = np.hstack((patch_arr,np.array(([patch_end],[patch[1,1]])))) #perform hstack of the endingx- and y-coordinates. The patch must end no matter what

        patch_start = xi #The next patch_start is where the shape starts
        
    hinge_vec = np.ones(len(shapes)-1)*180 #Generate vector of hinges. These represent the interdipolar angle between each of the cubes at
                                           #a hinge point.

    return hinge_vec,shape_arr,sequence,linelist,patch_arr

def sequence_from_shapes(shapes):
    '''This functions determines what the sequence of a structure is from the shapes dicitonary
    Inputs:
        shapes: the input dictionary used to generate a sequence
    Outputs:
        sequence: sequence of shape orientations'''

    sequence = [] #Initialize list to hold the sequence
    for key in shapes.keys(): #For each shape
        sequence.append(shapes[key][1]) #Record the orientation of that shape in sequence

    return sequence

def shapeplots(shape_arr, linelist, blocking = False, title = '', show = True, bounds = '', mag_vecs = []):
    '''Plots all of the shapes in 2D space
    Inputs:
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
        linelist:List of integers describing the number of sides in each shape
        blocking: (optional) boolean describes if plot appearing blocks further code or not
        title: (optional) string title of plot
        bounds: (optional) list of x and y bounds for the plot [xmin xmax ymin ymax]
        mag_vecs: (optinal) use patch_arr as the input for this to plot the magnetic vectors over the patches'''
    

    plt.figure() #Initialize figure
    shape_start = 0 #Initialize index where the shapes start
    shape_end = 0 #initialize index where the shapes end
    for i in range(len(linelist)): #for each shape
        shape_end += linelist[i]*2 #The new index is twice the number of sides in the shape
        plt.plot(shape_arr[0,shape_start:shape_end],shape_arr[1,shape_start:shape_end],'k') #Plot all values of shape array in black
        patch = np.hstack((shape_arr[:,shape_start][:,None],shape_arr[:,shape_start+1][:,None])) #Identify the patch based on shape_arr
        if len(mag_vecs) == 0: #if magvecs are not plotted
            plt.plot(patch[0,:],patch[1,:],'k',linewidth = 5) #plot the patch as a thick line
        shape_start = shape_end #The new start is the previous ending

    if len(mag_vecs) != 0: #If plotting magnetization vectors is desired
        #Plot each magnetization vector as an arrow
        for i in range(0,np.shape(mag_vecs)[1],2):
            plt.arrow(mag_vecs[0,i],mag_vecs[1,i],mag_vecs[0,i+1]-mag_vecs[0,i],mag_vecs[1,i+1]-mag_vecs[1,i], color = 'r', linewidth = 3, head_width = 3, length_includes_head = True)
    
    plt.axis('square') #Fix the axis
    if title != '': #Add title if desired
        plt.title(title)
    if bounds != '': #Add bounds if desired
        plt.axis(bounds)
    if show == True: #Block the rest of the code if desired
        plt.show(block = blocking) #Show plot

def moving_hinges(sequence):
    '''Determines which hinges can move
    Inputs:
        sequence: List of strings describing the orientation of each shape
            Example: ['A','B','B','A']
    Outputs:
        hingenum: list describing which hinges have the ability to move. Hinge indexing starts at 0'''

    hingenum = [] #Initialize list
    
    #For the second shape through the end of the chain
    for i in range(1,len(sequence)):
        if sequence[i] == sequence[i-1]: #Check if the orientation of shape is the same as the one before it
            hingenum.append(i-1) #Add that hinge to the list of potential hinge choices
    hingenum = np.array(hingenum) #Turn it into a numpy array
            
    return hingenum

def not_hinges(hinge_vec,hinges):
    '''Determine which junctions between shapes are not hinges. Necessary for varying the overlap in MC simulations
    Inputs:
        hinge_vec: Vector of hinge angles. Interdipolar angle between each shape
        hinges: list describing which hinges have the ability to move. Hinge indexing starts at 0
    Returns:
        nonhinges: list describing which hinges do not have the ability to move
    '''
    nonhinges = [i for i in range(len(hinge_vec)) if i not in hinges] #determine which junctions are not in hinges and create a list of them
    
    return nonhinges

def moving_indices(shape_arr, patch_arr, linelist, nonhinges, hinges):
    '''This function is necessary to determine the indexing of both shape_arr and patch_arr in order to vary their overlap in repeated MC
    simulations. This allows their initial state to be randomized for each simulation
    Inputs:
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes.
                    s is number of sides on that shape
        patch_arr: 2x(2n) array of x and y points that describe the lines of the magentic patch. n is number of distinct magnetic domains
        linelist: List of integers describing the number of sides in each shape
        nonhinges: list describing which hinges do not have the ability to move
        hinges: list describing which hinges have the ability to move. Hinge indexing starts at 0
    Outputs:
        shape_ind: List of indices that need to be modified in shape_arr during overlap
        patch_ind2: List of indices that need to be modified in patch_arr during overlap
        changes: List that tracks the number of changes made to shape_arr in between each change of patch_arr
    
    '''
    shape_ind = [] #initialize list to hold indices that need to be mofdified in shape_arr
    patch_ind = [] #initialize list to hold indices that need to be modified in patch_arr

    for i in nonhinges: #for each juntion that is not a hinge
        index = 2*np.sum(linelist[0:i+1]) #retrieve the index of that location in shape_arr (the latter shape is moved backwards)
        shape_ind.append(index) #add this index to the shape_ind list
    
    #patch_ind is nuanced because the magnetic vectors don't end at junctions that are not hinges. They only end at hinge junctions,
    #so they must be modified baased on this principle even though the shifting happens at nonhinges. Addtionally, the magentic vectors do
    #not extend to the edges of the shape- this ust be accounted for in modification
    for i in range(len(hinges)+1): #For each junction that is a hinge
        index = i*2 + 1 #Store the index of the patch at the location of a hinge junction (location before the hinge)
        patch_ind.append(index) #Add this index to the patch_list

    #Now, there is a list of indices that need to move in the patch list; however, they do not all move at these points. If there are subsequent
    #hinges, patches may not be affected. Additionally, the first patch is not affected. This must be accoutnedd for
    patch_ind2  = [] #initialize a list of patch indices
    changes = [] #initialize a list of changes
    shape_ind_track = copy.deepcopy(shape_ind) #make a deepcopy of shape_ind to be used for tracking purposes
    for j in range(len(patch_ind)): #for each recorded patch_index from the previous method
        clapback = 0 #Initialize variable to count if an index has already been added or not
        remove_list = [] #Initialize a list of indices that have already been counted
        for i in range(len(shape_ind_track)): #for each shape index that is adjusted
            if patch_arr[0,patch_ind[j]] > shape_arr[0,shape_ind_track[i]]: #if the final location of the patch is after that location in a shape
                if clapback == 0: #And if this variable has not yet been set to one
                    patch_ind2.append(patch_ind[j]) #Add that patch index to the patch index list because it will be affected by the moving
                                                    #shapes since the location on the end of patch is after the location of a nonhinge
                    clapback = 1 #set clapback to 1 so that this index is only added once
                remove_list.append(shape_ind_track[i])#remove this index from the shape index tracker to prevent adding multiples
        changes.append(len(remove_list)) #add the number of removed indices to changes list
        shape_ind_track = [item for item in shape_ind_track if item not in remove_list] #Modify the shape index tracker to only include indices
                                                                                        #not in the list of removed indices

    changes = [item for item in changes if item != 0] #Remove all 0s from the list of changes. Now, this list only tracks how many changes were
                                                      #made in between each patch due to changes in the shapes overlapping
    
    return shape_ind,patch_ind2, changes

def vary_overlap(shape_arr_init,patch_arr_init, shape_ind, patch_ind, changes, dipole_len = 9.3, dipole_len_err = 0.4):
    '''This function is used to vary the amount of overlap for all shapes based on a gaussian distribution derived from the average dipoole length being 9.3 μm and the error in that
    measurement being 0.4 μm. This allows randomization of initialization for all structures in each individual MC simulation.
    Inputs:
        shape_arr_init:2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes.
                       s is number of sides on that shape
        patch_arr_init: 2x(2n) array of x and y points that describe the lines of the magentic patch. n is number of distinct magnetic domains
        shape_ind: List of indices that need to be modified in shape_arr
        patch_ind: List of indices that need to be modified in patch_arr
        changes: List that tracks the number of changes made to shape_arr in between each change of patch_arr
        dipole_len: measured length of the dipole from literature
        dipole_len_err: measurment error in length of dipole from literature
    Outputs:
        shape_arr: Modified 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes.
                   s is number of sides on that shape
        patch_arr: Modified 2x(2n) array of x and y points that describe the lines of the magentic patch. n is number of distinct magnetic domains
    '''

    overlap = (10 - dipole_len)/2 #Determine the mean value of overlap from the measured dipole length in literature
    overlap_err = dipole_len_err/2 #Determine the standard error in that overlap length from literature
    shape_arr = copy.deepcopy(shape_arr_init) #Make a deepcopy of the shape_arr
    patch_arr = copy.deepcopy(patch_arr_init) #Make a deepcopy of the patch_arr
    total_changes  = np.sum(changes) #determine the amount of random numbers to draw for changing the overlap
    overlap_val = [] #Initialize list to hold all overlap values
    while len(overlap_val) < total_changes: #Keep drawing random numbers until at the number of total changes needed
        values = np.random.normal(overlap, overlap_err, total_changes - len(overlap_val)) #draw random numbers equal to the number of total
                                                                                          #changes minus the number already drawn. It is done in
                                                                                          #this way to draw as many numbers as possible and then
                                                                                          #discard negative values
        overlap_val.extend([val for val in values if val >= 0]) #Add all values taht are greater tahn or equal to zero to the list of overlap values

    for i in range(len(shape_ind)): #For each shape index that needs to be modified
        shape_arr[0,shape_ind[i]:] -= overlap_val[i] #Subtract teh overlap value to increase overlap between the shapes

    overlap_sum = 0 #Initialize the counting sum of overlap values
    start = 0 #initialize the starting point
    for i in range(len(changes)): #For each value of changes. --> for each patch that needs to be modified
        overlap_sum = np.sum(overlap_val[start:start+changes[i]])#Sum the overlap values corresponding to the number of shapes that changed 
                                                                 #position in between patches
        patch_arr[0,patch_ind[i]:] -=  overlap_sum #make the modification to the patch
        start += changes[i] #Increment the next starting point to ccount the number of changes from
    
    return shape_arr,patch_arr

def vary_specific_overlap(shape_arr_init,patch_arr_init, shape_ind, patch_ind, changes, overlap_val):
    
    shape_arr = copy.deepcopy(shape_arr_init)
    patch_arr = copy.deepcopy(patch_arr_init)
    total_changes  = np.sum(changes)

    for i in range(len(shape_ind)):
        shape_arr[0,shape_ind[i]:] -= overlap_val[i]

    overlap_sum = 0
    start = 0
    for i in range(len(changes)):
        overlap_sum = np.sum(overlap_val[start:start+changes[i]])
        patch_arr[0,patch_ind[i]:] -=  overlap_sum
        start += changes[i]
    
    return shape_arr,patch_arr

def count_shapes(shape_arr):
    '''This functions uses the shapely library to count the number of shapes in a structure
    Inputs:
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
    Outputs:
        polycount: number of shapes counted
    '''

    polygons = list(polygonize(sp.unary_union(sp.LineString(shape_arr.T)))) #Turn all the points into shapely polygons
    polycount = len(polygons) #The length of polygons is the number of polygons
    
    return polycount

def check_overlap(shape_arr, polycount):
    '''This function checks if any structures overlap each other
    Inputs:
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
        sequence: List of strings describing the orientation of each shape
                  Example: ['A','B','B','A']
    Outputs:
        Boolean value
            True for overlap
            False for no overlap
    '''

    #use shapely library to create a LineString object. This object can be truned into a unary_union which can be turned into separate polygons using shapely.ops polygonize
    polygons = list(polygonize(sp.unary_union(sp.LineString(shape_arr.T)))) #Create list of shapely polygons

    if len(polygons)!= polycount: #If there are more polygons created that the number of shapes we are using, the shapes must be overlapping
        return True #Overlap
    else:
        return False #No overlap

def mag_vectors(sequence, m):
    '''This function determines the experienced magnetization for each magnetic domain. The field experienced by other shapes is dependent
        upon the magnetization of the first patch in the domain and the last one- this was determined in COMSOL simulations.
    Inputs:
        sequence: sequence of the chain. i.e. ABBA or BABAA
        m: list contianing the magnetization/length values for each patch.
            In general, moment/length = 1.01E-06 A*m for a 100 nm cobalt patch
    Output:
        magvec: a vector contianing the magnetic moments/length of the first and last points of each magnetic domain
    '''

    shapecount = 1 #initialize counter of how many shapes are in a patch
    magshape_list = [] #Initialize list to hold number of shapes included in each patch
    for i in range(1,len(sequence)): #For each shape in the sequence except for the first one
        if sequence[i-1] != sequence[i]: #If the previous orientation is the same as the orientation of the current shape
            shapecount +=1 #increase the number of shapes included in a magnetic domain
        else: #If the previous orientation is different than the current orientation
            magshape_list.append(shapecount) #Add the number of shapes counted to list of shapes in each domain
            shapecount = 1 #Return shapecount to 1
    magshape_list.append(shapecount) #Add the last shapecount to magshape list

    sum = 0 #initialize the sum to be 0
    magvec = [] #Initialize a list to hold
    for i in magshape_list: #For each magnetic domain
        magvec.append(m[sum]) #add the magnetization of the first point in the domain to the list of magnetizations
        sum += i #Increment sum by the number of shapes included in each magnetic domain
        magvec.append(m[sum-1]) #Add the magnetization of the last point in the domain to the list of magnetizations
    magvec = np.array(magvec)  #Turn the list into an array

    return magvec

def initialize_energy(magvec):
    '''This function initializes a series fo matrices that help to reduce computational expense of later code. This is necessary for the
        magentic energy calculation performed at each step of the MC simulation. This initialized energy already acounts for the magentization
        values of the patches
    #Inputs:
        magvec: a vector contianing the magnetic moments/length of each point at the end of a patch. This should be input manually or scaled based on SQUID data (A*m)
    Outputs:
        #mask_arr: An special nxn upper diagonal matrix of 0's, 1's, and -1's useful for the energy calculation. n is 2x number of shapes
        #v_xmat: An nx2 array of 1's
        #h_xmat: A 2xn array of -1's
        #v_ymat: An nx2 array of 1's
        #h_ymat: A 2xn array of -1's
        #Ml_mat: An nxn array that contains combinations of magnetizations and lengths
    '''

    n = len(magvec) #Determine the number of points (n)
    mask_arr = np.ones((n, n)) #Initialize mask_arr with ones

    M0 = (4*np.pi)*10**(-7) #J*m #magnetic constant
    constant = M0/(4*np.pi) #J*m #divide magentic constant by 4pi
    
    #Perform matrix multiplication to obtain multiplication between every possible point combination. Additionaly, multiply constant
    #Ml[:,None] essential to perform matrix multiplication because Ml is 1D
    Ml_mat = np.matmul(magvec[:,None],magvec[None,:])*constant #M1_mat is an array that describes the M/L combo for each pairwise interaction

    #Turn mask-arr into one that alternates between 1 and negative 1
    for i in range(n):
            for l in range(n):
                if (i+l) % 2 != 0:
                    mask_arr[i,l] *= -1

    #mask array must be a special upper diagnonal matrix. entries below the diagonal are ommitted because they're the nugative of the upper. entries on the diagonal are
    #ommitted because they would be a point interacting with itself. entries one above the diagonal describe a point on one line interacting with a point on the same line and
    #must be ommitted
    k = 0 #Initialize counter
    for i in range(n): #columns
        #Every two rows, k increases by 2 such that the ommitted values are shifted down by two rows
        for j in range(k,n):
            mask_arr[j,i] = 0 #Replace unwanted entry with a zero
        if i>0 and i%2 != 0: #Increment k counter
            k+=2

    #Create horizontal and vertical matrices needed to subtract each x point from another using matrix math
    #Each vertical array will have its [:,0] entry as x or y-values. Every horizontal array will have its [1,:] entry as x or y values. Through matrix multiplication, this will
    #create an array with each entry as one point subtracted from another
    v_xmat = np.ones((n,2))
    h_xmat = np.ones((2,n))*-1
    v_ymat = np.ones((n,2))
    h_ymat = np.ones((2,n))*-1
            
    return mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat

def energy_math(patch_arr,mask_arr, v_xmat, h_xmat, v_ymat, h_ymat,Ml_mat):
    '''This function calculates the total interdipolar energy of the system
    Inputs:
        patch_arr_init: 2x(2n) array of x and y points that describe the lines of the magentic patch. n is number of distinct magnetic domains
        mask_arr: An special nxn upper diagonal matrix of 0's, 1's, and -1's useful for the energy calculation. n is 2x number of shapes
        v_xmat: An nx2 array of 1's
        h_xmat: A 2xn array of -1's
        v_ymat: An nx2 array of 1's
        h_ymat: A 2xn array of -1's
        Ml_mat: An nxn array that contains combinations of magnetizations and lengths
    #Outputs:
        #E: total energy of the system
    '''
    #Create a deepcopy of line_arr to prevent line_arr from being modified during the vector operations in this function
    patch_arr = copy.deepcopy(patch_arr)
    
    v_xmat[:,0] = patch_arr[0,:] #replace first column with x-values
    h_xmat[1,:] = patch_arr[0,:] #replace second row with x-values
    xmat = np.matmul(v_xmat,h_xmat)*1e-6 #perform matrix multiplication to obtain each x-value subtracted from another. Multiply by 1e-6 to dimensionalize
    xmat_upper = np.multiply(xmat,mask_arr) #Multiply by teh mask array to obtain the special upper diagnonal matrix and gain -1/1 pattern (this pattern is removed when squaring
                                            #and with have to be reinstated later)
    x_square = np.square(xmat_upper) #square all values. note that squaring removes -1/1 pattern
    
    v_ymat[:,0] = patch_arr[1,:] #replace first column with y-values
    h_ymat[1,:] = patch_arr[1,:] #replace second row with y-values
    ymat = np.dot(v_ymat,h_ymat) #perform matrix multiplication to obtain each y-value subtracted from another. Multiply by 1e-6 to dimensionalize
    ymat_upper = np.multiply(ymat,mask_arr)*1e-6 #Multiply by teh mask array to obtain the special upper diagnonal matrix and gain -1/1 pattern (this pattern is removed when squaring
                                            #and with have to be reinstated later)
    y_square = np.square(ymat_upper ) #square all values. note that squaring removes -1/1 pattern
    
    final_mat = np.multiply(np.sqrt(x_square + y_square),mask_arr) #add the squares of x and y values and take the square root. Then multiply this element-wise by mask_arr
                                                                   #to reinstate -1/1 pattern
    indices = np.nonzero(final_mat) #Determine the indices of all nonzero values to allow for taking the reciprocal
    vec = np.multiply(Ml_mat[indices],np.reciprocal(final_mat[indices])) #multiply piecewise the nonzero values of the inverse of the magnitude matrix with corresponding values of the 
                                                                         #M/L combination matrix
    #print(vec)
    E = np.sum(vec) #Sum all values to determine the total interaction energy
    
    return E

def translate_to_origin(patch_arr_init, shape_arr_init, linelist, hingechoice, sym = False):
    '''function translates first point of shape to the right of the hinge to the origin. Necessary for simple rotation
    Inputs:
        patch_arr: 2x(2n) array of x and y points that describe the lines of the magentic patch. n is number of shapes
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
        linelist: List of integers describing the number of sides in each shape
        hingechoice: Chosen hinge to revolve around
        sym: (optional) Boolean indicating whether or not the translation is for the symmetry function
    Outputs:
        new_line_arr: translated line_arr
        new_shape_arr: translated shape_arr
    '''
    #Create a deepcopy of both arrays (necessary to prevent modifying original when performing trial runs in Monte Carlo)
    patch_arr = copy.deepcopy(patch_arr_init)
    shape_arr = copy.deepcopy(shape_arr_init)
    #If the change is not for the symmetry function
    if sym == False:
        index = sum(linelist[i] for i in range(hingechoice+1))*2 #Determine the shape index where the selected hinge is
        xy_trans = np.array(([shape_arr[0,index]],[shape_arr[1,index]])) #Determine hpow far to translate each shape in order to place first point 
                                                                                           #past hinge on the origin
    else: #if the change is for the symmetry function
        index = sum(linelist[i] for i in range(hingechoice+1))*2-7 #only change is that the index is now before the hinge rather than after
        xy_trans = np.array(([shape_arr[0,index]],[shape_arr[1,index]])) #Determine how far to translate each shape in order to place first point 
                                                                                           #past hinge on the origin
    patch_arr -= xy_trans #translate line_arr
    shape_arr -= xy_trans #translate shape_arr
    
    return patch_arr,shape_arr

def translate_back(patch_arr_init,shape_arr_init):
    '''This function translates any structure back to the reference position where the leftmost point of the first shape is located at the origin
    Inputs:
        patch_arr: 2x(2n) array of x and y points that describe the lines of the magentic patch. n is number of shapes
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
    Outputs:
        new_line_arr: translated line_arr
        new_shape_arr: translated shape_arr
    '''

    patch_arr = copy.deepcopy(patch_arr_init)
    shape_arr = copy.deepcopy(shape_arr_init)
    #Create a deepcopy of both arrays (necessary to prevent modifying original when performing trial runs in Monte Carlo)
    xy_trans = np.array(([shape_arr[0,0]],[shape_arr[1,0]]))#Determine how far to translate each shape in order to place the first point of the first shape on the origin.
                                                          #This should be the negative of the location of the first point
    patch_arr -= xy_trans #translate line_arr by subtracting the value (subtracting a negative to make a positive)
    shape_arr -= xy_trans #translate shape_arr
    
    return patch_arr,shape_arr

def rotate(patch_arr_init, shape_arr_init, linelist, hinge_vec_init, hingechoice, hinges, angle):
    '''This function rotates a structure. All points to the right of the origin are rotated by multiplying by teh rotation matrix
    Inputs:
        patch_arr: 2x(2n) array of x and y points that describe the lines of the magentic patch. n is number of shapes
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
        linelist: List of integers describing the number of sides in each shape
        hinge_vec: Vector of hinge angles. Interdipolar angle between each shape
        hingechoice: integer chosen hinge about which to revolve
        hinges: all hinges capable of rotating
        angle: float chosen angly by which to revolve
    #Outputs:
        new patch-arr: rotated line_arr
        new_shape_arr: rotated shape_arr
        new_hinge_vec: rotated hinge_vec
    '''
    hinge_vec = copy.deepcopy(hinge_vec_init)
    patch_arr = copy.deepcopy(patch_arr_init)
    shape_arr = copy.deepcopy(shape_arr_init)
    #Create a deepcopy of three arrays to be modified (necessary to prevent modifying original when performing trial runs in Monte Carlo)
    hinge_vec[hingechoice] += angle #Modify hinge_vec by adding the angle
    
    angle = angle*np.pi/180 #convert angle to radians
    #Rotation matrix: [cos -sin]
    #                 [sin  cos]
    rotation_matrix =np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]) #Build the rotation matrix for that angle
    index = (np.nonzero(hinges == hingechoice)[0][0])*2 + 2
    #rotate line and shape arrays using matrix multiplication. Matrix multiplication performed on all points to the right of the hinge. Indexing reflects position of hinge in arrays
    patch_arr[:,index:] = np.matmul(rotation_matrix,patch_arr[:,index:])
    shape_arr[:,np.sum(linelist[:hingechoice+1]*2):] = np.matmul(rotation_matrix,shape_arr[:,np.sum(linelist[:hingechoice+1]*2):])
    
    return patch_arr, shape_arr, hinge_vec

def rotate_once(patch_arr,shape_arr,linelist,hinge_vec,hingechoice, hinges, angle):
    '''This function performs the translate,rotate,translate cycle all together and is important in later functions and for plotting or troubleshooting
    Inputs:
        patch_arr: 2x(2n) array of x and y points that describe the lines of the magentic patch. n is number of shapes
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
        linelist: List of integers describing the number of sides in each shape
        hinge_vec: Vector of hinge angles. Interdipolar angle between each shape
        hingechoice: integer chosen hinge about which to revolve
        hinges: all hinges capable of rotating
        angle: float chosen angly by which to revolve
    #Outputs:
        new patch-arr: rotated line_arr
        new_shape_arr: rotated shape_arr
        new_hinge_vec: rotated hinge_vec
    '''

    patch_arr, shape_arr = translate_to_origin(patch_arr, shape_arr, linelist, hingechoice) #translate hinge to the origin
    patch_arr, shape_arr, hinge_vec = rotate(patch_arr, shape_arr, linelist, hinge_vec, hingechoice, hinges, angle) #rotate structure
    patch_arr,shape_arr = translate_back(patch_arr,shape_arr) #translate structure back
    
    return patch_arr,shape_arr,hinge_vec

def energyscale_magnetic(hingechoice, hinges, patch_arr, shape_arr, linelist, std, hinge_vec, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat):
    '''This function determinese the reference energy change with which to scale the acceptance-rejection Criterion. It works by calculating
        the change in energy about a single hinge for a move that is exactly one standard deviation change in angle. This works because a
        standard deviation in the change in angle corresponds to a standard deviation in the change in energy when examining one hinge
    Inputs:
        hingechoice: the hinge about which the energy is calculated
        hinges: list of moving hinges
        patch_arr: 2x(2n) array of x and y points that describe the magnetic domains of the patch. n is number of distinct domains
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
        linelist: List of integers describing the number of sides in each shape
        std: standard deviation of gaussian distribution from which to pull each rotation angle
        hinge_vec: vector holding the value of all interdipolar angles
        mask_arr: An special nxn upper diagonal matrix of 0's, 1's, and -1's useful for the energy calculation. n is 2x number of shapes
        v_xmat: An nx2 array of 1's
        h_xmat: A 2xn array of -1's
        v_ymat: An nx2 array of 1's
        h_ymat: A 2xn array of -1's
        M1_mat: An nxn array that contains combinations of magnetizations and lengths
    Outputs:
        d_energy: scalar energy difference
        last_energy: The energy of the system before a change in the angle of the chosen hinge of one standard deviation
    '''

    angle_trial = std #Set the trial angle to the standard deviation
    lastenergy = energy_math(patch_arr,mask_arr, v_xmat, h_xmat, v_ymat, h_ymat,Ml_mat) #Calculate energy of the system before rotating
    patch_arr_trial,shape_arr_trial = translate_to_origin(patch_arr, shape_arr, linelist, hingechoice) #Translate the assembly to the origin
    patch_arr_trial,shape_arr_trial,hinge_vec_trial = rotate(patch_arr_trial, shape_arr_trial, linelist, hinge_vec, hingechoice, hinges, angle_trial)
    patch_arr_trial,shape_arr_trial,hinge_vec_trial = rotate(patch_arr_trial, shape_arr_trial, linelist, hinge_vec, hingechoice, hinges, angle_trial) #rotate the assembly on the right of the hinge
    newenergy = energy_math(patch_arr_trial,mask_arr, v_xmat, h_xmat, v_ymat, h_ymat,Ml_mat) #Calculate the new energy of the trial configuration
    d_energy = newenergy - lastenergy #Calculate energy difference

    return d_energy, lastenergy

def characteristic_energy(std,sequence,patch_arr_init,shape_arr_init,hinge_vec_init,linelist,polycount,initial_angle,mask_arr, v_xmat, h_xmat, v_ymat, h_ymat,Ml_mat):
    '''This function determines the characteristic energy by taking every movable hinge, rotating it by a specified angle (usually set to
        30 degrees) and applies energyscale magnetic to determine the energy landscape of that hinge. An average can be taken of the change in
        energy about each hinge in both positiive and negative direcctions to determine a relevant acceptance-rejection energy change.
        Additionally, it returns the energy value that the system energy must drop below for annealing to begin
    Inputs:
        std: standard deviation for angle selection about each hinge
        sequence: list of strings describing shape orientation
        patch_arr_init: 2x(2n) array of x and y points that describe the magnetic domains of the patch. n is number of distinct domains
        shape_arr_init: 2x(2n*s) array of x and y points that describe the lines enclosing each shape in the initial position. n is number of shapes. s is number of sides on that shape
        hinge_vec_init: Vector of initial hinge angles. Interdipolar angle between each shape
        linelist: List of integers describing the number of sides in each shape
        polycount: number of polygons shapely initially counts in the system
        initial_angle: Initial angle to rotate all hinges to when calculating characteristic energy
        mask_arr: An special nxn upper diagonal matrix of 0's, 1's, and -1's useful for the energy calculation. n is 2x number of shapes
        v_xmat: An nx2 array of 1's
        h_xmat: A 2xn array of -1's
        v_ymat: An nx2 array of 1's
        h_ymat: A 2xn array of -1's
        l_mat: An nxn array that contains combinations of magnetizations and lengths
    Outputs:
        E_refs: A list of reference energies (energy change) for each hinge in each rotation direction
        E_low: the energy value associated with crossing into the threshold of the characteristic energy
               This will be used to mark where annealing starts
    '''

    hinges = moving_hinges(sequence) #Determine what hinges are able to move
    E_refs = [] #Intialize list to hold reference energies
    E_low = []
    options = [1,-1] #Idicates that the configuration can be initially folded in either the positive or negative direction

    for i in range(len(hinges)): #Loop through all movable hinges
        hingechoice = hinges[i] #hingechoice is that particulat hinge

        for j in options: #Try out both a positive and negative rotation
            
            angle = initial_angle*j #angle is the desired angle multiplied by +/- 1
            patch_arr = copy.deepcopy(patch_arr_init)
            shape_arr = copy.deepcopy(shape_arr_init)
            hinge_vec = copy.deepcopy(hinge_vec_init)
            patch_arr,shape_arr,hinge_vec = rotate_once(patch_arr, shape_arr, linelist, hinge_vec, hingechoice, hinges, angle) #Rotate the structure to its
                                                                                                                #initial position                                                                                        
            overlap = check_overlap(shape_arr, polycount) #check to see if this initial position overlaps
            if overlap == False: #If there is no overlap
                 #Perform energyscale magnetic to determine the energy landscale about that hinge in that direction
                 d_e, last_e  = energyscale_magnetic(hingechoice, hinges, patch_arr, shape_arr, linelist, std, hinge_vec, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat)
                 d_e = np.abs(d_e) #Determine the absolute calue of the energy change
                 E_refs.append(d_e*np.sqrt(2*np.pi)/2) #generate the reference energies by determining thhe scale of the normal distribution
                 E_low.append(last_e) #Append the annealing energy for that hinge to the E_low list

    return E_refs, E_low

def magrun_worker_forward(patch_arr, shape_arr, hinge_vec, linelist, polycount, moves, hingenum, std, E_ref_initial, E_low, anneal_rate, mask_arr, v_xmat,
                          h_xmat, v_ymat, h_ymat, Ml_mat, anneal_cutoff, equilibrium_cutoff):
    '''This function is the new meat and potatoes of the entire code. Each instance of this code is one annealing MC simulation. Each step
        picks a hinge at random, rotates it by an angle drawn from a gaussian distribution, checks is the new structure overlaps, compares
        the change in energy between states to an acceptance-rejection criterion, and continually anneals that criterion over time until the
        system is at a mechanical equilibrium
    Inputs:
        patch_arr: 2x(2n) array of x and y points that describe the magnetic domains of the patch. n is number of distinct domains
        shape_arr: 2x(2n*sw) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
        hinge_vec: Vector of hinge angles. Interdipolar angle between each shape
        linelist: List of integers describing the number of sides in each shape
        polycount: number of distinct polygons (used for measuring overlap)
        moves: maximum number of monte carlo moves
        hingenum: list of hinges capable of moving/rotating
        std: standard deviation of gaussian distribution from which to pull each rotation angle
        E_ref_initial: Initial reference energy (Actually a reference energy change) (J/step) to be used for the acceptance/rejection criterion
                        before annealing
        E_low: The energy (representative of the energetic criterion) under which to begin annealing
        anneal_rate: the rate at which the process anneals (J/step)
        mask_arr: An special nxn upper diagonal matrix of 0's, 1's, and -1's useful for the energy calculation. n is 2x number of shapes
        v_xmat: An nx2 array of 1's
        h_xmat: A 2xn array of -1's
        v_ymat: An nx2 array of 1's
        h_ymat: A 2xn array of -1's
        Ml_mat: An nxn array that contains combinations of magnetizations and lengths
    #Returns:
        #save_patch_arr: patch_arr saved at minimum energy state
        #save_shape_arr: shape_arr saved at minimum energy state
        #save_hinge_vec: hinge_vec saved at minimum energy state
        #hinge_vec: final hinge_vec of mc
        #store_vec: records relevant information at all mc steps (both accepted and rejected)
        #totalhinge_vec: total_movesxhinges array where each row represents an mc step and the hinges during that step
        #accept_store_vec: Stores all values from accepted mc steps
    '''

    np.seterr(over='ignore')
    E_ref = 1#E_ref_initial
    count = 0 #Intialize counter to keep track of what cycle it's on
              #Allows the code to determine multiples of the interval and the last run
    
    store_vec = np.zeros(2) #Initialize store vec. Data will look like: ['energy','energy change','hinge','choice','angle','boltze','randnum','overlap']
    accept_store_vec = np.zeros(2)# initialize accept_store_vec
    movevec = np.zeros(2) #Initialize a vector to store information for each individual move that will be appended to store_vec
    total_moves = 0 #Initialize the counting of total moves

    lastenergy = energy_math(patch_arr, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat) #Initialize lastenergy (energy of the last state) by calculating the energy of the current state
    newenergy = 0 #Initialize variable to hold the new calculated energy of the system at each move
    d_energy = 0 #Initialize variable to hold the change in energy for each move
    
    min_e = lastenergy #Initialize a variable to store the overall minimum energy of the simulation
    min_e_try = lastenergy #Initialize variable to test for the new minimum energy state
    min_count = 0 #Initialize counter that will determine when the system is at an energy minimum
    e_count = 0 #Initialize counter of positive energy moves
    
    movevec[0] = lastenergy #Insert first value of energy into movevec
    store_vec[0] = lastenergy #Insert first value of energy into store_vec
    accept_store_vec[0] = lastenergy #Insert first value of energy into accept_store_vec

    total_hinge_vec = hinge_vec
    check1 = 0
    counte = 0
    E_check = 0

    forward_direction = np.ones(len(hinge_vec))
    current_direction = np.ones(len(hinge_vec))
    angle_counter = np.ones(len(hinge_vec))
    forward_only = 0
    forward_check = 1
    
    #While loop executes as long as count is less than the maximum number of specified Monte Carlo moves
    while total_moves < moves:

        total_moves += 1 #Increment total number of moves each instance of the while loop
        hingechoice = int(np.random.choice(hingenum)) #Randomly select a hinge to move
        sign = 0

        if forward_only == 1 and forward_check == 0:
            if np.any(angle_counter > 15):
                forward_check = 1
                exceed_indices = np.where(angle_counter > 10)[0]
                forward_direction[exceed_indices] *= -1
        
        patch_arr,shape_arr = translate_back(patch_arr,shape_arr) #Translate the shapes back to the origin
        
        if total_moves == 1: #Execture this code on the first round
            lastenergy = copy.deepcopy(store_vec[0])#The last energy is the current energy of the system. A deepcopy is used to prevent linking storevec to lastenergy.
                                                    #otherwise, changing lastenergy will update store_vec
        else: #Execute this code every round except for the first
            lastenergy = copy.deepcopy(store_vec[total_moves-1,0]) #The last energy will come from the energy on the previous row of store_vec. deepcopy is used for the same reason   
        
        if forward_only != 1:
            angle_trial = np.random.normal(0,std) #choose a trial angle from the Gaussian distribution
        else:
            while sign != forward_direction[hingechoice]:
                angle_trial = np.random.normal(0,std)
                sign = np.sign(angle_trial)
        patch_arr,shape_arr = translate_to_origin(patch_arr, shape_arr, linelist, hingechoice) #translate the selected hinge to the origin and update line_arr and shape_arr
        patch_arr_trial,shape_arr_trial,hinge_vec_trial = rotate(patch_arr, shape_arr, linelist, hinge_vec, hingechoice, hingenum, angle_trial) #rotate all shapes to the right of the hinge
                                                                                                                            #At this point, trial arrays/vectors are made to test
                                                                                                                            #out the move
        overlap = check_overlap(shape_arr_trial, polycount) #Check Overlap

        patch_arr_trial,shape_arr_trial = translate_back(patch_arr_trial,shape_arr_trial)#Translate the shapes back to the origin
        newenergy = energy_math(patch_arr_trial, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat,Ml_mat)#Calculate the new energy of the trial configuration
        d_energy = newenergy - lastenergy #Calculate energy difference
        movevec[1] = d_energy #Update movevec with the change in energy

        if E_check > 0 and check1 < 1: #If annealing is active and we are not yet at the point of only accepting favorable moves
            if anneal_rate !=0: #As long as the anneal rate does not equal zero (high t limit)
                E_ref = E_ref_initial - anneal_rate*counte #The reference energy decreases with each step
            counte += 1 #keep track of the decreasing reference energy
            if E_ref <= 0: #If the reference energy drops below zero
                E_ref = 1e-30 #set it to a constant value

        #metropolis criterion
        if d_energy <= 0: #if the change in energy is <=0, then boltze = 2 so that the move is always accepted
            boltze = 2
        elif d_energy > 0: # if the change in energy is >0
            boltze = np.exp(-d_energy/E_ref) #scale the energy to the reference energy and take the negative exponential
            e_count +=1 #Increment the positive energy counter
        
        randnum = random.uniform(0,1) #generate a random number between 0 and 1

        if overlap == False and randnum <= boltze: #If there is no overlap and the change in energy is either less than 0
                                                   #or a random number between 0 and a designated value (allowing for small
                                                   #chances of unfavorable movement), then the move is accepted
                                                                                   
            #Update all values with the trial values in the event of success
            patch_arr = patch_arr_trial
            shape_arr = shape_arr_trial
            hinge_vec = hinge_vec_trial
            total_hinge_vec = np.vstack((total_hinge_vec,hinge_vec))
            current_direction[hingechoice] = np.sign(angle_trial) #one issue here is that for very long structures, not every hinge gets to move before annealing begins,
                                                                  #so things get locked moving in one direction
            count += 1 #Increment accepted monte carlo move counter
            movevec[0] = newenergy
            accept_store_vec = np.vstack((accept_store_vec,movevec)) #Add movevec onto accept_store_ve
            
            if newenergy < E_low and E_check == 0: #If we are below the annealing energy for the first time
                E_check = 1 #Increment the criteria that allows annealing to begin
                forward_direction = copy.deepcopy(current_direction)
                forward_only = 1
                forward_check = 0
                
        elif overlap == False and randnum > boltze:
            movevec[0] = lastenergy
            angle_counter[hingechoice] += 1
              
        else:
            movevec[0] = lastenergy#Store energy

        #Whether accepted or rejected, add movevec onto store_vec
        store_vec = np.vstack((store_vec,movevec))

        #This segment of code determines if the system is at equilibrium
        if count > 0: #Only begin if the count is greater than 0
            min_e_try = np.min(store_vec[:,0]) #Determine the minimum energy of the system from store_vec and store it in trial variable
            if min_e_try < min_e: #If the new energy minimum is less than the previous energy minimum
                min_e = copy.deepcopy(min_e_try) #store the trial values as a deepcopy of the trial vector
                min_count = 0 #Set the minimum energy counter to 0
            else: #If the trial energy is greater than or equal to the previous energy minimum
                min_count +=1 #increment the minimum energy counter that counts to equilibrium

        #If we have been at an energy minimum for anneal_cutoff moves, this if statement hasen't been initiated yet, the anneal rate is greater than 0,
        #and annealing has been initiated
        if min_count > anneal_cutoff and check1 == 0 and anneal_rate !=0 and E_check > 0:
            std /= 2 #Decrease the standard deviation to enable smaller moves with higher probability
            check1 = 1 #Increment the check so that this step cannot happen again
            E_ref = 1e-30 #Set the reference energy to only allow favorable moves

        if min_count > equilibrium_cutoff and check1 > 0: #If the minimum counter reaches 100
            break #Break the while loop

    return patch_arr, hinge_vec

def magrun_forward(patch_arr_init,shape_arr_init,hinge_vec_init, hingenum, linelist, polycount, moves, std, E_ref_initial, E_low, anneal_rate, mask_arr, v_xmat,
                   h_xmat, v_ymat, h_ymat, Ml_mat, shape_ind, patch_ind, changes, anneal_cutoff, equilibrium_cutoff, change_overlap):
    '''This is the worker function for running the Monte Carlo Simulation repeatedly and collecting the results. It applies
        efficient_magnetic_anneal and records relevant values to return from the function
    Inputs:
        patch_arr_init: 2x(2n) array of x and y points that describe the magnetic domains of the patch. n is number of distinct domains
        shape_arr_init: 2x(2n*s) array of x and y points that describe the lines enclosing each shape in the initial position. n is number
                         of shapes. s is number of sides on that shape
        hinge_vec_init: Vector of initial hinge angles. Interdipolar angle between each shape
        hingenum: list of hinges that can move
        linelist: List of integers describing the number of sides in each shape
        polycount: number of polygons shapely initially counts in the system
        moves: number of allowable moves in an MC run
        std: standard deviation for angle selection about each hinge
        E_ref_initial: Reference energy to be used for the acceptance-rejection criterion before annealing begins
        E_low: the energy under which to begin annealing
        anneal_rate: The rate at which annealing occurs
        mask_arr: An special nxn upper diagonal matrix of 0's, 1's, and -1's useful for the energy calculation. n is 2x number of shapes
        v_xmat: An nx2 array of 1's
        h_xmat: A 2xn array of -1's
        v_ymat: An nx2 array of 1's
        h_ymat: A 2xn array of -1's
        Ml_mat: An nxn array that contains combinations of magnetizations and lengths
        shape_ind: List of indices that need to be modified in shape_arr
        patch_ind: List of indices that need to be modified in patch_arr
        changes: List that tracks the number of changes made to shape_arr in between each change of patch_arr
    Returns:
        save_hinge_vec: Minimum energy hinge positions
        Accept_store_vec: simsx2 array holding all energies and changes in energy foor all accepted moves
        store_vec: simsx2 array holding all energies and changes in energy
        E: Energy of the minimum energy configuration
        totalhingevec: all recorded hinge positions throughout the MC simulation
    '''
    hinge_vec = copy.deepcopy(hinge_vec_init)#Make a deepcopy of the initial hinges
    if change_overlap == True:
        shape_arr, patch_arr = vary_overlap(shape_arr_init, patch_arr_init, shape_ind, patch_ind, changes) #Vary the overlapping shapes in the
                                                                                                       #structure
        polycount = count_shapes(shape_arr) #count the shapes in the structure
    else:
        shape_arr = copy.deepcopy(shape_arr_init)
        patch_arr = copy.deepcopy(patch_arr_init)

    #Fold the system until equiliubrium
    save_patch_arr, save_hinge_vec = magrun_worker_forward(patch_arr, shape_arr, hinge_vec, linelist, polycount, moves, hingenum, std, E_ref_initial,
                                                   E_low, anneal_rate, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat, anneal_cutoff, equilibrium_cutoff)
    
    E = energy_math(save_patch_arr, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat,Ml_mat) #calculate the energy of the minimum energy state (save_line_arr)

    return save_hinge_vec, E

def show_final_structure(hingenum, shape_arr_initial, patch_arr_initial, linelist, hinge_vec_initial, final_hinges):
    
    hinge_vec = copy.deepcopy(hinge_vec_initial)
    shape_arr = copy.deepcopy(shape_arr_initial)
    patch_arr = copy.deepcopy(patch_arr_initial)
    for i in range(len(final_hinges)):
        hingechoice = hingenum[i]
        angle = final_hinges[i] - hinge_vec[hingechoice]
        patch_arr,shape_arr,hinge_vec = rotate_once(patch_arr,shape_arr,linelist,hinge_vec,hingechoice, hingenum, angle) #Rotate each hinge according to the angle

    shapeplots(shape_arr,linelist,title = str(i)) #Display the final orientation

def runrun_parallel(patch_arr,shape_arr,hinge_vec,hingenum, linelist, polycount, moves, std, E_ref_initial, E_low, anneal_rate, sims, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat, shape_ind, patch_ind, changes, change_overlap = True):
    '''This function runs the Monte Carlo (efficeint_magnetic) sims times in parallel (parallelization depends on number of CPUs).
        It runs the MCs and returns relevant information about the completed Monte Carlo simulations
    Inputs:
        patch_arr: 2x(2n) array of x and y points that describe the magnetic domains of the patch. n is number of distinct domains
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each shape in the initial position. n is number
                   of shapes. s is number of sides on that shape
        hinge_vec: Vector of initial hinge angles. Interdipolar angle between each shape
        hingenum: list of hinges that can move
        linelist: List of integers describing the number of sides in each shape
        polycount: number of polygons shapely initially counts in the system
        moves: number of allowable moves in an MC run
        std: standard deviation for angle selection about each hinge
        E_ref_initial: Reference energy to be used for the acceptance-rejection criterion before annealing begins
        E_low: the energy under which to begin annealing
        anneal_rate: The rate at which annealing occurs
        sims: Number of simulations to perform
        mask_arr: An special nxn upper diagonal matrix of 0's, 1's, and -1's useful for the energy calculation. n is 2x number of shapes
        v_xmat: An nx2 array of 1's
        h_xmat: A 2xn array of -1's
        v_ymat: An nx2 array of 1's
        h_ymat: A 2xn array of -1's
        Ml_mat: An nxn array that contains combinations of magnetizations and lengths
        shape_ind: List of indices that need to be modified in shape_arr
        patch_ind: List of indices that need to be modified in patch_arr
        changes: List that tracks the number of changes made to shape_arr in between each change of patch_arr
    Returns:
        final_hinges: array of all final hinge conformations
        final_e: vector of all final energies
    '''

    final_hinges = np.zeros((sims,len(hingenum))) #Initialize an array to store all final hinge conformations
    final_e = np.zeros(sims) #Initialize vector to store final energy state of each fold

    num_workers = mp.cpu_count() #Determine the number of available cpu cores
    pool = mp.Pool(processes = num_workers) #Instantiate the worker pool
    
    results = [] #Initialize results for multiprocessing pool
    for i in range(sims): #For loop runs through all simulations
        worker_id = i #Assign worker id based on for loop index
        #Use multiprocessing pool to assign tasks to worker function magrun. Each instance of magrun performs one full monte carlo simulation. These can be applied asynchronously
        #because each simulation is independent
        result = pool.apply_async(magrun,(patch_arr,shape_arr,hinge_vec, hingenum, linelist, polycount, moves, std, E_ref_initial,
                                          E_low, anneal_rate, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat, shape_ind, patch_ind, changes, change_overlap))
        results.append(result) #append the results of each instance of the process pool
    
    processed_results = [result.get() for result in results] #process the multiprocessing results

    for i in range(len(processed_results)): #loop through all processed results
        save_hinge_vec, E = processed_results[i] #record outputs from each simulation
        #For all hinges that actually move
        for j in range(len(hingenum)): #loop through number of movable hinges
            final_hinges[i,j]= save_hinge_vec[hingenum[j]] #Place all values of the final hinge angles into their corresponding index in final_hinges
            final_e[i] = E #The minimum energy of a fold is stored
    
    pool.close() #close the process pool
    pool.join() #join the process pool
                
    return final_hinges, final_e

def runrun_parallel_forward(patch_arr,shape_arr,hinge_vec,hingenum, linelist, polycount, moves, std, E_ref_initial, E_low, anneal_rate, sims, mask_arr, v_xmat,
                            h_xmat, v_ymat, h_ymat, Ml_mat, shape_ind, patch_ind, changes, anneal_cutoff, equilibrium_cutoff, change_overlap = True):
    '''This function runs the Monte Carlo (efficeint_magnetic) sims times in parallel (parallelization depends on number of CPUs).
        It runs the MCs and returns relevant information about the completed Monte Carlo simulations
    Inputs:
        patch_arr: 2x(2n) array of x and y points that describe the magnetic domains of the patch. n is number of distinct domains
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each shape in the initial position. n is number
                   of shapes. s is number of sides on that shape
        hinge_vec: Vector of initial hinge angles. Interdipolar angle between each shape
        hingenum: list of hinges that can move
        linelist: List of integers describing the number of sides in each shape
        polycount: number of polygons shapely initially counts in the system
        moves: number of allowable moves in an MC run
        std: standard deviation for angle selection about each hinge
        E_ref_initial: Reference energy to be used for the acceptance-rejection criterion before annealing begins
        E_low: the energy under which to begin annealing
        anneal_rate: The rate at which annealing occurs
        sims: Number of simulations to perform
        mask_arr: An special nxn upper diagonal matrix of 0's, 1's, and -1's useful for the energy calculation. n is 2x number of shapes
        v_xmat: An nx2 array of 1's
        h_xmat: A 2xn array of -1's
        v_ymat: An nx2 array of 1's
        h_ymat: A 2xn array of -1's
        Ml_mat: An nxn array that contains combinations of magnetizations and lengths
        shape_ind: List of indices that need to be modified in shape_arr
        patch_ind: List of indices that need to be modified in patch_arr
        changes: List that tracks the number of changes made to shape_arr in between each change of patch_arr
    Returns:
        final_hinges: array of all final hinge conformations
        final_e: vector of all final energies
    '''

    final_hinges = np.zeros((sims,len(hingenum))) #Initialize an array to store all final hinge conformations
    final_e = np.zeros(sims) #Initialize vector to store final energy state of each fold

    num_workers = mp.cpu_count() #Determine the number of available cpu cores
    pool = mp.Pool(processes = num_workers) #Instantiate the worker pool
    
    results = [] #Initialize results for multiprocessing pool
    for i in range(sims): #For loop runs through all simulations
        worker_id = i #Assign worker id based on for loop index
        #Use multiprocessing pool to assign tasks to worker function magrun. Each instance of magrun performs one full monte carlo simulation. These can be applied asynchronously
        #because each simulation is independent
        result = pool.apply_async(magrun_forward,(patch_arr,shape_arr,hinge_vec, hingenum, linelist, polycount, moves, std, E_ref_initial, E_low,
                                                  anneal_rate, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat, shape_ind, patch_ind, changes,
                                                  anneal_cutoff, equilibrium_cutoff, change_overlap))
        results.append(result) #append the results of each instance of the process pool
    
    processed_results = [result.get() for result in results] #process the multiprocessing results

    for i in range(len(processed_results)): #loop through all processed results
        save_hinge_vec, E = processed_results[i] #record outputs from each simulation
        #For all hinges that actually move
        for j in range(len(hingenum)): #loop through number of movable hinges
            final_hinges[i,j]= save_hinge_vec[hingenum[j]] #Place all values of the final hinge angles into their corresponding index in final_hinges
            final_e[i] = E #The minimum energy of a fold is stored
    
    pool.close() #close the process pool
    pool.join() #join the process pool
                
    return final_hinges, final_e

def show_final_structures(hingenum, shape_arr_initial, patch_arr_initial, linelist, hinge_vec_initial, clustercenters):
    '''Function displays the final conformation of the sequence in each cluster based on the minimum-energy centroid of that cluster
    #Inputs:
        #hingenum: list of moving hinges
        #shapes: disctionary of all shapes
        #clustercenters: centroid point of each cluster
    '''
    #Loops through number of clusters
    for i in range(len(clustercenters)):
        hinge_vec = copy.deepcopy(hinge_vec_initial)
        shape_arr = copy.deepcopy(shape_arr_initial)
        patch_arr = copy.deepcopy(patch_arr_initial)
        if len(hingenum) == 1:
            cluster_index = 1
        else:
            cluster_index = np.shape(clustercenters)[1]
        #Loops through each angle of the centroid point
        for j in range(cluster_index):
            hingechoice = hingenum[j] #hingechoice is the hinge that corresponds to the centroid point
            angle = clustercenters[i,j]-hinge_vec[hingechoice] #Angle is the difference between current hinge angle (180) and centroid
            patch_arr,shape_arr,hinge_vec = rotate_once(patch_arr,shape_arr,linelist,hinge_vec,hingechoice, hingenum, angle) #Rotate each hinge according to the angle

        shapeplots(shape_arr,linelist,title = str(i)) #Display the final orientation

def cluster_num(dataset,cluster_max,states, Plot = True):
    '''This function combines the previous cluster number determination functiosn such that less computational time and effort is used. This way, the same model is used in all 3 scenarios. This may potentially
        be less robust
    Inputs:
        dataset: the set of data for which you want to find clusters. In this case, te function is using equilibrium angle data from "final_hinges"
        cluster_max: the maximum number of clusters to look for
        states: number of states to initialize for enhanced optimization
        Plot: boolean of whether or not the distortions should be plotted
    Outputs:
        clusternumber: the optimal number of clusters to describe the data
    '''
    silhouette_arr = np.zeros((states,cluster_max)) #Initialize silhouette score list
    davies_arr = np.zeros((states,cluster_max)) #Initialize list to hold davies-bouldin scores
    kvals = list(range(0,cluster_max))#Initialize list to hold sequential clusternumbers
    kvals = np.array(kvals)#Turn this into an array

    for i in range(2,cluster_max): #Determine silhouette score for numbers of clusters from 2 to the max number
        for j in range(states):
            km = KMeans(n_clusters=i, init='k-means++',n_init=10, max_iter=300,tol=1e-04, random_state=j).fit(dataset) #instantiate KMeans
            preds = km.predict(dataset)#predict the dataset with Kmeans
            silhouette = silhouette_score(dataset,preds)#Determine the silhouette score
            silhouette_arr[j,i] = silhouette #Append to the silhouette score
            davies_score = davies_bouldin_score(dataset, preds) #Determine the davies-bouldin score
            davies_arr[j,i] = davies_score #Add the score to the list

    km_silhouette = np.max(silhouette_arr, axis = 0)
    sil_clusternumber = np.argmax(km_silhouette)

    davies_scores = np.min(davies_arr, axis = 0)
    score_ind = np.argmin(davies_scores[2:]) + 2 #The ideal score is the minimum value of all scores
    davies_clusternumber = kvals[score_ind] #find the corresponding k value for that minimum score
    clusternumber = np.max((sil_clusternumber,davies_clusternumber))

    if Plot == True:

        fig, ax1 = plt.subplots(figsize=(10, 8))
        # Plot Silhouette score on the left y-axis
        ax1.scatter([i for i in range(2, cluster_max)], km_silhouette[2:], color='#08d1f9', s=240, alpha=0.8, edgecolors='black', linewidth=2)
        ax1.plot([i for i in range(2, cluster_max)], km_silhouette[2:], color='#08d1f9', linestyle='--', linewidth=2,label = 'Silhouette')  # Dashed line
        ax1.set_xlabel("Number of clusters", fontsize=14)
        ax1.set_ylabel("Silhouette score", fontsize=15)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.tick_params(axis='y', labelsize=15)

        # Create the second y-axis sharing the same x-axis
        ax2 = ax1.twinx()

        # Plot Davies-Bouldin score on the right y-axis
        ax2.scatter([i for i in range(2, cluster_max)], davies_scores[2:], color='#f1910c', s=240, alpha=0.8, edgecolors='black', linewidth=2)
        ax2.plot([i for i in range(2, cluster_max)], davies_scores[2:], color='#f1910c', linestyle='--', linewidth=2, label = 'Davies-Bouldin')  # Dashed line
        ax2.set_ylabel("Davies-Bouldin score", fontsize=15)
        ax2.tick_params(axis='y', labelsize=15)

        ax1.grid(True)
        ax1.legend(loc = 'best')

        # Show the plot
        plt.show(block=False)

    return clusternumber

def min_cluster_centers(dataset, clusternumber, final_e, Plots=''):
    """
    Determine the center of each cluster and the points in the dataset that correspond to each cluster.

    Inputs:
        dataset: np.array, The dataset to be analyzed (equilibrium angle data). 
                 Shape: (m x n), where m is the number of equilibrium states and n is the number of hinges.
        clusternumber: int, The ideal number of clusters to represent the data.
        final_e: np.array, A vector of final energies corresponding to all Monte Carlo (MC) simulations.
        Plots: str (optional), Specifies the dimension of the plot to generate ('1D', '2D', or '3D').
               Default is an empty string '', which means no plot will be generated.

    Returns:
        clustercenters: np.array, Centroids (center points) of each cluster.
        cluster_labels: np.array, Labels indicating which cluster each point in the dataset belongs to.
        clusternumber: int, The optimal number of clusters.
    """

    # Create a combined dataset including the original dataset and final energies
    full_data = np.zeros((np.shape(dataset)[0], np.shape(dataset)[1] + 1))  # Initialize full data array
    full_data[:, 0:np.shape(dataset)[1]] = dataset  # Add the original dataset
    full_data[:, np.shape(dataset)[1]] = final_e  # Append the final energies as an additional column

    # Initialize an array to store the centroids of clusters
    clustercenters = np.zeros((clusternumber, np.shape(dataset)[1]))

    # Instantiate the KMeans clustering algorithm
    km = KMeans(
        n_clusters=clusternumber, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=1
    )

    # Perform clustering on the dataset
    y_km = km.fit_predict(dataset)  # Fit and predict cluster assignments for each point
    clustercenters1 = km.cluster_centers_  # Retrieve the centroids of the clusters
    cluster_labels = km.labels_  # Retrieve the cluster labels for each point in the dataset

    # Adjust cluster centers based on the member with the lowest energy in each cluster
    for i in range(clusternumber):
        # Collect all points (and their corresponding energies) belonging to the current cluster
        cluster = full_data[cluster_labels == i, :]
        # Determine the member of the cluster with the lowest energy and its hinge positions
        clustercenter = cluster[
            np.argmin(cluster[:, np.shape(dataset)[1]]), :np.shape(dataset)[1]
        ]
        clustercenters[i, :] = clustercenter  # Update the cluster centers with the lowest energy member

    # Visualization based on the specified `Plots` dimension
    if Plots == '1D':
        # 1D plot of cluster centers and their corresponding energies
        plt.figure()
        plt.scatter(
            km.cluster_centers_[:, 0], 
            km.cluster_centers_[:, 1], 
            color='#08d1f9', s=120, alpha=0.8, 
            edgecolors='black', linewidth=1, zorder=0
        )
        plt.xlabel('Hinge Angle')
        plt.ylabel('Energy')
        plt.grid()
        plt.show(block=True)

    elif Plots == '2D':
        # 2D plot of cluster centroids on the x-y plane
        plt.figure()
        plt.scatter(
            km.cluster_centers_[:, 0], 
            km.cluster_centers_[:, 1], 
            color='#08d1f9', s=120, alpha=0.8, 
            edgecolors='black', linewidth=1, zorder=0
        )
        plt.xlabel('Angle 1')
        plt.ylabel('Angle 2')
        plt.grid()
        plt.show(block=True)

    elif Plots == '3D':
        # 3D plot of cluster centroids on a Cartesian grid
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            km.cluster_centers_[:, 0], 
            km.cluster_centers_[:, 2], 
            color='#08d1f9', s=120, alpha=0.8, 
            edgecolors='black', linewidth=1, zorder=0
        )
        ax.set_xlabel('Angle 1')
        ax.set_ylabel('Angle 3')
        ax.set_zlabel('Angle 2')

    # Return the calculated cluster centers, labels, and number of clusters
    return clustercenters, cluster_labels, clusternumber

def min_cluster_centers_experimental(dataset, clusternumber, final_e, experimental_hinges, Plots='', max_iter=300, tol=1e-4, n_init=3):
    """
    Determine the center of each cluster and the points in the dataset that correspond to each cluster.
    Incorporates a fixed experimental hinge as one of the cluster centroids.

    Inputs:
        dataset: np.array, The dataset to be analyzed (equilibrium angle data).
                 Shape: (m x n), where m is the number of equilibrium states and n is the number of hinges.
        clusternumber: int, The total number of clusters to represent the data.
        final_e: np.array, A vector of final energies corresponding to all Monte Carlo (MC) simulations.
        experimental_hinges: np.array, A fixed hinge configuration to be used as one of the cluster centroids.
        Plots: str (optional), Specifies the dimension of the plot to generate ('1D', '2D', or '3D').
               Default is an empty string '', which means no plot will be generated.
        max_iter: int (optional), Maximum number of iterations for cluster updates. Default is 300.
        tol: float (optional), Tolerance for centroid convergence. Default is 1e-4.
        n_init: int (optional), Number of reinitializations for KMeans to avoid local minima. Default is 3.

    Returns:
        clustercenters: np.array, Centroids (center points) of each cluster.
        cluster_labels: np.array, Labels indicating which cluster each point in the dataset belongs to.
    """

    # Combine the dataset and final energies into a single array for easier manipulation
    full_data = np.zeros((np.shape(dataset)[0], np.shape(dataset)[1] + 1))  # Initialize combined array
    full_data[:, 0:np.shape(dataset)[1]] = dataset  # Fill with dataset values
    full_data[:, np.shape(dataset)[1]] = final_e  # Append final energies as the last column

    # Set the fixed experimental hinge as the first cluster centroid
    fixed_centroid = np.array([experimental_hinges])

    # Initialize variables for tracking the best clustering result
    best_inertia = np.inf  # Start with the highest possible inertia
    best_labels = None  # Placeholder for the best cluster labels
    best_centroids = None  # Placeholder for the best centroids

    # Perform clustering multiple times to avoid local minima
    for _ in range(n_init):
        # Initialize KMeans for (clusternumber - 1) clusters and fit the dataset
        kmeans = KMeans(n_clusters=clusternumber - 1, init='k-means++', n_init=1, random_state=0).fit(dataset)
        centroids = kmeans.cluster_centers_  # Extract initial centroids from KMeans

        # Add the fixed centroid to the list of centroids
        centroids = np.vstack([fixed_centroid, centroids])

        # Iterate to refine centroids
        for _ in range(max_iter):
            # Assignment step: Assign each data point to the closest centroid
            distances = np.linalg.norm(dataset[:, np.newaxis] - centroids, axis=2)  # Compute distances to centroids
            cluster_labels = np.argmin(distances, axis=1)  # Assign points to the nearest cluster

            # Update step: Recalculate centroids based on cluster membership
            new_centroids = np.array([
                dataset[cluster_labels == i].mean(axis=0) if i != 0 else fixed_centroid[0]  # Keep fixed centroid unchanged
                for i in range(clusternumber)
            ])

            # Convergence check: Stop if centroids do not significantly change
            if np.all(np.linalg.norm(centroids - new_centroids, axis=1) < tol):
                break  # Exit the iteration loop if convergence criteria are met

            centroids = new_centroids  # Update centroids for the next iteration

        # Calculate inertia (sum of squared distances to centroids) for the current clustering
        inertia = np.sum((dataset - centroids[cluster_labels]) ** 2)

        # Update the best clustering if the current inertia is the lowest so far
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = cluster_labels
            clustercenters = centroids

    # Handle the case where no data points are assigned to the fixed centroid cluster
    if len(full_data[best_labels == 0, :]) == 0:
        # Fall back to standard KMeans if the fixed centroid cluster is empty
        clustercenters = np.zeros((clusternumber, np.shape(dataset)[1]))
        km = KMeans(n_clusters=clusternumber, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=1)
        y_km = km.fit_predict(dataset)
        clustercenters = km.cluster_centers_
        cluster_labels = km.labels_
    else:
        # Use the best clustering results
        cluster_labels = best_labels
        clustercenters = centroids

    # Refine centroids by selecting the lowest-energy member within each cluster
    for i in range(clusternumber):
        cluster = full_data[cluster_labels == i, :]  # Extract data points belonging to cluster i
        clustercenter = cluster[np.argmin(cluster[:, np.shape(dataset)[1]]), :np.shape(dataset)[1]]  # Find the lowest-energy point
        clustercenters[i, :] = clustercenter  # Update the centroid to match the lowest-energy point

    # Ensure the fixed centroid remains unchanged
    clustercenters[0, :] = fixed_centroid[0, :]

    # Generate plots if requested
    if Plots == '1D':
        # 1D plot of centroids
        plt.figure()
        plt.scatter(clustercenters[:, 0], clustercenters[:, 1], s=100, marker='o', c='red', edgecolor='black', label='centroids')
        plt.xlabel('Hinge Angle')
        plt.ylabel('Energy')
        plt.grid()
        plt.show(block=True)

    elif Plots == '2D':
        # 2D plot of centroids
        plt.figure()
        plt.scatter(clustercenters[:, 0], clustercenters[:, 1], s=100, marker='o', c='red', edgecolor='black', label='centroids')
        plt.xlabel('Angle 1')
        plt.ylabel('Angle 2')
        plt.grid()
        plt.show(block=True)

    elif Plots == '3D':
        # 3D plot of centroids
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(clustercenters[:, 0], clustercenters[:, 2], clustercenters[:, 1], color='red')
        ax.set_xlabel('Angle 1')
        ax.set_ylabel('Angle 3')
        ax.set_zlabel('Angle 2')

    # Return the final cluster centers and labels
    return clustercenters, cluster_labels

def refine_experimental_labels(clustercenters, cluster_labels, clusternumber, final_hinges, final_e, angle_var=[], Plots=''):
    """
    Refine cluster assignments to ensure points within a specified variance of the experimental hinge
    are included in its cluster and recluster the remaining points.

    Inputs:
        clustercenters: np.array, Initial centroids of the clusters, where the first center is the experimental hinge.
        cluster_labels: np.array, Initial cluster labels for the data points.
        clusternumber: int, Total number of clusters to be formed.
        final_hinges: np.array, The dataset representing final hinge positions of each simulation.
        final_e: np.array, Final energies corresponding to the final_hinges.
        angle_var: np.array (optional), The allowed variance for each hinge angle to determine cluster inclusion.
                  Default is an empty list, meaning no variance is applied.
        Plots: str (optional), Specifies whether to generate plots ('1D', '2D', or '3D').

    Returns:
        clustercenters: np.array, Updated centroids of all clusters.
        cluster_labels: np.array, Updated cluster labels for the dataset.
        final_hinges: np.array, Updated hinge positions for all simulations.
        final_e: np.array, Updated energies for all simulations.
    """

    # Extract the experimental hinge center (first cluster center)
    center = clustercenters[0]
    # Extract the remaining cluster centers
    remaining_centers = clustercenters[1:]
    # Standard deviations or tolerances for each hinge angle
    std_deviation = angle_var

    # Identify which data points (final_hinges) are within the allowed variance from the experimental hinge
    # `condition` is a boolean array where True indicates inclusion in the experimental hinge cluster
    condition = np.all(np.abs(final_hinges - center) <= std_deviation, axis=1)

    # Assign points meeting the condition to the experimental hinge cluster
    assigned_hinges = final_hinges[condition]
    # Points not meeting the condition are retained for reclustering
    remaining_hinges = final_hinges[~condition]

    # Separate the corresponding energy values
    assigned_e = final_e[condition]  # Energies of points assigned to the experimental hinge cluster
    remaining_e = final_e[~condition]  # Energies of points left for reclustering

    # Generate labels for the points assigned to the experimental hinge cluster (all zeros)
    extra_labels = np.zeros(len(assigned_hinges))

    # Update the original cluster_labels to reflect the reassignment of points to the experimental hinge cluster
    cluster_labels[condition] = 0

    # Reclustering the remaining points using the updated cluster number and dataset
    new_clustercenters, new_cluster_labels, new_clusternumber = min_cluster_centers(
        remaining_hinges, 
        clusternumber - 1,  # Reduce the cluster number since the experimental hinge cluster is fixed
        remaining_e, 
        Plots=Plots
    )

    # Update the cluster centers to include the fixed experimental hinge center
    clustercenters = np.vstack((center, new_clustercenters))

    # Update cluster labels to merge the experimental hinge labels with the new cluster labels
    cluster_labels = np.hstack((extra_labels, new_cluster_labels + 1))  # Offset new labels by 1 to account for the fixed cluster

    # Recombine the assigned and remaining hinge datasets
    final_hinges = np.vstack((assigned_hinges, remaining_hinges))
    final_e = np.hstack((assigned_e, remaining_e))

    # Update the total cluster number
    clusternumber = new_clusternumber + 1

    # Return the updated cluster information
    return clustercenters, cluster_labels, final_hinges, final_e

def cluster_stats(data, clusternumber, clustercenters, clusterlabels, final_e, plot=True, blocking=False):
    """
    This function processes the clusters and performs statistical analysis, calculating their standard deviations
    about each hinge and determining the fraction of simulations ending in each cluster (probability). Optionally, it can
    plot these probabilities as histograms.

    Inputs:
        data: np.array, dataset to be analyzed (equilibrium angles or similar).
        clusternumber: int, number of clusters.
        clustercenters: np.array, centroids of the clusters.
        clusterlabels: np.array, list assigning each final conformation to a cluster.
        final_e: np.array, final energies of each equilibrium conformation.
        plot: bool (optional), determines whether to plot the histogram.
        blocking: bool (optional), whether to block the script during plotting.

    Returns:
        clusternumbers: np.array, sorted labels assigned to each cluster.
        cluster_count: np.array, vector of the number of points in each cluster.
        cluster_prob: np.array, vector of probabilities of each fold (normalized cluster sizes).
        energies: np.array, vector of average final energies for each cluster.
        ordered_centers: np.array, sorted cluster centers by increasing energy.
        all_stds: np.array, array of standard deviations for hinges in each cluster.
    """

    # Initialize arrays and variables to store results
    cluster_std = np.zeros(clusternumber)  # Standard deviations for each cluster
    cluster_count = np.zeros(clusternumber)  # Number of points in each cluster
    hist_keys = [''] * clusternumber  # Keys for histogram annotations
    energies = np.zeros(clusternumber)  # Average final energy for each cluster
    # Sorting array: rows store various cluster attributes (energy, size, etc.)
    sort_array = np.zeros((4 + np.shape(clustercenters)[1] + np.shape(clustercenters)[1], clusternumber))
    sort_array[4:(4 + np.shape(clustercenters)[1]), :] = clustercenters.T  # Assign cluster centers to sorting array

    # Process each cluster
    for i in range(clusternumber):
        # Extract data points belonging to the current cluster
        cluster = data[clusterlabels == i]
        # Calculate deviations of points from the cluster center
        r = cluster - clustercenters[i, :]
        hinge_std = np.std(r, axis=0)  # Standard deviation for each hinge
        cluster_count[i] = len(cluster)  # Number of points in this cluster
        # Extract and calculate average energy for this cluster
        e_tot = final_e[clusterlabels == i]
        energies[i] = np.mean(e_tot) / 1.381e-23 / 298  # Scale by kT

        # Populate sorting array with cluster properties
        sort_array[0, i] = np.mean(e_tot)  # Mean energy
        sort_array[2, i] = i  # Cluster number
        sort_array[3, i] = cluster_count[i]  # Number of simulations in the cluster
        sort_array[(4 + np.shape(clustercenters)[1]):, i] = hinge_std.T  # Standard deviations for hinges

    # Calculate probabilities of each final state (normalized cluster sizes)
    cluster_prob = cluster_count / sum(cluster_count)
    sort_array[1, :] = cluster_prob  # Assign probabilities to sorting array

    # Sort clusters by their mean energy
    sort_array = sort_array[:, sort_array[0, :].argsort()]
    energies = sort_array[0, :]  # Sorted energies

    # Create keys for the histogram indicating energy of each cluster
    for i in range(len(energies)):
        hist_keys[i] = '{:0.3e}'.format(energies[i] / 1.381e-23 / 298)  # Scientific notation scaled by kT

    # Extract sorted cluster properties
    cluster_prob = sort_array[1, :]  # Sorted probabilities
    clusternumbers = sort_array[2, :]  # Sorted cluster labels
    cluster_std = sort_array[3, :]  # Standard deviations for each cluster
    cluster_count = sort_array[4, :]  # Cluster sizes

    # Optional plotting
    if plot:
        norm = plt.Normalize(min(energies), max(energies))
        cmap = plt.get_cmap('viridis')  # Colormap for energy values

        # Create a bar graph
        plt.figure(figsize=(6, 5))
        colors = cmap(norm(energies))  # Map energies to colors
        x_pos = np.arange(len(energies))  # X-axis positions for bars

        # Plot histogram of cluster probabilities
        plt.bar(x_pos, cluster_prob * 1000, width=0.85, color=colors, edgecolor='black', linewidth=3)
        plt.ylabel('Folded State Frequency', fontsize=8)
        plt.xlabel('Final Energy (kT)', fontsize=8)

        # Add a color bar to indicate energy scale
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label='Final Energy (kT)')

        # Enhance plot aesthetics with borders
        for spine in plt.gca().spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        plt.xticks([])  # Remove x-tick labels (optional)
        plt.show(block=blocking)

    # Return sorted and processed cluster information
    ordered_centers = sort_array[4:(4 + np.shape(clustercenters)[1]), :].T
    all_stds = sort_array[(4 + np.shape(clustercenters)[1]):, :].T

    return clusternumbers, cluster_count, cluster_prob, energies, ordered_centers, all_stds

def show_probable_structures(hingenum, shapes, clustercenters, blocking=False):
    """
    Function displays the final conformation of the sequence in each cluster based on the minimum-energy centroid of that cluster.

    Inputs:
        hingenum: list, indices of moving hinges.
        shapes: dictionary, all shapes and their initial configurations.
        clustercenters: np.array, centroid points of each cluster.
        blocking: bool (optional), determines whether the plots block the script execution.

    Returns:
        None. Displays plots of the initial state and each cluster's final conformation.
    """
    
    final_blocking = False  # Determine if the last plot blocks the script

    # Generate the initial state of all shapes
    hinge_vec, shape_arr, sequence, linelist, patch_arr = generate(shapes)
    shapeplots(shape_arr, linelist)  # Display the initial orientation of the system

    # Loop through each cluster center
    for i in range(len(clustercenters)):
        # Reset to the initial state for each cluster's visualization
        hinge_vec, shape_arr, sequence, linelist, patch_arr = generate(shapes)

        # Determine the number of hinges to iterate over for the current cluster
        if len(hingenum) == 1:
            cluster_index = 1  # Only one hinge to adjust
        else:
            cluster_index = np.shape(clustercenters)[1]  # Number of hinges in the centroid

        # Loop through each hinge angle in the centroid point
        for j in range(cluster_index):
            hingechoice = hingenum[j]  # Select the hinge corresponding to the centroid point
            # Calculate the angle adjustment required to match the centroid's hinge position
            angle = clustercenters[i, j] - hinge_vec[hingechoice]
            # Rotate the hinge by the calculated angle
            patch_arr, shape_arr, hinge_vec = rotate_once(
                patch_arr, shape_arr, linelist, hinge_vec, hingechoice, hingenum, angle
            )

        # Determine if the final cluster should block script execution
        if i == len(clustercenters) - 1:
            final_blocking = blocking

        # Plot the final conformation of the current cluster
        shapeplots(shape_arr, linelist, title=str(i), blocking=final_blocking)

def define_savedict():
    """
    Function to define a dictionary template for storing simulation data.

    This dictionary will hold various attributes and results from simulations, 
    with keys corresponding to relevant parameters and values initialized to `None`.

    Returns:
        savedict: dict, a dictionary with predefined keys for simulation data, all initialized to `None`.
    """

    # List of column names representing keys for the dictionary.
    # Each key corresponds to a specific parameter or result from the simulation.
    columnlist = [
        'shapes',             # Geometries or structures used in the simulation
        'length',             # Length of the structures
        'magnetization',      # Magnetization values
        'initial angle',      # Initial angles of the hinges or components
        'std',                # Standard deviation of the angles or parameters
        'E_ref',              # Reference energy value
        'E_low',              # Lowest energy achieved in the simulation
        'anneal steps',       # Number of annealing steps performed
        'hinge number',       # Number of hinges in the structure
        'hinges',             # List or array of hinge configurations
        '% triangles',        # Percentage of triangular configurations
        'patch strengths',    # Strengths of patches used in the simulation
        'patch numbers',      # Number of patches
        'final hinges',       # Final hinge configurations after the simulation
        'probability',        # Probability of reaching certain states
        'energies',           # Energy values for different states
        'hinge stds',         # Standard deviation of hinge configurations
        'clusternumber',      # Number of clusters in the simulation results
        'anneal cutoff',      # Cutoff criteria for the annealing process
        'equilibrium cutoff'  # Cutoff criteria for equilibrium states
    ]

    # Create a dictionary with keys from columnlist and initialize all values to None
    savedict = dict.fromkeys(columnlist, None)

    # Return the initialized dictionary template
    return savedict

def parallellel_powerful_forward(moves, std, m, shapes, initial_angle, anneal_steps, r_sims, cluster_max, anneal_cutoff, equilibrium_cutoff, savedict, overlap_val = []):
    '''
    This function serves as the worker function for the megasim functions. It calculates the energy criterion, runs the annealing and Monte Carlo simulations, 
    performs clustering analysis, and returns relevant measurable data, including the final cluster labels and ordered equilibrium structures.

    Inputs:
        moves: number of maximum Monte Carlo (MC) moves per simulation
        std: standard deviation of the Gaussian distribution used to draw angles
        m: list of magnetization/length values for each patch. 
           Typically, moment/length = 1.01E-06 A*m for a 100 nm cobalt patch
        shapes: dictionary containing shape configurations and side lengths
        initial_angle: the initial angle to rotate the shapes for characteristic energy estimation
        overlap_val: (optional) list specifying overlap values for AB junctions to match experimental conditions
        anneal_steps: the number of annealing steps
        r_sims: number of Monte Carlo simulations to run
        cluster_max: the maximum number of clusters to look for in the analysis
        anneal_cutoff: energy threshold for the annealing process
        equilibrium_cutoff: energy threshold for the equilibrium state detection
        savedict: dictionary to store the results from the simulation
    Returns:
        savedict: the dictionary with the relevant simulation results, including cluster labels, equilibrium energies, and hinge data
    '''

    warnings.filterwarnings('ignore')  # Ignore specific warning from Kmeans clustering

    # Generate initial shape and energy setup
    hinge_vec, shape_arr, sequence, linelist, patch_arr = generate(shapes)
    triangles = linelist.count(3)  # Count the number of triangles in the shape configuration
    hingenum = moving_hinges(sequence)  # Identify which hinges can move
    nonhinges = not_hinges(hinge_vec, hingenum)  # Get the hinges that remain static
    shape_ind, patch_ind, changes = moving_indices(shape_arr, patch_arr, linelist, nonhinges, hingenum)

    # Apply specific overlap settings if provided
    if len(overlap_val) != 0:
        shape_arr, patch_arr = vary_specific_overlap(shape_arr, patch_arr, shape_ind, patch_ind, changes, overlap_val)
        change_overlap = False  # Overlap values are specified, no need for random overlap changes
    else:
        change_overlap = True  # Allow overlap to change randomly in the absence of specific values

    # Count the shapes in the configuration
    polycount = count_shapes(shape_arr)

    # Initialize magnetization vectors and energy matrices
    magvec = mag_vectors(sequence, m)
    mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat = initialize_energy(magvec)

    # Calculate the reference and low energies for characteristic energy estimation
    E_refs, E_lows = characteristic_energy(std, sequence, patch_arr, shape_arr, hinge_vec, linelist, 
                                           polycount, initial_angle, mask_arr, v_xmat, h_xmat, v_ymat,
                                           h_ymat, Ml_mat)
    
    E_ref_initial = np.mean(E_refs)  # Initial reference energy
    E_low = np.mean(E_lows)  # Initial low energy
    anneal_rate = E_ref_initial / anneal_steps  # Calculate annealing rate

    # Run Monte Carlo simulations to sample equilibrium configurations
    start2 = time.time()  # Start timer for simulation
    final_hinges, final_e = runrun_parallel_forward(
        patch_arr, shape_arr, hinge_vec, hingenum, linelist, polycount, moves, std, 
        E_ref_initial, E_low, anneal_rate, r_sims, mask_arr, v_xmat, h_xmat, v_ymat, 
        h_ymat, Ml_mat, shape_ind, patch_ind, changes, anneal_cutoff, equilibrium_cutoff, change_overlap)
    time2 = time.time() - start2  # Calculate time taken for simulations

    # Perform clustering analysis on the final simulation results
    clusternumber = cluster_num(final_hinges, cluster_max, 3, Plot=False)  # Estimate the number of clusters
    clustercenters, cluster_labels, clusternumber = min_cluster_centers(final_hinges, clusternumber, final_e)  # Calculate cluster centers
    clusternumbers, cluster_count, cluster_prob, energies, ordered_centers, all_stds = cluster_stats(
        final_hinges, clusternumber, clustercenters, cluster_labels, final_e, plot=False)

    # Store simulation results into the provided savedict dictionary
    savedict['shapes'] = shapes  # Store the shapes configuration
    savedict['length'] = len(shapes)  # Store the length of the shapes
    savedict['magnetization'] = m  # Store the magnetization/length values
    savedict['initial angle'] = initial_angle  # Store the initial angle
    savedict['std'] = std  # Store the standard deviation of the Gaussian distribution
    savedict['E_ref'] = E_ref_initial  # Store the initial reference energy
    savedict['E_low'] = E_low  # Store the initial low energy
    savedict['anneal steps'] = anneal_steps  # Store the number of anneal steps
    savedict['hinge number'] = len(hingenum)  # Store the number of hinges
    savedict['hinges'] = hingenum.tolist()  # Store the hinges involved in the simulation
    savedict['% triangles'] = triangles / len(shapes)  # Store the percentage of triangles in the shape
    savedict['final hinges'] = ordered_centers  # Store the ordered equilibrium structures
    savedict['probability'] = cluster_prob  # Store the probability distribution of clusters
    savedict['energies'] = energies  # Store the energies of each cluster
    savedict['hinge stds'] = all_stds  # Store the standard deviations of hinge angles in each cluster
    savedict['clusternumber'] = clusternumber  # Store the number of clusters
    savedict['anneal cutoff'] = anneal_cutoff  # Store the annealing cutoff energy
    savedict['equilibrium cutoff'] = equilibrium_cutoff  # Store the equilibrium cutoff energy
    
    return savedict  # Return the dictionary containing all the simulation results

def vary_structure(shape_options, orientation_options, len_options, chain_len):
    '''This function varies the structure based on a series of given options. Specifically, it is used to vary the length of the structure in run_all_megasims, but it also allows 
    for the creation of completely random and unique structures based on input information.

    Inputs:
        shape_options: List of allowable shapes to be included in the structure.
        orientation_options: List of allowable orientations ('A' or 'B') for each shape in the structure.
        len_options: List of allowable side lengths for the shapes.
        chain_len: The number of shapes to include in the chain.

    Outputs:
        shapes: A dictionary object that holds the shape, orientation, and length of each shape under the corresponding shape's key.
                This dictionary can be used directly in the generate function to create the relevant arrays for simulation.
    '''
        
    key_list = []  # Initialize a list to store the keys for the shapes dictionary

    # Create keys for each shape in the chain
    for i in range(chain_len):  
        key_list.append('shape ' + str(i+1))  # Add keys like 'shape 1', 'shape 2', ..., 'shape N'

    # Initialize an empty dictionary with the keys and None as initial values
    shapes = dict.fromkeys(key_list, None)

    # For each key (representing a shape in the chain)
    for key in shapes.keys():
        infolist = []  # Initialize a list to store information for the current shape (shape, orientation, length)
        
        # Randomly select a shape from the shape_options
        shape = np.random.choice(shape_options)
        
        # Randomly select an orientation from the orientation_options
        orientation = np.random.choice(orientation_options)
        
        # Randomly select a length from the len_options
        length = np.random.choice(len_options)
        
        # Append the shape, orientation, and length to the infolist
        infolist.append(shape)
        infolist.append(orientation)
        infolist.append(int(length))  # Ensure length is an integer
        
        # Assign the infolist to the dictionary entry for the current shape
        shapes[key] = infolist

    # Return the dictionary containing all the shape data
    return shapes

def vary_shapes(shape_options,shape_nums,orientation_options,len_options,chain_len):
    '''This function is intended to vary the sequence with specific ratios of shapes in order to determnine the effect that shape has on folding structure. It works by making a list of unchosen shapes for
    the number of shapes desired in the chain. It determines the number of shapes of each kind that should be in the chain based on the given percentages and draws that number of shapes from the unchosen
    shapes at random in order to palce those shapes in random spots
    Inputs:
        shape_options: list of shapes allowed in the structure
        shape_nums: list of the same length as shape_options where each entry is the number of shapes in the chain of that respective shape
        orientation_options: list of what orientations 'A' or 'B' are to be allowed
        len_options: list of allowable sidelengths of each shape
        chain_len: list of number of allowable shapes in the chain
    Outputs:
        Shapes: A dictionary object that holds the shape, orientation, and length of a shape under that shapes key. This can be used directly in the generate function to create the relevant shape and line arrays
    '''
    
    key_list = [] #Initialize list of keys
    for i in range(chain_len): #for each shape
        key_list.append('shape ' + str(i+1)) #Create a key for that shape and ad dit to the key list
    
    shapes = dict.fromkeys(key_list,None) #Initialize a dictionary object with the shapes as keys
    num_shapes = [] #Initialize num_shapes list to hold the number of each shape in the chain
    for i in range(len(shape_options)): #For all possible shapes
        shape_num = shape_nums[i] #The number of shapes of that kind in the chain should be the total chain length multiplied by the percentage makeup of that shape in the chain
        num_shapes.append(shape_num) #Add the number of a particular shape in the chain to num_shapes
    unchosen_shapes = [i for i in range(chain_len)] #Initialize a list of unchosen shapes indexed 0,1,2,... for the number of shapes in the chain
    shapelist = [shape_options[0]]*chain_len #Initialize shapelist to be the length of the number of shapes, but all the first shape listed in shape_options

    for i in range(len(shape_options[1:])): #For each different type of shape. Neglect the first entry in shape_options because shapelist is currently made up entirely of that shape
        for j in range(int(num_shapes[i+1])): #For each shape except for the first shape. Again, indexing starts afte the fist since shapelist is currently made entirely of the first shape
            random_shape = np.random.choice(unchosen_shapes) #Choose a random shape from unchosen_shapes
            shapelist[random_shape] = shape_options[i+1] #Update the shapelist of that unchosen_shape entry to be the current shape of interest
            unchosen_shapes.remove(random_shape) #remove taht index from unchosen shapes so that it cannot be chosen again

    index = 0 #Initialize index counter to keep track of the shapelist entries
    for key in shapes.keys(): #For each key in the dictionary
        infolist = [] #Initialize infolist
        shape = shapelist[index] #grab the shape from that entry of shapelist
        orientation = np.random.choice(orientation_options) #Choose a random orientation for each shape in the structure
        length = np.random.choice(len_options) #Choose a random length for the shape from len_options
        infolist.append(shape) #Add the shape to the infolist
        infolist.append(orientation) #Add the orientation to the infolist
        infolist.append(int(length)) #Add the patch length to the infolist
        shapes[key] = infolist #Add the infolist to the disctionary entry for the current shape
        index +=1 #Increment the index counter

    return shapes

def generate_sequence(orientation_options, chain_len):
    '''This function generates a random sequence of orientations for a chain of shapes.

    Inputs:
        orientation_options: List of allowable orientations (e.g., ['A', 'B']) for each shape in the chain.
        chain_len: The number of shapes in the chain for which the orientation sequence is generated.

    Outputs:
        sequence: A list containing the randomly selected orientations for each shape in the chain.
                  The length of the sequence corresponds to the number of shapes in the chain (chain_len).
    '''

    sequence = []  # Initialize an empty list to hold the orientation sequence
    
    # Loop over the desired chain length and randomly choose an orientation for each shape
    for i in range(chain_len):
        orientation = np.random.choice(orientation_options)  # Randomly select an orientation from the given options
        sequence.append(orientation)  # Add the selected orientation to the sequence list

    # Return the generated sequence of orientations
    return sequence

def vary_only_shapes(shape_options, shape_nums, len_options, chain_len, sequence):
    '''This function varies the shapes in a chain of shapes while keeping the orientations and lengths of the shapes random.
    The number of each shape is specified, and their distribution is used to create a chain with randomized shape types, lengths, and orientations.

    Inputs:
        shape_options: List of available shape types to choose from.
        shape_nums: List of integers specifying the number of each shape type to be included in the chain.
        len_options: List of allowable lengths for each shape in the chain.
        chain_len: Total number of shapes in the chain.
        sequence: Predefined orientation sequence for each shape in the chain.

    Outputs:
        shapes: A dictionary where each key corresponds to a shape in the chain, and the value contains the shape's type, orientation, and length.
    '''
    
    key_list = []  # Initialize a list to hold the keys for the dictionary
    for i in range(chain_len):  # Loop over the chain length
        key_list.append('shape ' + str(i+1))  # Create keys for each shape in the chain and append to key_list

    shapes = dict.fromkeys(key_list, None)  # Initialize a dictionary with keys corresponding to the shape names, with None as default values
    num_shapes = []  # List to hold the number of each shape type in the chain

    for i in range(len(shape_options)):  # Loop over all possible shapes in shape_options
        shape_num = shape_nums[i]  # Determine how many shapes of the current type should be in the chain
        num_shapes.append(shape_num)  # Add the count of the current shape type to num_shapes

    unchosen_shapes = [i for i in range(chain_len)]  # Create a list of indices (0 to chain_len-1) representing unchosen shapes in the chain
    shapelist = [shape_options[0]] * chain_len  # Start by assigning all shapes in the chain to the first shape in shape_options

    for i in range(len(shape_options[1:])):  # Loop over each remaining shape type in shape_options (except the first one)
        for j in range(int(num_shapes[i+1])):  # Loop over the number of times the current shape should appear in the chain
            random_shape = np.random.choice(unchosen_shapes)  # Choose a random index from unchosen_shapes
            shapelist[random_shape] = shape_options[i+1]  # Assign the current shape type to the chosen index in shapelist
            unchosen_shapes.remove(random_shape)  # Remove the index from unchosen_shapes to prevent it from being chosen again

    index = 0  # Initialize an index counter to keep track of which entry in shapelist is being processed
    for key in shapes.keys():  # Loop through each key in the shapes dictionary (representing the shapes in the chain)
        infolist = []  # Initialize an empty list to hold the shape's properties
        shape = shapelist[index]  # Get the shape type from the shapelist at the current index
        orientation = sequence[index]  # Get the corresponding orientation from the sequence
        length = np.random.choice(len_options)  # Randomly select a length from len_options for the current shape
        infolist.append(shape)  # Add the shape to the infolist
        infolist.append(orientation)  # Add the orientation to the infolist
        infolist.append(int(length))  # Add the length to the infolist
        shapes[key] = infolist  # Add the infolist to the shapes dictionary for the current key (shape)
        index += 1  # Increment the index counter to move to the next shape in the chain

    return shapes  # Return the generated dictionary containing the shapes and their properties

def vary_m(strength_percent, patch_nums, chain_len, mval):
    '''This function randomizes the strength of the magnetic patch on each shape in the chain.
    The patch strength for each shape is determined by a relative percentage of the initial magnetic strength (mval), 
    and the distribution of patch types is controlled by patch_nums.

    Inputs:
        strength_percent: List of relative percentages (as decimals) that determine how strong each patch is compared to mval.
        patch_nums: List of integers specifying the number of patches of each type (as defined by strength_percent) in the chain.
        chain_len: The total number of shapes in the chain (i.e., the number of patches).
        mval: The base magnetic strength of the patch, which is multiplied by the values in strength_percent to determine the final magnetic strength.

    Outputs:
        m: A list of magnetization values for each patch, where each entry corresponds to the strength of a specific patch in the chain.
    '''

    m = [mval] * chain_len  # Initialize the list of magnetizations with the base strength (mval) for all patches
    unchosen_shapes = [i for i in range(chain_len)]  # Create a list of indices representing unchosen patches in the chain
    
    for i in range(len(strength_percent)):  # Loop over each patch type (defined by strength_percent)
        for j in range(int(patch_nums[i])):  # Loop over the number of patches of this type
            random_shape = np.random.choice(unchosen_shapes)  # Randomly select a shape (index) from unchosen_shapes
            m[random_shape] = strength_percent[i] * mval  # Assign the magnetization strength for this patch (scaled by strength_percent)
            unchosen_shapes.remove(random_shape)  # Remove the selected index from unchosen_shapes to prevent it from being chosen again

    return m  # Return the list of magnetization values for each patch

def define_mainframe(l_sims, l_number_options, s_sims, s_number_options, m_sims, m_number_options):
    '''This function initializes a pandas DataFrame to store the results of various simulations. 
    It generates unique indices for each simulation sequence and defines the column names for the data to be collected during the simulations.

    Inputs:
        l_sims: int
            The number of replicate simulations to perform for each chain length in the study.
        l_number_options: list
            A list containing the different chain lengths to be examined in the length simulations.
            Each length will be tested with `l_sims` replicates.
        s_sims: int
            The number of replicate simulations to perform for each number of shapes (triangles) considered in the shape simulations.
        s_number_options: list
            A list of shape options (e.g., number of triangles) to be considered during the shape simulations.
            Each option will be tested with `s_sims` replicates.
        m_sims: int
            The number of replicate simulations to perform for each number of magnetizations considered in the magnetization simulations.
        m_number_options: list
            A list of magnetization options (e.g., number of patches) to be considered during the magnetization simulations.
            Each option will be tested with `m_sims` replicates.

    Outputs:
        mainframe: pandas.DataFrame
            A DataFrame initialized with appropriate column names and index entries, designed to store the results of all simulations.
            This DataFrame will store data for all simulations (length, shape, magnetization combinations).
        indexlist: list
            A list containing the index labels for each simulation sequence (e.g., 'Sequence 1', 'Sequence 2', ...), 
            corresponding to each unique combination of parameters tested during the simulations.
    '''
    
    # Initialize the index list for the DataFrame, representing each simulation sequence (e.g., 'Sequence 1', 'Sequence 2', ...)
    indexlist = []

    # Generate index labels based on the total number of simulations for all simulation types (length, shape, and magnetization)
    for i in range(l_sims * len(l_number_options) + s_sims * len(s_number_options) + m_sims * len(m_number_options)):
        indexlist.append('Sequence ' + str(i + 1))  # Create an index entry for each simulation sequence

    # Define the column names that will be used to store simulation data for each sequence
    columnlist = ['shapes', 'length', 'magnetization', 'initial angle', 'std', 'E_ref', 'E_low', 'anneal steps', 'hinge number', 'hinges',
                  '% triangles', 'patch strengths', 'patch numbers', 'final hinges', 'probability', 'energies', 'hinge stds', 'clusternumber',
                  'anneal cutoff', 'equilibrium cutoff']
    
    # Create a pandas DataFrame with the generated index and column names
    mainframe = pd.DataFrame(columns=columnlist, index=indexlist)

    # Return the initialized DataFrame (mainframe) and the index list (indexlist) for use in further simulations
    return mainframe, indexlist

def megasim_varylength(moves, std, mval, initial_angle, anneal_steps, l_sims, r_sims, cluster_max, number_options, shape_options, orientation_options,
                       len_options, indexlist, mainframe, savedict, count, anneal_cutoff, equilibrium_cutoff):
    '''This function runs Monte Carlo simulations for different chain lengths as defined in `number_options`. It generates unique structures for each chain length,
    runs the folding process using Monte Carlo simulations, and stores the resulting data for analysis. Each structure is randomly generated and subjected to multiple 
    Monte Carlo simulations to determine various physical properties, which are recorded in a DataFrame.

    Inputs:
        moves: int
            The maximum number of allowable moves during each Monte Carlo step.
        std: float
            Standard deviation for drawing a random angle from a Gaussian distribution to move a hinge at each step.
        mval: float
            The strength of the magnetic patch, typically in the order of 1.01E-6.
        initial_angle: float
            The initial angle used to rotate the structure, determining its characteristic energy.
        anneal_steps: int
            The number of steps to be applied when performing annealing on the structure during the Monte Carlo process.
        l_sims: int
            The number of random structures to generate and simulate for each chain length.
        r_sims: int
            The number of times the Monte Carlo simulation is performed for each generated structure.
        cluster_max: int
            The maximum number of clusters to consider and analyze in the system.
        number_options: list
            A list of chain lengths to be examined in the simulations. Each length will be simulated with `l_sims` number of random structures.
        shape_options: list
            A list of allowable shapes (e.g., types of polygons) to be used in each structure.
        orientation_options: list
            A list of allowable orientations for the shapes generated within each structure.
        len_options: list
            A list of allowable side lengths for each shape to be considered.
        indexlist: list
            A list containing indices for each sequence (e.g., 'Sequence 1', 'Sequence 2', ...) for storing simulation results.
        mainframe: pandas.DataFrame
            A DataFrame used to store all simulation data for later analysis.
        savedict: dict
            A dictionary used to store the calculated measurables and results from each simulation.
        count: int
            A counter to track the current simulation number in the process.
        anneal_cutoff: float
            A threshold value used to determine when the annealing process should stop.
        equilibrium_cutoff: float
            A threshold value to determine when equilibrium has been reached during the simulation.

    Outputs:
        mainframe: pandas.DataFrame
            A DataFrame containing the results of all simulations, indexed by sequence number and with columns for various physical properties.
        count: int
            The updated simulation counter after processing all simulations, indicating the number of simulations completed.
    '''
    
    # Iterate over each chain length defined in `number_options`
    for i in number_options:
        shapelist = []  # Initialize a list to store all generated unique structures for the current chain length
        print(f'Simulating Chains of Length {i}\n')
        
        # Perform `l_sims` number of simulations for each chain length
        for j in range(l_sims):
            print(f'Simulation #{j}')

            # Ensure that each structure generation and simulation runs smoothly, retrying in case of failure
            while True:
                try:
                    go_on = 0  # Flag to control the process of generating valid structures
                    shapes = {}  # Initialize a dictionary to store the generated shape data
                    
                    # Generate random structures and validate them until a valid one is found
                    while go_on == 0:
                        # Generate a random structure with the given options
                        shapes = vary_structure(shape_options, orientation_options, len_options, i)
                        
                        # Check if the structure has already been generated (i.e., is a duplicate)
                        if str(shapes) in shapelist:
                            go_on = 0  # Retry if the structure is a duplicate
                        # Ensure that at least one hinge is movable (i.e., valid structure)
                        elif len(moving_hinges(sequence_from_shapes(shapes))) == 0:
                            go_on = 0  # Retry if no hinges can move
                        else:
                            go_on = 1  # Continue if a valid structure is generated
                    
                    # Initialize the magnetization list for the structure, assigning the same magnetization value for each part of the structure
                    m = [mval] * len(shapes)  # Assign initial magnetization to each shape in the structure
                    
                    # Record the strength of the magnetic patch in savedict
                    savedict['patch strengths'] = mval / 1.01e-6  # Normalize by reference value
                    savedict['patch percentages'] = 1  # Assuming 100% for simplicity

                    # Run the Monte Carlo simulation for the generated structure and save the results in savedict
                    savedict = parallellel_powerful_forward(moves, std, m, shapes, initial_angle, anneal_steps, r_sims, cluster_max, anneal_cutoff, equilibrium_cutoff, savedict)

                    break  # Exit the while loop if no errors occurred during the simulation
                except Exception:
                    # If any error occurs, retry generating the structure and rerun the simulation
                    continue

            # After successful simulation, store the generated structure in shapelist
            shapelist.append(str(shapes))  
            
            # Store the results of the simulation (from savedict) in the mainframe DataFrame at the appropriate index
            mainframe.loc[indexlist[count]] = savedict  
            count += 1  # Increment the simulation counter

    return mainframe, count  # Return the updated DataFrame and the simulation counter

def megasim_varyonlyshapes(moves, std, mval, initial_angle, anneal_steps, s_sims, r_sims, cluster_max, chain_len, shape_options, orientation_options,
                            len_options, shape_numbers, indexlist, mainframe, savedict, count, anneal_cutoff, equilibrium_cutoff):
    '''This function runs Monte Carlo simulations for varying shapes (triangular patches) in a chain, varying the composition of shapes across different simulations.
    It generates unique sequences of shapes for each simulation and simulates their folding using the Monte Carlo method. The results are stored in a pandas DataFrame.

    Inputs:
        moves: int
            The maximum number of allowable moves in each Monte Carlo simulation step.
        std: float
            Standard deviation for random angle generation in the Monte Carlo process.
        mval: float
            Strength of the magnetic patch, typically around 1.01E-6.
        initial_angle: float
            The initial angle used to determine the characteristic energy of the structure.
        anneal_steps: int
            The number of steps for the annealing process in the Monte Carlo simulation.
        s_sims: int
            The number of simulations to run, each with a different shape configuration.
        r_sims: int
            The number of times each structure undergoes a Monte Carlo simulation.
        cluster_max: int
            The maximum number of clusters to be considered in the system during the simulation.
        chain_len: int
            The length of the chain (number of shapes) to be simulated.
        shape_options: list
            A list of possible shapes that can be included in the structure.
        orientation_options: list
            A list of possible orientations for the shapes in the structure.
        len_options: list
            A list of allowable side lengths for the shapes.
        shape_numbers: list
            A list of the percentages of each type of shape (e.g., the proportion of triangles).
        indexlist: list
            A list of indices to store results for each simulation.
        mainframe: pandas.DataFrame
            A DataFrame to store the results of all simulations.
        savedict: dict
            A dictionary that will hold the simulation results (e.g., energies, magnetization) for each run.
        count: int
            A counter to track the current simulation number.
        anneal_cutoff: float
            A cutoff value for stopping the annealing process.
        equilibrium_cutoff: float
            A threshold value to determine when equilibrium is reached in the simulation.

    Outputs:
        mainframe: pandas.DataFrame
            The DataFrame containing the results of all simulations, indexed by sequence number.
        count: int
            The updated simulation counter, incremented after each successful simulation.
    '''
    
    # Initialize lists and dictionaries to keep track of sequences and their corresponding shape configurations
    sequencelist = []
    sequencedict = {}
    
    # Initialize the dictionary for sequence simulations, where the key is the simulation number
    for l in range(s_sims):
        sequencedict[l] = 0

    # Generate random sequences of shapes for each simulation
    for i in range(s_sims):
        go_on = 0
        while go_on == 0:  # Continue until a valid, unique structure is found
            # Generate a random sequence of shapes for the structure
            sequence = generate_sequence(orientation_options, chain_len)
            
            # Ensure the sequence is unique and has at least one movable hinge
            if str(sequence) in sequencelist:
                go_on = 0  # Retry if the sequence already exists
            elif len(moving_hinges(sequence)) == 0:
                go_on = 0  # Retry if there are no movable hinges
            else:
                sequencelist.append(str(sequence))  # Store the valid sequence
                sequencedict[i] = sequence  # Store the sequence in the dictionary
                go_on = 1  # Proceed with this valid sequence

    # Loop through each shape composition (i.e., varying the number of triangles)
    for i in shape_numbers:
        # Define the number of triangles and the remaining shapes in the sequence
        shape_nums = [i, chain_len - i]  # Composition of shapes: i triangles and the rest other shapes
        print(f'Simulating Sequences with {i} Triangles\n')

        # Simulate for the desired number of samples at this specific shape composition
        for j in range(s_sims):
            print(f'Simulation #{j}')
            
            # Retry the simulation until it runs successfully
            while True:
                try:
                    # Generate the structure using the defined shapes, proportions, and length
                    shapes = vary_only_shapes(shape_options, shape_nums, len_options, chain_len, sequencedict[j])

                    # Initialize magnetization list and assign the patch strength
                    m = [mval] * len(shapes)  # Assign the same magnetization for each part of the shape
                    savedict['patch strengths'] = mval / 1.01e-6  # Normalize the patch strength
                    savedict['patch numbers'] = 1  # Assuming 100% of the patch is this type of shape

                    # Run the Monte Carlo simulation with the generated structure and store results in savedict
                    savedict = parallellel_powerful_forward(moves, std, m, shapes, initial_angle, anneal_steps, r_sims, cluster_max, anneal_cutoff, equilibrium_cutoff, savedict)

                    # Store the results of this simulation in the DataFrame at the current index
                    mainframe.loc[indexlist[count]] = savedict
                    count += 1  # Increment the simulation counter

                    break  # Exit the loop if no error occurred
                except Exception:
                    # If an error occurs, retry the simulation
                    continue

    return mainframe, count  # Return the updated DataFrame and simulation counter

def megasim_varyonlypatch(moves, std, mval, initial_angle, anneal_steps, m_sims, r_sims, cluster_max, chain_len, shape_options, orientation_options,
                          len_options, strength_percents, patch_numbers, indexlist, mainframe, savedict, count, anneal_cutoff, equilibrium_cutoff):
    '''This function runs Monte Carlo simulations for varying patch magnetization percentages and strength percentages for the generated structures. 
    It creates unique structures for each patch strength in the specified range and runs the Monte Carlo simulation for each structure multiple times.
    This function is typically used in conjunction with the `run_all_megasims` function.
    
    Inputs:
        moves: int
            The maximum number of allowable moves in each Monte Carlo simulation step.
        std: float
            Standard deviation for random angle generation in the Monte Carlo process.
        mval: float
            Strength of the magnetic patch, typically around 1.04E-12.
        initial_angle: float
            The initial angle used to determine the characteristic energy of the structure.
        anneal_steps: int
            The number of steps for the annealing process in the Monte Carlo simulation.
        m_sims: int
            The number of simulations to run, each with a different patch strength and composition.
        r_sims: int
            The number of times each structure undergoes a Monte Carlo simulation.
        cluster_max: int
            The maximum number of clusters to analyze during the simulation.
        chain_len: int
            The length of the chain (number of shapes) to be simulated.
        shape_options: list
            A list of allowable shapes that can be used in the sequence.
        orientation_options: list
            A list of possible orientations for the shapes in the sequence.
        len_options: list
            A list of allowable side lengths for the shapes in the sequence.
        strength_percents: list
            A list of percentages for the magnetic strength to be applied to the patches.
        patch_numbers: list
            A list of percentages representing the composition of patches in the sequence.
        indexlist: list
            A list of indices to store results for each simulation.
        mainframe: pandas.DataFrame
            A DataFrame used to store the results of all simulations.
        savedict: dict
            A dictionary that holds the calculated properties and measurables for each run.
        count: int
            A counter that tracks the number of simulations completed.
        anneal_cutoff: float
            A cutoff value for the annealing process to determine when to stop.
        equilibrium_cutoff: float
            A threshold to determine when equilibrium is reached in the system.

    Outputs:
        mainframe: pandas.DataFrame
            The DataFrame containing all results from the simulations.
        count: int
            The updated simulation counter, incremented after each successful simulation.
    '''
    
    # Initialize list to hold generated shapes
    shapelist = []

    # Generate unique structures for each simulation
    for i in range(m_sims):
        go_on = 0
        while go_on == 0:  # Continue until a valid structure is found
            # Generate a random structure of shapes based on input options
            shapes = vary_structure(shape_options, orientation_options, len_options, chain_len)
            
            # Check if the generated structure already exists (avoid duplicates)
            if str(shapes) in shapelist:
                go_on = 0  # Retry if the sequence is a duplicate
            elif len(moving_hinges(sequence_from_shapes(shapes))) == 0:  # Ensure the structure has at least one movable hinge
                go_on = 0  # Retry if the structure is rigid
            else:
                shapelist.append(str(shapes))  # Store the valid, unique structure
                go_on = 1  # Proceed with this structure

    # Convert the list of shapes to a DataFrame for storage
    shapeframe = pd.DataFrame(shapelist)
    title = "shapeframe"
    path = os.path.join(os.getcwd(), "Simulation results")  # Define the file path for saving results
    filename = os.path.join(path, title + ".csv")
    shapeframe.to_csv(filename)  # Save the shapes DataFrame as a CSV file

    # Iterate over each patch composition (i.e., varying the patch number percentage)
    for i in patch_numbers:
        patch_nums = [i, chain_len - i]  # Define the composition of patches: i and the rest of the shapes
        print(f'Simulating Sequences with {i} patches of {strength_percents[0]*100} % strength\n')

        # For the number of desired samples at this patch percent composition
        for j in range(m_sims):
            print(f'Simulation #{j}')

            # Generate patch strengths based on the specified percentages
            m = vary_m(strength_percents, patch_nums, chain_len, mval)

            # Run the Monte Carlo simulation and store results
            savedict['patch strengths'] = strength_percents
            savedict['patch numbers'] = patch_nums
            shapes = shapelist[j]
            
            # Format the shape string properly for parsing
            shapes = re.sub(r"'", r'"', shapes)
            shapes = json.loads(shapes)
            
            # Print the generated sequence and its movable hinges for debugging purposes
            print('      ', sequence_from_shapes(shapes))
            print('      ', moving_hinges(sequence_from_shapes(shapes)))
            
            # Perform the Monte Carlo simulation and update the saved dictionary
            savedict = parallellel_powerful_forward(moves, std, m, shapes, initial_angle, anneal_steps, r_sims, cluster_max, anneal_cutoff, equilibrium_cutoff, savedict)

            # Store the simulation results in the DataFrame at the appropriate index
            mainframe.loc[indexlist[count]] = savedict
            count += 1  # Increment the simulation counter

    # Return the updated DataFrame and the current simulation count
    return mainframe, count

def define_input_dict(moves, std, mval, initial_angle, anneal_steps, r_sims, cluster_max, chain_len, anneal_cutoff, equilibrium_cutoff,
                      l_sims, l_number_options, l_shape_options, l_orientation_options, l_len_options,
                      s_sims, s_shape_options, s_orientation_options, s_len_options, s_number_options,
                      m_sims, m_shape_options, m_orientation_options, m_len_options, m_strength_percents, m_number_options):
    '''This function creates and returns a dictionary containing the input parameters for the run_all_megasims function. 
    It is designed for organizing and formatting the input data in a structured way, making the data more readable and easier to pass to other functions.
    
    Inputs:
        moves: int
            Maximum number of allowable moves for the Monte Carlo simulation.
        std: float
            Standard deviation used to draw a random angle from a Gaussian distribution for hinge movements.
        mval: float
            The strength of the magnetic patch.
        initial_angle: float
            The initial angle used to rotate the structure for determining characteristic energy.
        anneal_steps: int
            The number of steps over which the annealing process is applied.
        r_sims: int
            The number of Monte Carlo simulations to perform for each structure.
        cluster_max: int
            The maximum number of clusters to analyze.
        chain_len: int
            The length of the chain for the simulation.
        anneal_cutoff: float
            The threshold value for stopping the annealing process.
        equilibrium_cutoff: float
            The threshold value for determining when equilibrium is reached.

        l_sims, s_sims, m_sims: int
            The number of simulations to be performed for length, shape, and magnetization respectively.

        l_number_options, s_number_options, m_number_options: list
            Lists containing options for the number of elements to be used in the length, shape, and magnetization simulations.

        l_shape_options, s_shape_options, m_shape_options: list
            Lists of allowable shapes for the corresponding simulations.

        l_orientation_options, s_orientation_options, m_orientation_options: list
            Lists of allowable orientations for the corresponding simulations.

        l_len_options, s_len_options, m_len_options: list
            Lists of allowable lengths for the corresponding simulations.

        m_strength_percents: list
            List of percentages for patch strength used in magnetization simulations.

    Outputs:
        input_dict: dict
            A dictionary containing all input parameters needed for the simulations.
    '''
    
    # Construct a dictionary with all the provided input parameters.
    input_dict = {'moves': moves, 
                  'std': std, 
                  'mval': mval, 
                  'initial_angle':initial_angle, 
                  'anneal steps': anneal_steps, 
                  'r_sims':r_sims,
                  'cluster_max':cluster_max,
                  'chain length':chain_len, 
                  'anneal cutoff':anneal_cutoff, 
                  'equilibrium cutoff':equilibrium_cutoff,
                  'l_sims':l_sims,
                  'l_number_options':l_number_options, 
                  'l_shape_options':l_shape_options,
                  'l_orientation_options':l_orientation_options,
                  'l_len_options':l_len_options,
                  's_sims':s_sims,
                  's_shape_options':s_shape_options, 
                  's_orientation_options':s_orientation_options,
                  's_len_options':s_len_options,
                  's_number_options':s_number_options,
                  'm_sims':m_sims,
                  'm_shape_options':m_shape_options, 
                  'm_orientation_options':m_orientation_options,
                  'm_len_options':m_len_options,
                  'm_strength_percents':m_strength_percents,
                  'm_number_options':m_number_options}
    
    # Return the dictionary containing all input parameters
    return input_dict

def run_all_magasims_varyonlyshape(input_dict):
    '''This function runs multiple simulations with varying parameters such as structure length, shape, and patch strength.
    The results of these simulations are appended to a pandas dataframe, which is then returned. The function runs simulations in sequence for length, shape, and patch variations.
    
    Inputs:
        input_dict: dict
            A dictionary containing all relevant input parameters. The dictionary is structured as follows:
            
            - 'moves': int
                The maximum number of allowable moves in the Monte Carlo simulation. Controls the number of iterations.
            
            - 'std': float
                The standard deviation used for drawing random angles from a Gaussian distribution to move the hinges at each step of the Monte Carlo simulation.
            
            - 'mval': float
                The strength of the magnetic patch used in the simulation (e.g., 1.04E-12).
            
            - 'initial_angle': float
                The initial angle to rotate the structure to, in order to determine the characteristic energy for the simulation.
            
            - 'anneal_steps': int
                The number of annealing steps to run in the Monte Carlo simulation, controlling the thermalization process.
            
            - 'r_sims': int
                The number of times the Monte Carlo folding simulation is performed for each sequence.
            
            - 'cluster_max': int
                The maximum number of clusters to consider for analysis during the simulation.
            
            - 'chain_len': int
                The chain length used in simulations to define the number of segments in the structure.
            
            - 'anneal_cutoff': float
                The cutoff value used during the annealing process, determining when the simulation can be terminated.
            
            - 'equilibrium_cutoff': float
                The cutoff value to determine when the system has reached equilibrium and further simulation steps are not necessary.
            
            - 'l_sims': int
                The number of length simulations. This controls how many times a random structure should be generated and placed through the full Monte Carlo procedure for length simulations.
            
            - 'l_number_options': list of ints
                A list containing different chain lengths to be examined in the simulation during the length variation step.
            
            - 'l_shape_options': list of str
                A list containing the allowable shapes to be generated in each sequence during the length simulations.
            
            - 'l_orientation_options': list of str
                A list containing the allowable orientations to be generated in each sequence during the length simulations.
            
            - 'l_len_options': list of float
                A list containing the allowable lengths of each shape in each sequence for length simulations.
            
            - 's_sims': int
                The number of shape simulations. This controls how many times a random structure should be generated and placed through the full Monte Carlo procedure for shape simulations.
            
            - 's_shape_options': list of str
                A list containing the allowable shapes to be generated in each sequence during the shape simulations.
            
            - 's_orientation_options': list of str
                A list containing the allowable orientations to be generated in each sequence during the shape simulations.
            
            - 's_len_options': list of float
                A list containing the allowable lengths of each shape in each sequence for shape simulations.
            
            - 's_number_options': list of ints
                A list containing different chain lengths to be examined in the simulation during the shape simulations.
            
            - 'm_sims': int
                The number of magnetization simulations. This controls how many times the Monte Carlo simulation should be performed for magnetization variation.
            
            - 'm_shape_options': list of str
                A list containing the allowable shapes to be generated in each sequence during the magnetization simulations.
            
            - 'm_orientation_options': list of str
                A list containing the allowable orientations to be generated in each sequence during the magnetization simulations.
            
            - 'm_len_options': list of float
                A list containing the allowable lengths of each shape in each sequence during the magnetization simulations.
            
            - 'm_strength_percents': list of float
                A list containing percentages of the second magnetic patch strength to be tested for magnetization simulations.
            
            - 'm_number_options': list of ints
                A list containing different chain lengths to be examined in the simulation during the magnetization simulations.
    
    Outputs:
        mainframe: pandas.DataFrame
            The final dataframe containing all relevant data from the simulations, including results from varying structure length, shape, and patch strength.
    '''
    
    # Retrieve individual simulation parameters from the input dictionary
    moves = input_dict['moves']
    std = input_dict['std']
    mval = input_dict['mval']
    initial_angle = input_dict['initial_angle']
    anneal_steps = input_dict['anneal steps']
    r_sims = input_dict['r_sims']
    cluster_max = input_dict['cluster_max']
    chain_len = input_dict['chain length']
    anneal_cutoff = input_dict['anneal cutoff']
    equilibrium_cutoff = input_dict['equilibrium cutoff']

    # Retrieve parameters for length, shape, and magnetization simulations
    l_sims = input_dict['l_sims']
    l_number_options = input_dict['l_number_options']
    l_shape_options = input_dict['l_shape_options']
    l_orientation_options = input_dict['l_orientation_options']
    l_len_options = input_dict['l_len_options']

    s_sims = input_dict['s_sims']
    s_shape_options = input_dict['s_shape_options']
    s_orientation_options = input_dict['s_orientation_options']
    s_len_options = input_dict['s_len_options']
    s_number_options = input_dict['s_number_options']

    m_sims = input_dict['m_sims']
    m_shape_options = input_dict['m_shape_options']
    m_orientation_options = input_dict['m_orientation_options']
    m_len_options = input_dict['m_len_options']
    m_strength_percents = input_dict['m_strength_percents']
    m_number_options = input_dict['m_number_options']
    
    # Define the mainframe (a pandas dataframe to store results) and index list for simulations
    mainframe, indexlist = define_mainframe(l_sims, l_number_options, s_sims, s_number_options, m_sims, m_number_options)
    
    # Create a dictionary to store results from each individual simulation
    savedict = define_savedict()
    
    print('Total number of simulations:', len(indexlist))  # Print the total number of simulations
    print('')
    
    count = 0  # Initialize simulation counter
    t0 = time.time()  # Start the timer to track total simulation time
    
    # 1. Run simulations varying length to see how changing length of the structure affects folding behavior
    print('Varying Length')
    mainframe, count = megasim_varylength(moves, std, mval, initial_angle, anneal_steps, l_sims, r_sims, cluster_max, 
                                          l_number_options, l_shape_options, l_orientation_options, l_len_options, 
                                          indexlist, mainframe, savedict, count, anneal_cutoff, equilibrium_cutoff)
    tl = time.time()
    print('Length Simulations in ', tl - t0, 's')  # Print time taken for length simulations

    # 2. Run simulations varying shapes to see how changing shape of the structure affects folding behavior
    print('Varying Shapes')
    mainframe, count = megasim_varyonlyshapes(moves, std, mval, initial_angle, anneal_steps, s_sims, r_sims, cluster_max, 
                                              chain_len, s_shape_options, s_orientation_options, s_len_options, 
                                              s_number_options, indexlist, mainframe, savedict, count, 
                                              anneal_cutoff, equilibrium_cutoff)
    ts = time.time()
    print('Shape Simulations in ', ts - tl, 's')  # Print time taken for shape simulations

    # 3. Run simulations varying patch strength to see how changing magnetization of the structure affects folding behavior
    print('Varying Patches')
    mainframe, count = megasim_varyonlypatch(moves, std, mval, initial_angle, anneal_steps, m_sims, r_sims, cluster_max, 
                                             chain_len, m_shape_options, m_orientation_options, m_len_options, 
                                             m_strength_percents, m_number_options, indexlist, mainframe, 
                                             savedict, count, anneal_cutoff, equilibrium_cutoff)
    tp = time.time()
    print('Patch Simulations in ', tp - ts, 's')  # Print time taken for patch simulations
    
    # Print total simulation time
    tf = time.time() - t0
    print('Total Simulation time: ', tf, 's')
    
    # Return the dataframe containing all results
    return mainframe

def serialize_complex_data(data):
    '''In order to store the data in a way that it can be easily from excel, it needs to be serialized. This code was generated by chatGPT
    Inputs:
        data: pandas dataframe that needs to be serialized
    Outputs:
        data: serialized data
    '''

    if isinstance(data, (dict, list)):
        return json.dumps(data)
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            return json.dumps(data.tolist())  # Convert 1D numpy array to list
        else:
            return json.dumps(data.tolist(), separators=(',', ':'))  # Convert higher-dimensional numpy array to list while preserving shape
    return data

def deserialize_complex_data(data):
    '''When loading the data from an excel csv file, the data needs to be deserialized. This code was generated by chatGPT
    Inputs:
        data: downloaded data from an excel csv file
    Outputs:
        data: pandas dataframe with all datatypes preserved
    '''
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return data

def save_mainframe(mainframe,title):
    '''This function takes the data stored from megasims in a pandas dataframe, serializes it, and stores it as a csv file
    Inputs:
        mainframe: a pandas dataframe holding all information about folding from the megasims
        title: desired name of the file
    Outputs:
        filename: the entire filename (including the path)
    '''

    mainframe_serialized = mainframe.applymap(serialize_complex_data) #serialize the data
    path = os.path.join(os.getcwd(),'Simulation Results') #Identify the filepath
    filename = os.path.join(path, title + ".csv") # Save the DataFrame to Excel
    os.makedirs(path, exist_ok=True) # Ensure that the directory exists
    mainframe_serialized.to_csv(filename)

    return filename

def megasim_varylength_withshapes(moves, std, mval, initial_angle, anneal_steps, l_sims, r_sims, cluster_max, number_options, shape_options, shape_percent, orientation_options,
                                  len_options, indexlist, mainframe, savedict, count, anneal_cutoff, equilibrium_cutoff):
    '''
    This function runs Monte Carlo simulations to simulate chains of varying lengths with different shapes.
    For each chain length, the function generates a random structure, performs the simulation, and records the results.
    
    Inputs:
        moves: int
            The maximum number of allowable moves for each Monte Carlo step.
        
        std: float
            The standard deviation used for drawing random angles in each simulation step.
        
        mval: float
            The strength of the magnetic patch in the simulation.
        
        initial_angle: float
            The initial angle for the structure's rotation, determining the starting energy configuration.
        
        anneal_steps: int
            The number of annealing steps in the simulation to thermalize the system.
        
        l_sims: int
            The number of Monte Carlo simulations per length.
        
        r_sims: int
            The number of random structure generations for each simulation.
        
        cluster_max: int
            The maximum number of clusters to analyze during the simulation.
        
        number_options: list of ints
            A list of chain lengths to be tested in the simulation.
        
        shape_options: list of str
            A list of the shapes that can be used in the simulation.
        
        shape_percent: float
            The percentage of triangles used for each chain length (as a fraction).
        
        orientation_options: list of str
            A list of possible orientations for the shapes in the simulation.
        
        len_options: list of floats
            A list of possible lengths for the shapes to be used in the simulation.
        
        indexlist: list of ints
            The indices of the dataframe where the simulation results will be stored.
        
        mainframe: pandas.DataFrame
            A dataframe to store the results of the simulation.
        
        savedict: dict
            A dictionary that holds intermediate results and data to be added to the dataframe.
        
        count: int
            The current count of simulations performed.
        
        anneal_cutoff: float
            The cutoff value for the annealing process to stop early if necessary.
        
        equilibrium_cutoff: float
            The cutoff value for equilibrium, determining when the system has reached a stable state.
    
    Outputs:
        mainframe: pandas.DataFrame
            The dataframe containing all simulation results after processing.
        
        count: int
            The updated count of simulations performed.
    '''
    
    # Loop over each chain length to perform simulations for different lengths
    for i in number_options:  # For each chain length in the study
        shape_numbers = int(np.round(shape_percent * i, 0))  # Determine the number of triangles for this length based on the given shape percentage
        shape_nums = [shape_numbers, i - shape_numbers]  # Split the remaining number of elements
        shapelist = []  # Initialize the list to store all used sequences for uniqueness checks
        print('Simulating Chains of Length ', i, ' with ', shape_numbers, ' triangles')
        print('')
        
        # Run the simulation for the desired number of times (l_sims)
        for j in range(l_sims):  # For the desired sample size (number of times a random structure is simulated and put through MC)
            print('Simulation #', j)
            
            # Try generating the structure and running the simulation. If any error occurs, the simulation will retry.
            while True:
                try:
                    go_on = 0  # Flag to track if a valid structure has been generated
                    shapes = {}  # Initialize an empty dictionary to store the structure's shapes

                    # Keep generating shapes until a valid configuration is found
                    while go_on == 0:  # While the stopping criteria has not been met
                        shapes = vary_shapes(shape_options, shape_nums, orientation_options, len_options, i)  # Generate a random structure
                        
                        # Check if the structure already exists in shapelist (avoiding duplicates)
                        if str(shapes) in shapelist:
                            go_on = 0  # Retry if duplicate is found
                        
                        # Check if the generated structure has no movable hinges, which would make it invalid
                        elif len(moving_hinges(sequence_from_shapes(shapes))) == 0:
                            go_on = 0  # Retry if there are no movable hinges
                        else:
                            go_on = 1  # Proceed with the current structure if valid
                    
                    # Initialize the magnetization list with the given magnetic patch value
                    m = [mval]  # Start with the given magnetic patch strength
                    m = m * len(shapes)  # Extend the list to match the number of shapes

                    # Save the values in the savedict dictionary for later processing
                    savedict['patch strengths'] = mval / 1.01e-6
                    savedict['patch percentages'] = 1
                    savedict = parallellel_powerful_forward(moves, std, m, shapes, initial_angle, anneal_steps, r_sims, cluster_max, anneal_cutoff, equilibrium_cutoff, savedict)
                    savedict['% triangles'] = shape_percent  # Store the percentage of triangles used for this simulation

                    break  # End the while loop once the simulation is successful

                except Exception:  # If an error occurs during simulation, retry
                    print('Error')  # Print the error message
                    continue  # Retry the simulation
            
            # Append the successfully generated shape configuration to the shapelist for uniqueness tracking
            shapelist.append(str(shapes))
            
            # Add the simulation results from savedict to the main dataframe at the specified index
            mainframe.loc[indexlist[count]] = savedict
            count += 1  # Increment the simulation counter
    
    return mainframe, count  # Return the updated dataframe and simulation count

def define_input_dict_ls(moves, std, mval, initial_angle, anneal_steps, r_sims, cluster_max, chain_len, anneal_cutoff, equilibrium_cutoff,
                      l_sims, l_number_options, l_shape_options, l_shape_percent, l_orientation_options, l_len_options,
                      s_sims, s_shape_options, s_orientation_options, s_len_options, s_number_options,
                      m_sims, m_shape_options, m_orientation_options, m_len_options, m_strength_percents, m_number_options):
    '''
    This function defines an input dictionary that is used as input for the `run_all_megasims` function. 
    While not strictly necessary, it makes organizing and passing the input data more convenient.

    Inputs:
        moves: int
            The number of moves for the Monte Carlo simulation.
        
        std: float
            The standard deviation used for random angle generation in each simulation step.
        
        mval: float
            The strength of the magnetic patch used in the simulation.
        
        initial_angle: float
            The starting angle used in the simulation to define the initial energy configuration.
        
        anneal_steps: int
            The number of annealing steps used in the Monte Carlo simulation to allow the system to thermalize.
        
        r_sims: int
            The number of simulations for generating random structures.
        
        cluster_max: int
            The maximum number of clusters considered in the analysis.
        
        chain_len: int
            The length of the chains to be simulated.
        
        anneal_cutoff: float
            The cutoff for the annealing process to terminate early if the system has stabilized.
        
        equilibrium_cutoff: float
            The cutoff for determining when the system has reached equilibrium.
        
        l_sims: int
            The number of simulations for a given chain length.
        
        l_number_options: list of ints
            A list of different chain lengths to be simulated for the "L" group.
        
        l_shape_options: list of str
            A list of shapes that will be used for the "L" group simulations.
        
        l_shape_percent: float
            The percentage of shapes that should be triangles for the "L" group simulations.
        
        l_orientation_options: list of str
            A list of possible orientations for the shapes in the "L" group.
        
        l_len_options: list of floats
            A list of possible lengths for the shapes in the "L" group simulations.
        
        s_sims: int
            The number of simulations for the "S" group.
        
        s_shape_options: list of str
            A list of shapes for the "S" group.
        
        s_orientation_options: list of str
            A list of possible orientations for the "S" group.
        
        s_len_options: list of floats
            A list of possible lengths for the shapes in the "S" group.
        
        s_number_options: list of ints
            A list of possible chain lengths for the "S" group.
        
        m_sims: int
            The number of simulations for the "M" group.
        
        m_shape_options: list of str
            A list of shape options for the "M" group.
        
        m_orientation_options: list of str
            A list of possible orientations for the "M" group.
        
        m_len_options: list of floats
            A list of possible lengths for the shapes in the "M" group.
        
        m_strength_percents: list of floats
            A list of percentage values for the magnetic strength in the "M" group.
        
        m_number_options: list of ints
            A list of possible chain lengths for the "M" group.

    Outputs:
        input_dict: dict
            A dictionary containing all the input parameters, ready to be used for the simulation function.
    '''
    
    # Define the input dictionary with all the input parameters
    input_dict = {
        'moves': moves,
        'std': std,
        'mval': mval,
        'initial_angle': initial_angle,
        'anneal steps': anneal_steps,
        'r_sims': r_sims,
        'cluster_max': cluster_max,
        'chain length': chain_len,
        'anneal cutoff': anneal_cutoff,
        'equilibrium cutoff': equilibrium_cutoff,
        'l_sims': l_sims,
        'l_number_options': l_number_options,
        'l_shape_options': l_shape_options,
        'l_shape_percent': l_shape_percent,
        'l_orientation_options': l_orientation_options,
        'l_len_options': l_len_options,
        's_sims': s_sims,
        's_shape_options': s_shape_options,
        's_orientation_options': s_orientation_options,
        's_len_options': s_len_options,
        's_number_options': s_number_options,
        'm_sims': m_sims,
        'm_shape_options': m_shape_options,
        'm_orientation_options': m_orientation_options,
        'm_len_options': m_len_options,
        'm_strength_percents': m_strength_percents,
        'm_number_options': m_number_options
    }
    
    # Return the input dictionary
    return input_dict

def run_all_magasims_ls(input_dict):
    '''
    This function independently runs each of the megasim functions and appends the results to a pandas DataFrame,
    which is then stored as a CSV file. This is the master function that currently runs all simulation tasks.
    
    Inputs:
        input_dict: dict
            A dictionary containing all the input parameters required for the simulation.
            The dictionary is structured as:
                {'moves': moves, 'std': std, 'mval': mval, 'initial_angle':initial_angle, 'r_sims': r_sims, ...}
            
            Detailed explanation of each key and its value in input_dict:
            
            # General Simulation Parameters
            'moves': int
                Number of maximum allowable moves in the Monte Carlo (MC) simulation.
            'std': float
                Standard deviation for drawing a random angle from a Gaussian distribution to move a hinge at each step of the MC simulation.
            'mval': float
                The strength of the magnetic patch (typically a small value like 1.04E-12).
            'initial_angle': float
                The initial angle used to rotate the structure to determine its characteristic energy.
            'anneal_steps': int
                The number of steps for annealing during the simulation (used to optimize the structure).
            'r_sims': int
                The number of times the Monte Carlo fold is performed for each sequence.
            'cluster_max': int
                The maximum number of clusters to analyze during the simulation (used for clustering results).
            'chain_len': int
                The chain length parameter, used for various simulations, typically representing the number of elements in the structure.
            'anneal_cutoff': float
                A cutoff value for the annealing process, typically used to decide when the simulation should stop based on energy minimization.
            'equilibrium_cutoff': float
                A cutoff value for determining when the system has reached equilibrium during the simulation.
            
            # Length Simulation Parameters
            'l_sims': int
                Number of length simulations to run (how many times the structure should be randomly generated and placed through the full MC simulation for length).
            'l_number_options': list of int
                List containing the different chain lengths to be examined during the length simulations.
            'l_shape_options': list of str
                List containing the allowable shapes to be generated in each sequence for length simulations.
            'l_shape_percent': list of float
                List of shape composition percentages, used to control the proportion of different shapes in each structure for length simulations.
            'l_orientation_options': list of str
                List containing the allowable orientations to be applied to each sequence for length simulations.
            'l_len_options': list of float
                List containing the allowable lengths of each shape to be used in the simulations.

            # Shape Simulation Parameters
            's_sims': int
                Number of shape simulations to run (how many times the structure should be randomly generated and analyzed based on shape properties).
            's_shape_options': list of str
                List containing the allowable shapes to be generated in each sequence for shape simulations.
            's_orientation_options': list of str
                List containing the allowable orientations to be applied to each shape sequence for shape simulations.
            's_len_options': list of float
                List containing the allowable lengths of each shape to be used for shape simulations.
            's_number_options': list of int
                List containing different chain lengths to be examined during the shape simulations.

            # Magnetization Simulation Parameters
            'm_sims': int
                Number of magnetization anisotropy simulations to run (studying how magnetization changes with anisotropy).
            'm_shape_options': list of str
                List containing the allowable shapes to be generated in each sequence for magnetization simulations.
            'm_orientation_options': list of str
                List containing the allowable orientations to be applied to each sequence for magnetization simulations.
            'm_len_options': list of float
                List containing the allowable lengths of each shape used for magnetization simulations.
            'm_strength_percents': list of float
                List containing percentages of patch strengths to study in the magnetization simulations (determining how the magnetization strength varies).
            'm_number_options': list of int
                List containing the different chain lengths to be examined in the magnetization simulations.

    Outputs:
        mainframe: pandas DataFrame
            A DataFrame containing the results from all simulations.
    '''

    #Retrieve all values from the input dictionary
    moves = input_dict['moves']
    std = input_dict['std']
    mval = input_dict['mval']
    initial_angle = input_dict['initial_angle']
    anneal_steps = input_dict['anneal steps']
    r_sims = input_dict['r_sims']
    cluster_max = input_dict['cluster_max']
    chain_len = input_dict['chain length']
    anneal_cutoff = input_dict['anneal cutoff']
    equilibrium_cutoff = input_dict['equilibrium cutoff']

    l_sims = input_dict['l_sims']
    l_number_options = input_dict['l_number_options']
    l_shape_options = input_dict['l_shape_options']
    l_shape_percent = input_dict['l_shape_percent']
    l_orientation_options = input_dict['l_orientation_options']
    l_len_options = input_dict['l_len_options']

    s_sims = input_dict['s_sims']
    s_shape_options = input_dict['s_shape_options']
    s_orientation_options = input_dict['s_orientation_options']
    s_len_options = input_dict['s_len_options']
    s_number_options = input_dict['s_number_options']

    m_sims = input_dict['m_sims']
    m_shape_options = input_dict['m_shape_options']
    m_orientation_options = input_dict['m_orientation_options']
    m_len_options = input_dict['m_len_options']
    m_strength_percents = input_dict['m_strength_percents']
    m_number_options = input_dict['m_number_options']


    mainframe,indexlist = define_mainframe(l_sims, l_number_options, s_sims, s_number_options, m_sims,  m_number_options)
    savedict = define_savedict() #create dictionary to store results of each individual simulation

    print('Total number of simulations:',len(indexlist)) #Show total number of simulations to be performed
    print('')
    
    count = 0 #initialize simulation counter
    t0 = time.time()

    print('Varying Length')
    #Run simulations varying length to see how changing length of the structure affects folding behavior
    mainframe, count = megasim_varylength_withshapes(moves, std, mval, initial_angle, anneal_steps, l_sims, r_sims, cluster_max, l_number_options, l_shape_options, l_shape_percent,
                                                    l_orientation_options, l_len_options, indexlist, mainframe, savedict, count, anneal_cutoff, equilibrium_cutoff)
    tl = time.time()
    print('Length Simulations in ', tl-t0, 's')

    print('Varying Shapes')
    #Run simulations varying shapes to see how changing length of the structure affects folding behavior
    mainframe, count = megasim_varyonlyshapes(moves, std, mval, initial_angle, anneal_steps, s_sims, r_sims, cluster_max, chain_len, s_shape_options,
                                          s_orientation_options, s_len_options, s_number_options, indexlist, mainframe, savedict, count, anneal_cutoff, equilibrium_cutoff)
    ts = time.time()
    print('Shape Simulations in ', ts-tl, 's')

    print('varying Patches')
    #Run simulations varying patch strength to see how changing magentization of the structure affects folding behavior
    mainframe, count = megasim_varyonlypatch(moves, std, mval, initial_angle, anneal_steps, m_sims, r_sims, cluster_max, chain_len, m_shape_options,
                                         m_orientation_options, m_len_options, m_strength_percents, m_number_options, indexlist, mainframe, savedict, count, anneal_cutoff, equilibrium_cutoff)
    tp = time.time()
    print('Patch Simulations in ', tp-ts, 's')
    
    tf = time.time() - t0
    print('Total Simulation time: ', tf, 's')

    return mainframe

def end_to_end_number_norm(ordered_centers, patch_arr_init, shape_arr_init, hinge_vec_init, linelist, hinges):
    '''
    This function determines the end-to-end distance from the first point of the first patch 
    to the last point of the last patch for each equilibrium structure.

    Inputs:
        ordered_centers: numpy.ndarray
            An mxn array containing the final hinge positions of the cluster centers, 
            where m is the number of equilibrium states and n is the number of hinges. 
            This array is typically derived from `cluster_stats`.

        patch_arr_init: numpy.ndarray
            The initial array representing the patch coordinates of the sequence before 
            any Monte Carlo (MC) simulation.

        shape_arr_init: numpy.ndarray
            The initial array representing the shape of the sequence before any MC simulation.

        hinge_vec_init: numpy.ndarray
            The initial vector containing hinge positions before any MC simulation.

        linelist: list
            A list containing the number of lines in each structure, used for structural organization.

        hinges: list
            A list of hinge indices representing hinge locations in the sequence.

    Outputs:
        e2e_vec: numpy.ndarray
            A 1D array containing the end-to-end distances for each equilibrium structure in `ordered_centers`.
    '''
    # Initialize a vector to store the end-to-end distances for each equilibrium structure.
    e2e_vec = np.zeros(np.shape(ordered_centers)[0])

    # Loop over each equilibrium structure in `ordered_centers`.
    for i in range(np.shape(ordered_centers)[0]):
        # Create deep copies of the initial arrays to preserve their original states.
        hinge_vec = copy.deepcopy(hinge_vec_init)
        patch_arr = copy.deepcopy(patch_arr_init)
        shape_arr = copy.deepcopy(shape_arr_init)

        # Rotate the hinges one by one to align with their final positions.
        for j in range(len(hinges)):
            # Calculate the angle by which to rotate the current hinge.
            angle = ordered_centers[i, j] - hinge_vec[hinges[j]]

            # Select the hinge to rotate.
            hingechoice = hinges[j]

            # Perform the hinge rotation and update the arrays accordingly.
            patch_arr, shape_arr, hinge_vec = rotate_once(
                patch_arr, shape_arr, linelist, hinge_vec, hingechoice, hinges, angle
            )

        # Calculate the end-to-end distance as the Euclidean distance from the origin 
        # to the last point in the `patch_arr`.
        e2e_vec[i] = np.linalg.norm(patch_arr[:, -1])

    return e2e_vec

def delta_end_to_end_number_norm(ordered_centers, patch_arr_init, shape_arr_init, hinge_vec_init, linelist, hinges):
    '''
    This function calculates the change in end-to-end distance for each equilibrium structure 
    by comparing the initial state (hinges at 180°) to the final state (given by `ordered_centers`).

    Inputs:
        ordered_centers: numpy.ndarray
            An mxn array containing the final hinge positions of the cluster centers, 
            where m is the number of equilibrium states and n is the number of hinges.
            Derived from `cluster_stats`.

        patch_arr_init: numpy.ndarray
            The initial array representing the patch coordinates of the sequence before 
            any Monte Carlo (MC) simulation.

        shape_arr_init: numpy.ndarray
            The initial array representing the shape of the sequence before any MC simulation.

        hinge_vec_init: numpy.ndarray
            The initial vector containing hinge positions before any MC simulation.

        linelist: list
            A list containing the number of lines in each structure, used for structural organization.

        hinges: list
            A list of hinge indices representing hinge locations in the sequence.

    Outputs:
        delta_e2e: numpy.ndarray
            A 1D array containing the change in end-to-end distances (initial minus final) 
            for each equilibrium structure in `ordered_centers`.
    '''
    
    # Initialize a matrix where all hinge angles are set to 180° for the initial state.
    ordered_centers0 = np.ones(np.shape(ordered_centers)) * 180

    # Calculate the end-to-end distances for the initial state (all hinges at 180°).
    e2e0 = end_to_end_number_norm(
        ordered_centers0, patch_arr_init, shape_arr_init, hinge_vec_init, linelist, hinges
    )

    # Calculate the end-to-end distances for the final state (hinges at positions in `ordered_centers`).
    e2ef = end_to_end_number_norm(
        ordered_centers, patch_arr_init, shape_arr_init, hinge_vec_init, linelist, hinges
    )

    # Compute the change in end-to-end distance as the difference between the initial and final states.
    delta_e2e = e2e0 - e2ef

    return delta_e2e

def find_Rg_number_norm(ordered_centers,patch_arr_init,shape_arr_init,hinge_vec_init,linelist,hinges):
    '''calculate the actual 2D geometric radius of gyration of the structure. This is independent from mass of the structure
    #Inputs:
        ordered_centers: The mxn numpy array containing the final hinge positions of the clustercenters where m is the number of equilibrium states and n is the number of hinges
                         This comes directly from cluster_stats
        patch_arr_init: the initial patch_arr for the sequence before any MC simulation
        shape_arr_init: the initial shape_arr for the sequence before any MC simulation
        hinge_vec_init: the initial vector containing hinges before any MC simulation
        linelest: the list containing the number of lines in each structure
        hinges: list of hinge locations (often called as hingenum)
    #Outputs:
        #Rg_vec: a vector of radii of gyration for each equilibrium structure in ordered_centers
    '''

    Rg_vec = np.zeros(np.shape(ordered_centers)[0]) #Initialize the vector to hold radii of gyration based on the number of equilibrium states in the structure
    total_M = np.sum(linelist) #The 'total mass' is the total number of points. Since this is a 2D geometric calculation, the radius of gyration is weighted by number of points rather than weight
                               #located at that point
    COMx_init = np.sum(shape_arr_init[0,:])/np.shape(shape_arr_init)[1] #The x-coordinate of the 2D center of geometric mass is the sum of all of the x-coordinates divided by the number of x-coordinates
    COMy_init = np.sum(shape_arr_init[1,:])/np.shape(shape_arr_init)[1] #The y-coordinate of the 2D center of geometric mass is the sum of all of the y-coordinates divided by the number of y-coordinates
    total_shapes = len(linelist)

    for i in range(np.shape(ordered_centers)[0]): #For each equilibrium structure
        #Make copies of relevant vectors/arrays
        hinge_vec = hinge_vec_init.copy()
        patch_arr = patch_arr_init.copy()
        shape_arr = shape_arr_init.copy()
        for j in range(len(hinges)): #For every hinge in the final structure
            angle = ordered_centers[i,j] - hinge_vec[hinges[j]] #The angle by which to rotate is the difference between the final angle and the initial angle
            hingechoice = hinges[j]#Select the hinge to rotate
            patch_arr,shape_arr,hinge_vec = rotate_once(patch_arr, shape_arr, linelist, hinge_vec, hingechoice, hinges, angle) #Rotate the hinge to the final position

        COMx = np.sum(shape_arr[0,:])/np.shape(shape_arr)[1] #The x-coordinate of the 2D center of geometric mass is the sum of all of the x-coordinates divided by the number of x-coordinates
        COMy = np.sum(shape_arr[1,:])/np.shape(shape_arr)[1] #The y-coordinate of the 2D center of geometric mass is the sum of all of the y-coordinates divided by the number of y-coordinates

        r_vec = np.linalg.norm(shape_arr - np.array([[COMx], [COMy]]), axis=0) #subtract all points in the shape_arrs from the centers of mass and take the norm of this
        Rg_vec[i] = np.sqrt(np.sum(np.square(r_vec))/total_M)/total_shapes #Divide the moment of intertia by total number of points and take the square to get a radius of gyration

    return Rg_vec

def find_Rg(ordered_centers,patch_arr_init,shape_arr_init,hinge_vec_init,linelist,hinges):
    '''calculate the actual 2D geometric radius of gyration of the structure. This is independent from mass of the structure
    #Inputs:
        ordered_centers: The mxn numpy array containing the final hinge positions of the clustercenters where m is the number of equilibrium states and n is the number of hinges
                         This comes directly from cluster_stats
        patch_arr_init: the initial patch_arr for the sequence before any MC simulation
        shape_arr_init: the initial shape_arr for the sequence before any MC simulation
        hinge_vec_init: the initial vector containing hinges before any MC simulation
        linelest: the list containing the number of lines in each structure
        hinges: list of hinge locations (often called as hingenum)
    #Outputs:
        #Rg_vec: a vector of radii of gyration for each equilibrium structure in ordered_centers
    '''

    Rg_vec = np.zeros(np.shape(ordered_centers)[0]) #Initialize the vector to hold radii of gyration based on the number of equilibrium states in the structure
    total_M = np.sum(linelist) #The 'total mass' is the total number of points. Since this is a 2D geometric calculation, the radius of gyration is weighted by number of points rather than weight
                               #located at that point
    COMx_init = np.sum(shape_arr_init[0,:])/np.shape(shape_arr_init)[1] #The x-coordinate of the 2D center of geometric mass is the sum of all of the x-coordinates divided by the number of x-coordinates
    COMy_init = np.sum(shape_arr_init[1,:])/np.shape(shape_arr_init)[1] #The y-coordinate of the 2D center of geometric mass is the sum of all of the y-coordinates divided by the number of y-coordinates
    r_vec_init = np.linalg.norm(shape_arr_init - np.array([[COMx_init], [COMy_init]]), axis=0) #subtract all points in the shape_arrs from the centers of mass and take the norm of this
    Rg_initial = np.sqrt(np.sum(np.square(r_vec_init))/total_M)

    for i in range(np.shape(ordered_centers)[0]): #For each equilibrium structure
        #Make copies of relevant vectors/arrays
        hinge_vec = hinge_vec_init.copy()
        patch_arr = patch_arr_init.copy()
        shape_arr = shape_arr_init.copy()
        for j in range(len(hinges)): #For every hinge in the final structure
            angle = ordered_centers[i,j] - hinge_vec[hinges[j]] #The angle by which to rotate is the difference between the final angle and the initial angle
            hingechoice = hinges[j]#Select the hinge to rotate
            patch_arr,shape_arr,hinge_vec = rotate_once(patch_arr, shape_arr, linelist, hinge_vec, hingechoice, hinges, angle) #Rotate the hinge to the final position

        COMx = np.sum(shape_arr[0,:])/np.shape(shape_arr)[1] #The x-coordinate of the 2D center of geometric mass is the sum of all of the x-coordinates divided by the number of x-coordinates
        COMy = np.sum(shape_arr[1,:])/np.shape(shape_arr)[1] #The y-coordinate of the 2D center of geometric mass is the sum of all of the y-coordinates divided by the number of y-coordinates

        r_vec = np.linalg.norm(shape_arr - np.array([[COMx], [COMy]]), axis=0) #subtract all points in the shape_arrs from the centers of mass and take the norm of this
        Rg_vec[i] = np.sqrt(np.sum(np.square(r_vec))/total_M)/Rg_initial #Divide the moment of intertia by total number of points and take the square to get a radius of gyration

    return Rg_vec

def delta_Rg(ordered_centers,patch_arr_init,shape_arr_init,hinge_vec_init,linelist,hinges):
    '''Determine the change in the radius of gyration from the initial state of the structure to the each folded equilibrium state
    Inputs:
        ordered_centers: The mxn numpy array containing the final hinge positions of the clustercenters where m is the number of equilibrium states and n is the number of hinges
                         This comes directly from cluster_stats
        patch_arr_init: the initial patch_arr for the sequence before any MC simulation
        shape_arr_init: the initial shape_arr for the sequence before any MC simulation
        hinge_vec_init: the initial vector containing hinges before any MC simulation
        linelest: the list containing the number of lines in each structure
        hinges: list of hinge locations (often called as hingenum)
    Outputs:
        delta_Rg: a vector of change in radii of gyration for each equilibrium structure in ordered_centers
    '''

    Rgf = find_Rg(ordered_centers,patch_arr_init,shape_arr_init,hinge_vec_init,linelist,hinges) #The final radius of gyration vector hold all values of the radius of gyration in the final state
    delta_Rg_vec = 1-Rgf #Calculate the change

    return delta_Rg_vec

def find_tortuosity(ordered_centers, patch_arr_init, shape_arr_init, hinge_vec_init, linelist, hinges, shapes):
    '''
    Determines the linear tortuosity of each equilibrium structure by fitting a spline curve to 
    the patch points of a sequence and using curvature-based formulas.

    Inputs:
        ordered_centers: numpy.ndarray
            An mxn array containing the final hinge positions of the cluster centers, 
            where m is the number of equilibrium states and n is the number of hinges.
            Derived from `cluster_stats`.

        patch_arr_init: numpy.ndarray
            Initial patch positions of the sequence before any MC simulation.

        shape_arr_init: numpy.ndarray
            Initial shape array of the sequence before any MC simulation.

        hinge_vec_init: numpy.ndarray
            Initial vector containing hinge positions before any MC simulation.

        linelist: list
            A list describing the number of lines in each structure.

        hinges: list
            A list of hinge indices representing hinge locations.

        shapes: dict
            A dictionary describing the shape, orientation, and side length of each structure.

        p: int
            Number of spline points to generate per patch. This value is chosen to ensure
            consistent tortuosity calculations.

    Outputs:
        T_vec: numpy.ndarray
            A 1D array containing the tortuosity of each equilibrium structure.
    '''

    # Initialize vector for storing tortuosity values.
    T_vec = np.zeros(np.shape(ordered_centers)[0])

    for i in range(np.shape(ordered_centers)[0]):  # Loop through each equilibrium structure.
        # Copy relevant initial arrays for modification.
        hinge_vec = hinge_vec_init.copy()
        patch_arr = patch_arr_init.copy()
        shape_arr = shape_arr_init.copy()

        # Rotate hinges to their final positions for this equilibrium structure.
        for j in range(len(hinges)):  # Loop through all hinges.
            angle = ordered_centers[i, j] - hinge_vec[hinges[j]]  # Compute rotation angle.
            hingechoice = hinges[j]  # Select the hinge to rotate.
            patch_arr, shape_arr, hinge_vec = rotate_once(
                patch_arr, shape_arr, linelist, hinge_vec, hingechoice, hinges, angle
            )  # Apply rotation.

        # Convert hinge angles to radians, accounting for angles > 180°.
        moving_hinges = hinge_vec[hinges]
        moving_hinges = np.where(moving_hinges > 180, 360 - moving_hinges, moving_hinges)
        moving_hinges *= (2 * np.pi / 360)  # Convert degrees to radians.

        # Extract line segment start and end points.
        line_starts = patch_arr[:, ::2]  # Every other column represents start points.
        line_ends = patch_arr[:, 1::2]  # The next columns represent end points.

        # Calculate run-rise vectors (differences between end and start points).
        run_rise = line_ends - line_starts
        lengths = np.linalg.norm(run_rise, axis=0)  # Compute segment lengths.

        # Variables `a` and `b` represent lengths of adjacent segments.
        a = lengths[:-1]  # All but the last length.
        b = lengths[1:]   # All but the first length.
        gamma = moving_hinges  # Hinge angles in radians.

        # Compute `c` (third side of the triangle formed by `a`, `b`, and `gamma`).
        c = np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(gamma))

        # Handle potential numerical warnings/errors during `alpha` calculation.
        while True:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    alpha = np.arcsin(a / c * np.sin(gamma))  # Compute angles.
                    break
            except RuntimeWarning:
                alpha = np.pi / 2  # Default to 90° if a runtime error occurs.
                break

        # Compute the triangle height using `a` and `alpha`.
        height = a * np.sin(alpha)  # Approximation; assumes a right triangle.

        # Calculate the area of the triangle.
        area = 0.5 * c * height

        # Compute circumradius `r` and curvature (1/r) for each segment.
        r = (a * b * c) / (4 * area)
        curvature = 1 / r

        # Weighted sum of curvatures for adjacent segments.
        curve_sum = np.sum((a + b) * curvature)

        # Calculate tortuosity: ratio of curvature sum to total length.
        T = curve_sum / np.sum(lengths)

        # Store the calculated tortuosity for this structure.
        T_vec[i] = T

    return T_vec

def translate_symmetry_to_origin(sym_vec,shape_arr,linelist,hingechoice):
    '''This function is essential for find_symmetry to work. It moves the line of symmetry to the origin along with the rest of the structure so that it can be rotated
    Inputs:
        sym_vec: (array) the line of symmetry (xs,ys)
        shape_arr: (numpy array) the array containing the points of all shapes
        linelist: (list) contains the number of lines in each shape
        hingechoice: hinge about which the system is rotating
    Outputs:
        sym_vec: the symmetry vector moved with the rest of the structure'''

    #sym_vec = copy.deepcopy(sym_vec_init) #Create a deepcopy of the initial symmetry vector so as not to modify it
    index = sum(linelist[i] for i in range(hingechoice+1))*2-7
    xy_trans = np.array(([shape_arr[0,index]],[shape_arr[1,index]])) #the symmetry vector will have to be moved in both the x and y direction by the amount of the current x and y components
                                                                                           #of the chosen hinge
    sym_vec -= xy_trans #subtracy the x and y values from the symmetry line to move it with the rest of the structure

    return sym_vec

def rotate_symmetry(sym_vec, angle,halfit):
    '''This function rotates the line of symmetry with the rest of the structure as it folds. It is essential for fins_symmetry
    Inputs:
        sym_vec: (nummpy array) the line of symmetry to be rotated
        angle: The angle by which to rotate the line of symmetry
        halfit: boolean value determining if the angle of rotation needs to be halved (if the line of symmetry is in the middle of a hinge)
    #Outputs:
        sym_vec: rotated line of symmetry'''
    
    #sym_vec = copy.deepcopy(sym_vec_init) #Make a deepcopy of the symmetry lien so as not to modify it
    if halfit == True: #If the angle needs to be halved
        angle /= 2 #half the angle
    angle = angle*np.pi/180 #convert angle to radians
    rotation_matrix =np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]) #Build a rotation matrix
    sym_vec = np.matmul(rotation_matrix,sym_vec)#matrix multiply the rotation matrix and the lien of symmetry to rotate it

    return sym_vec

def translate_symmetry_back(sym_vec,shape_arr):
    '''This function is essential for find_symmetry to work. It moves the line of symmetry along with the rest of the structure. Note that the structure rotates such atht teh leftmost point of the first shape's patch
    is located at the origin
    Inputs:
        sym_vec: (numpy array) the line of symmetry (xs,ys)
        shape_arr: (numpy array) the array containing the points of all shapes
    Outputs:
        sym_vec: the symmetry vector moved with the rest of the structure'''

    xy_trans = np.array(([shape_arr[0,0]],[shape_arr[1,0]])) #The amount by which to translate is just teh current point where the leftmost point of the first shape is located
    sym_vec -= xy_trans #perform the tranlation to the line of symmetry

    return sym_vec

def slope_intercept(sym_vec):
    '''This function determiness the slope and intercept of the symmetry vector in order to create a line
    Inputs:
        sym_vec: nx2 array where n is the number of points. This is the line of symmetry
    Outputs:
        m: slope of the line of symmetry
        b: y-intercept of the line of symmetry'''

    m = (sym_vec[1,2]-sym_vec[1,0])/(sym_vec[0,2]-sym_vec[0,0]) #Calcualte the slope of the line of symmetry
    b = sym_vec[1,1]-m*sym_vec[0,1] #Calculate the y-intercept of the line of symmetry

    return m,b

def reflect_params(sym_vec):
    '''This function creates parameters that allow points to be reflected across a line. Detail for how this is done can be found here https://stackoverflow.com/questions/8954326/how-to-calculate-the-mirror-point-along-a-line
    Inputs:
        sym_vec: nx2 array where n is the number of points. This is the line of symmetry
    Outputs:
        Anorm: y reflection parameter
        Bnorm: x reflection parameter
        Cnorm: x,y reflection parameter
    '''
    y1 = sym_vec[1,0]
    y2 = sym_vec[1,2]
    x1 = sym_vec[0,0]
    x2 = sym_vec[0,2]
    A = y2 - y1
    B = -(x2 - x1)
    C = -A * x1 - B * y1
    M = np.sqrt(A**2 + B**2)
    Anorm = A / M
    Bnorm = B / M
    Cnorm = C / M 

    return Anorm, Bnorm, Cnorm

def rearrange_columns(array1, array2):
    '''This function solves the linear sum assignment problem and pairs points that have been reflected across the line of symmetry to corresponding points already on the other side. This is necessary
    because points refected across have no knowledge of which point the best line up with.
    Inputs:
        array1: the unreflected points(2xn) of shape_array; the points above the line of symmetry
        array2: the reflected points of(2xm) of shape array
    Return:
        r: nxm array that contains the total distance of reflected points to unreflected points. This works as a matrix of residuals for the sum assignment
        row_ind: the modified row indices
        col_ind: optimal arrangement of columns such that points are paired with each other'''
    
    xsub = np.subtract.outer(array1[0,:], array2[0,:]) #subtract all x-points from each other. This creates an nxm array of every x point in m subtracted from every point in n
    ysub = np.subtract.outer(array1[1,:], array2[1,:]) #subtract all y-points from each other. This creates an nxm array of every y point in m subtracted from every point in n
    xsquare = np.square(xsub) #square the x-distances
    ysquare = np.square(ysub) #square the y-distances
    r = np.sqrt(xsquare + ysquare) #Create an nxm array that contains the total distance of reflected points to unreflected points

    # Apply the Hungarian algorithm to find the optimal column arrangement
    row_ind, col_ind = linear_sum_assignment(r)

    # Rearrange the columns of array2 according to the assignment
    rearranged_array2 = array2[:, col_ind]

    return r, row_ind, col_ind

def find_symmetry(ordered_centers,patch_arr_init,shape_arr_init,hinge_vec_init,linelist,hinges):
    '''This function (painstakingly) determines a dimensionless symmetry score for each equilibrium structure of a sequence. It calls a variety of functions to complete all operations. This function works by drawing a line of
    symmetry down the middle of a structure, rotating and translating that line of symmetry as the structure folds, and matching all points on either side at the completion of the fold.
    Inputs:
        ordered_centers: The mxn numpy array containing the final hinge positions of the clustercenters where m is the number of equilibrium states and n is the number of hinges
                                #This comes directly from cluster_stats
        patch_arr_init: the initial patch_arr for the sequence before any MC simulation
        shape_arr_init: the initial shape_arr for the sequence before any MC simulation
        hinge_vec_init: the initial vector containing hinges before any MC simulation
        linelist: the list containing the number of lines in each structure
        hinges: list of hinges able to move
    Outputs:
        symmetryscore_vec: a vector of all symmetry scores for each equilibrium structure in ordered_centers'''

    #Create an initial line of symmetry that will be rotated and translated with the structure to determine a final symmetry score
    #Create an initial line of symmetry that will be rotated and translated with the structure to determine a final symmetry score
    normcst = patch_arr_init[0,-1]
    symmetryscore_vec = np.zeros(np.shape(ordered_centers)[0]) #Initialize symmetry score vector with length equal to the number of equilibrium states for the sequence
    max_y = 10 #Determine an initial value for the maximum in y. 10 works well
    symmetry_y_vec0 = np.linspace(-max_y,max_y,3)#Create a vector of 3 y-values -y_max,0, and +y_max
    symmetry_vec0 = np.zeros((2,3))#create a 2x3 vector of zeros. This will be the line of symmetry
    symmetry_vec0[1,:] = symmetry_y_vec0 #input the y-values (-10,0,10)
    symmetry_x = patch_arr_init[0,-1]/2 #Determine the x-coordinate of the initial line of symmetry. Halfway through the structure
    symmetry_vec0[0,:] = symmetry_x #Place all x-value into the line of symmetry

    #Calculate symmetry for each equilibrium structure
    for i in range(np.shape(ordered_centers)[0]): #For every equilibrium structure
        #make deepcopies of relevant vectors that will need to be reused
        hinge_vec = copy.deepcopy(hinge_vec_init)
        patch_arr = copy.deepcopy(patch_arr_init)/normcst
        shape_arr = copy.deepcopy(shape_arr_init)/normcst
        sym_vec = copy.deepcopy(symmetry_vec0)/normcst

        for j in range(len(hinges)):#For each hinge with the ability to move
            angle = ordered_centers[i,j] - hinge_vec[hinges[j]] #The angle by which to rotate is the difference between the final angle and the initial angle
            hingechoice = hinges[j]#Select the hinge to rotate

            if symmetry_vec0[0,1] > shape_arr_init[0,sum(linelist[i] for i in range(hingechoice+1))*2-1]: #If the line of symmetry comes after the chosen hinge
                movit = True #set movit boolean = True. This means that the line of symmetry will need to be moved
            else:
                movit = False #set movit = False meaning that the line of symmetry will not be moved

            if symmetry_vec0[0,1] < shape_arr_init[0,sum(linelist[i] for i in range(hingechoice+1))*2]: #If the line of symmetry is on the hinge (in the small gap between the hinge)
                halfit = True #set halfit boolean= True meaning that the rotation of the lien of symmetry will be halved
            else:
                halfit = False #set halfit= False meaning that the rotation will proceed normally

            sym_vec = translate_symmetry_to_origin(sym_vec,shape_arr,linelist,hingechoice) #Translate the symmetry line along with the rest of the structure. The line of symmetry does not move to the origin, it just moves with the
                                                                                    #rest of the structure
            patch_arr, shape_arr = translate_to_origin(patch_arr,shape_arr,linelist,hingechoice,sym = True)#Translate the rest of the structure to the origin. This is slightly different than translate_symmetry because this function
                                                                                                #Moves the rightmost point of the shape to the left of the hinge to the origin rather than the leftmost point on the shape to the right
            if movit == True: #If the line of symmetry needs to rotate
                sym_vec = rotate_symmetry(sym_vec,angle,halfit) #Rotate the line of symmetry appropriately
            patch_arr, shape_arr, hinge_vec = rotate(patch_arr, shape_arr, linelist, hinge_vec, hingechoice, hinges, angle) #Rotate the structure about that hinge

        sym_vec = translate_symmetry_back(sym_vec,patch_arr) #Translate the line of symmetry back with the rest of the structure
        patch_arr, shape_arr = translate_back(patch_arr,shape_arr) #Translate the rest of the structure back so that the leftmost point of the first cube is at the origin

        sym_vec = sym_vec[:, sym_vec[0, :].argsort()] #sort the columns of sym_vec in order of increasing x value
        #Points of the line array will need to be reflected across the line of symmetry. https://stackoverflow.com/questions/8954326/how-to-calculate-the-mirror-point-along-a-line details the process
        #of reflecting points across a line
        A,B,C = reflect_params(sym_vec) #Create reflection parameters
        D = lambda x,y: A*x + B*y + C #Define a lamda equation that takes each x and y value of a point and calculates the distance from that point to the line of symmetry
        r = np.array([D(shape_arr[0,i],shape_arr[1,i]) for i in range(np.shape(shape_arr)[1])]) #Create a row vector contianing the distance from each point in shape_arr to the line of symmetry
        #Separate points from one side of the lien of symmetry to the other
        mask = r >= 0 #Determine which points in r have a negative distance from the line of symmetry
        uppermask = r <= 0 #Determine all points in r that have a positive distance from the line of symmetry
        x = shape_arr[0,mask] #record all x-values on the side of the line taht need to be reflected
        y = shape_arr[1,mask] #record all y-values on the side of the line that need to be reflected
        r = r[mask] #record all of the distances from each point that needs to be reflected to the line of reflection
        xprime = x-2*A*r #mirror the x points to the other side of the line of symmetry
        yprime = y-2*B*r #mirror the x points to the other side of the line of symmetry
        mirror_shape_arr = np.vstack((xprime,yprime)) #create one 2xn array with all x and y points that have been reflected where n is the nuber of points with negative distance from the LOS
        upper_shape_arr = shape_arr[:,uppermask] #Create a 2xm array of all points that were not reflected where m is the number of points not reflected/above the line of symmetry

        #Solve the linear sum assignment problem to pair reflected and unreflected points to calculate a symmetry score
        residual_arr, row_ind,col_ind = rearrange_columns(upper_shape_arr, mirror_shape_arr) #perform the linear suma ssignemnt problem to calculate residuals and determine the optimal row and column
                                                                                                #positions to satisfy the assignment problem for the structure mirrorred across the line of symmetry
        #normal_resids,row,col = rearrange_columns(upper_shape_arr,shape_arr[:,mask]) #perform the linear suma ssignemnt problem to calculate residuals and determine the optimal row and column
                                                                                        #positions to satisfy the assignment problem for the structure in its native, unfolded state. This is for normalization
        rmean = np.mean(residual_arr[row_ind,col_ind])#Sum the residuals for the correct pairing of points. residual_arr[row_ind,col_ind] is the residuals for the points in the optimal orientation to solve the
                                                    #assignment problem
        symmetry_score = 1-rmean #Determine the symmetry score. The normalsum will always be greater than or equal to the rsum, so this is chosen in order to have a symmetry score between 0 and 1
        symmetryscore_vec[i] = symmetry_score #Store the symmetry score for that structure

    return symmetryscore_vec

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

def load_mainframe(filename):
    '''This function loads the saved data from a csv file, deserializes it, and reformats it into a pandas dataframe
    Inputs:
        filename: the entire filename (including the path)
    Outputs:
        mainframe: pandas dataframe holding all folding data from megasims
    '''
    mainframe_serialized = pd.read_csv(filename,index_col=0) #read in the data
    mainframe = mainframe_serialized.applymap(deserialize_complex_data) #serialize it

    return mainframe

def analyze_structure_numbernorm(analyzed_mainframe, row):
    """
    Analyzes the structural properties of a sequence at a given row in the analyzed_mainframe DataFrame.

    Parameters:
    analyzed_mainframe (DataFrame): DataFrame containing information about the sequences, including shapes, hinges, and probabilities.
    row (int): The index of the row to analyze in the analyzed_mainframe.

    Returns:
    tuple: A dictionary with calculated structural metrics and the simulation name (string).
    """

    # Create a simulation name for the row
    sim = 'Sequence ' + str(row + 1)
    
    # Extract the data for the specified simulation row
    mf = pd.DataFrame(index=[sim], columns=analyzed_mainframe.columns)
    mf.iloc[0, :] = analyzed_mainframe.loc[sim, :]
    # Convert the row data into a dictionary for easier access
    data_dict = {col: mf[col].values[0] for col in mf.columns}
    
    # Extract shapes and generate relevant structures and properties
    shapes = data_dict['shapes']
    hinge_vec, shape_arr, sequence, linelist, patch_arr = generate(shapes)
    # Get the hinge locations from the sequence
    hinges = moving_hinges(sequence)
    
    # Extract final hinge positions and probabilities
    ordered_centers = np.array(data_dict['final hinges'])
    probs = np.array(data_dict['probability'])
    
    # Calculate various structural metrics
    e2e_vec = end_to_end_number_norm(ordered_centers, patch_arr, shape_arr, hinge_vec, linelist, hinges)  # End-to-end distances
    de2e_vec = delta_end_to_end_number_norm(ordered_centers, patch_arr, shape_arr, hinge_vec, linelist, hinges)  # Change in end-to-end distances
    Rg_vec = find_Rg_number_norm(ordered_centers, patch_arr, shape_arr, hinge_vec, linelist, hinges)  # Radius of gyration
    dRg_vec = delta_Rg(ordered_centers, patch_arr, shape_arr, hinge_vec, linelist, hinges)  # Change in radius of gyration
    T_vec = find_tortuosity(ordered_centers, patch_arr, shape_arr, hinge_vec, linelist, hinges, shapes)  # Tortuosity
    sym_vec = find_symmetry(ordered_centers, patch_arr, shape_arr, hinge_vec, linelist, hinges)  # Symmetry
    
    # Compute weighted averages and standard deviations for each metric
    avg_e2e, std_e2e = weighted_avg_and_std(e2e_vec, probs)
    avg_de2e, std_de2e = weighted_avg_and_std(de2e_vec, probs)
    avg_Rg, std_Rg = weighted_avg_and_std(Rg_vec, probs)
    avg_dRg, std_dRg = weighted_avg_and_std(dRg_vec, probs)
    avg_T, std_T = weighted_avg_and_std(T_vec, probs)
    avg_sym, std_sym = weighted_avg_and_std(sym_vec, probs)
    
    # Store calculated metrics back into the data dictionary
    data_dict['e2e'] = e2e_vec
    data_dict['avg e2e'] = avg_e2e
    data_dict['std e2e'] = std_e2e
    data_dict['de2e'] = de2e_vec
    data_dict['avg de2e'] = avg_de2e
    data_dict['std de2e'] = std_de2e
    data_dict['Rg'] = Rg_vec
    data_dict['avg Rg'] = avg_Rg
    data_dict['std Rg'] = std_Rg
    data_dict['dRg'] = dRg_vec
    data_dict['avg dRg'] = avg_dRg
    data_dict['std dRg'] = std_dRg
    data_dict['T'] = T_vec
    data_dict['avg T'] = avg_T
    data_dict['std T'] = std_T
    data_dict['symmetry'] = sym_vec
    data_dict['avg symmetry'] = avg_sym
    data_dict['std symmetry'] = std_sym
    
    # Return the dictionary of metrics and the simulation name
    return data_dict, sim

def analyze_dataset_numbernorm(mainframe, row_min, row_max):
    """
    Analyzes a subset of rows in the given mainframe dataset, calculating structural metrics
    for each sequence using multiprocessing for efficiency.

    Parameters:
    mainframe (DataFrame): The main dataset containing sequence information.
    row_min (int): The starting row index for the subset to analyze.
    row_max (int): The ending row index (exclusive) for the subset to analyze.

    Returns:
    DataFrame: A new DataFrame containing the original data along with calculated structural metrics.
    """

    # Select the subset of rows to analyze
    mainframe = mainframe.iloc[row_min:row_max, :]
    cols = list(mainframe.columns)  # Extract column names of the original DataFrame
    numcols = len(cols)  # Number of original columns
    indices = list(mainframe.index)  # Row indices of the subset
    
    # Define additional columns for the new metrics
    more_cols = [
        'e2e', 'de2e', 'Rg', 'dRg', 'T', 'symmetry', 
        'avg e2e', 'avg de2e', 'avg Rg', 'avg dRg', 'avg T', 'avg symmetry', 
        'std e2e', 'std de2e', 'std Rg', 'std dRg', 'std T', 'std symmetry'
    ]
    new_cols = cols + more_cols  # Combine original and additional columns
    
    # Create a new DataFrame to hold the analyzed data
    analyzed_mainframe = pd.DataFrame(index=indices, columns=new_cols)
    analyzed_mainframe.iloc[:, 0:numcols] = mainframe.iloc[:, 0:numcols]  # Copy original data to the new DataFrame

    # Set up multiprocessing
    num_workers = mp.cpu_count()  # Determine the number of available CPU cores
    pool = mp.Pool(processes=num_workers)  # Instantiate a pool of workers
    results = []  # List to store asynchronous results
    
    # Launch a separate process for each row in the subset
    for row in range(len(indices)):
        result = pool.apply_async(analyze_structure_numbernorm, (analyzed_mainframe, row))
        results.append(result)

    # Close the pool to prevent further task submissions and wait for all tasks to complete
    pool.close()
    pool.join()

    # Collect the results and update the analyzed DataFrame
    for result in results:
        data_dict, index = result.get()  # Retrieve the output from each worker
        analyzed_mainframe.loc[index] = data_dict  # Update the corresponding row in the DataFrame

    # Return the fully analyzed DataFrame
    return analyzed_mainframe

def length_analysis(analyzed_mainframe, length_list, param_list, hinge_list=[]):
    """
    Analyzes structural metrics from the given DataFrame based on specified lengths and hinge numbers.

    Parameters:
    analyzed_mainframe (DataFrame): DataFrame containing analyzed structural data.
    length_list (list): List of lengths to filter the data by.
    param_list (list): List of parameters to analyze (e.g., ['e2e', 'Rg']).
    hinge_list (list, optional): List of hinge numbers to further filter the data. Default is an empty list.

    Returns:
    Tuple[DataFrame, DataFrame]: 
        - av_df: DataFrame containing average values of parameters for each length and hinge number.
        - std_df: DataFrame containing standard deviations of parameters for each length and hinge number.
    """

    # Initialize arrays to store averages and standard deviations
    av_array = np.zeros((0, len(param_list)))  # Array for averages
    std_array = np.zeros((0, len(param_list)))  # Array for standard deviations
    track_len = []  # To track the lengths corresponding to each row
    track_hinges = []  # To track the hinge numbers corresponding to each row

    # Iterate over the lengths in the length list
    for i in length_list:
        # Filter rows in the DataFrame by the current length
        l_mainframe = analyzed_mainframe.loc[analyzed_mainframe['length'] == i]
        
        if len(hinge_list) != 0:  # If hinge_list is provided, analyze by hinge numbers as well
            for k in hinge_list:
                # Further filter by hinge number
                l_h_mainframe = l_mainframe.loc[l_mainframe['hinge number'] == k]
                
                # Extract parameter values as a numpy array
                arr = l_h_mainframe.loc[:, 'avg e2e':'avg symmetry'].to_numpy()
                
                # Scale the third column (e.g., Rg) by the length
                arr[:, 2] = arr[:, 2] * i
                
                # Calculate means and standard deviations for the parameters
                avs = np.mean(arr, axis=0)
                if len(l_h_mainframe) == 1:  # Handle cases with a single row
                    stds = np.zeros(np.shape(arr)[1])
                else:
                    stds = st.tstd(arr, axis=0)
                
                # Append results to arrays and track corresponding lengths and hinges
                av_array = np.vstack((av_array, avs))
                std_array = np.vstack((std_array, stds))
                track_len.append(i)
                track_hinges.append(k)
        else:  # If no hinge_list is provided, analyze only by length
            arr = l_mainframe.loc[:, 'avg e2e':'avg symmetry'].to_numpy()
            
            # Scale the third column (e.g., Rg) by the length
            arr[:, 2] = arr[:, 2] * i
            
            # Calculate means and standard deviations for the parameters
            avs = np.mean(arr, axis=0)
            stds = st.tstd(arr, axis=0)
            
            # Append results to arrays and track corresponding lengths
            av_array = np.vstack((av_array, avs))
            std_array = np.vstack((std_array, stds))
            track_len.append(i)
            track_hinges.append(np.nan)  # Use NaN to indicate no specific hinge number

    # Convert tracking lists to arrays and append them as columns to the results
    track_len = np.array(track_len)[:, None]  # Reshape to column vector
    track_hinges = np.array(track_hinges)[:, None]  # Reshape to column vector
    av_array = np.hstack((track_len, track_hinges, av_array))
    std_array = np.hstack((track_len, track_hinges, std_array))

    # Define column names for the output DataFrames
    columnlist = ['length', 'hinges', 'e2e', 'de2e', 'Rg', 'dRg', 'tortuosity', 'symmetry']
    
    # Create DataFrames for averages and standard deviations
    av_df = pd.DataFrame(av_array, columns=columnlist)
    std_df = pd.DataFrame(std_array, columns=columnlist)

    # Return the resulting DataFrames
    return av_df, std_df

def fit_length_data(av_df, std_df, length_list, log_slope=0):
    '''
    This function takes the analyzed averages from av_df for each chain length and fits an exponential function to the data. 
    The resulting fit appears to have an exponent of ~3/4.
    Inputs:
        av_df: The dataframe with averages for each tested chain length.
        std_df: The standard deviations of the data collected for each chain length.
        length_list: The list of lengths with included analyses in av_df and std_df.
    Outputs:
        log_slope: The slope of the linearized exponential. The exponent of the exponential fit.
        log_intercept: The intercept of the linearized exponential. The exponential of this number is the coefficient of the exponential fit.
        residuals: The r² value of the linearized fit.
    '''

    # Extract and make deep copies of Rg values and their standard deviations from the dataframes
    Rg_vals = copy.deepcopy(av_df.loc[:, 'Rg'].to_numpy())
    Rg_stds = copy.deepcopy(std_df.loc[:, 'Rg'].to_numpy())

    # Transform Rg values and chain lengths to their logarithms for linearizing the exponential relationship
    log_Rg_vals = np.log(Rg_vals)  # Log-transform Rg values
    log_Rg_stds = np.abs(1 / Rg_vals) * Rg_stds  # Propagate standard deviations for log-transformed values
    log_length_list = np.log(length_list)  # Log-transform chain lengths

    if log_slope == 0:
        # Perform linear fit (log-log space) if log_slope is not predefined
        fit_arr = np.polyfit(log_length_list, log_Rg_vals, 1, full=True)
        log_slope = fit_arr[0][0]  # Slope of the log-log fit
        log_intercept = fit_arr[0][1]  # Intercept of the log-log fit
        residuals = fit_arr[1][0]  # Residual sum of squares
    else:
        # Use predefined log_slope to calculate log_intercept and residuals
        log_intercept = np.mean(log_Rg_vals - log_slope * log_length_list)
        residuals = np.std(log_Rg_vals - log_slope * log_length_list)

    # Generate points for plotting the fitted curves in both log-log and linear spaces
    log_x = np.linspace(0, log_length_list[-1], 100)  # Log-log x-values for plotting
    log_y = log_slope * log_x + log_intercept  # Log-log y-values using the fit
    exp_x = np.linspace(0, length_list[-1], 100)  # Original space x-values for plotting
    exp_y = np.exp(log_intercept) * exp_x ** log_slope  # Original space exponential fit

    # Plot log-log data and the linearized fit
    plt.figure(figsize=(6, 5))
    plt.errorbar(
        log_length_list, log_Rg_vals, yerr=log_Rg_stds, fmt='o',
        color='black',  # Marker line color
        ecolor='black',  # Error bar color
        markerfacecolor='#08d1f9',  # Marker fill color
        markeredgecolor='black',  # Marker outline color
        markersize=8, capsize=5,  # Marker and error bar style
        label='Simulation Data', zorder=1  # Data appearance settings
    )
    plt.plot(
        log_x, log_y, linestyle='--', color='#f1910c', linewidth=3, zorder=0,
        label='y = ' + str(round(log_slope, 3)) + 'x + ' + str(round(log_intercept, 3))
    )
    plt.xlabel('Ln(Chain Length)', fontname='Helvetica')  # X-axis label
    plt.ylabel('Ln(Rg [µm])')  # Y-axis label
    plt.xticks([0, 1, 2, 3, 4], fontname='Helvetica')  # X-axis ticks
    plt.xlim([0, 4.3])  # X-axis limits
    # plt.legend(loc='best')  # Legend placement (commented out)
    plt.show()

    # Plot original space data and the exponential fit
    plt.figure(figsize=(6, 5))
    plt.errorbar(
        length_list, Rg_vals, Rg_stds, fmt='o',
        color='black', ecolor='black', markerfacecolor='#08d1f9', markeredgecolor='black',
        markersize=8, capsize=5, label='Simulation Data', zorder=1
    )
    plt.plot(
        exp_x, exp_y, linestyle='--', color='#f1910c', linewidth=3, zorder=0,
        label='y = ' + str(round(np.exp(log_intercept), 3)) + 'x$^{{{}}}$'.format(round(log_slope, 3))
    )
    plt.xlabel('Sequence Length')  # X-axis label
    plt.ylabel('Radius of Gyration (µm)')  # Y-axis label
    # plt.legend(loc='best')  # Legend placement (commented out)
    plt.show()

    # Plot scaled data against fitted curve in original space
    plt.figure()
    plt.errorbar(
        np.power(length_list, log_slope), Rg_vals, Rg_stds, fmt='o',
        capsize=5, color='b', label='Simulation Data'
    )
    plt.plot(
        exp_x ** log_slope, exp_y, linestyle='--', color='g', label='Best Fit'
    )
    plt.xlabel('Sequence Length$^{{{}}}$'.format(round(log_slope, 3)))  # X-axis label
    plt.ylabel('Radius of Gyration (µm)')  # Y-axis label
    plt.show()

    return log_slope, log_intercept, residuals  # Return calculated fit parameters and residuals

def normRg_vs_hinges(analyzed_mainframe, log_intercept, log_slope, lengths=[], total_length = False, show=True):
    """
    This function normalizes the radius of gyration (Rg) data by scaling it based on chain length 
    and a given log-log fit. It also computes averages and standard deviations of the normalized Rg 
    as a function of hinge numbers.
    
    Inputs:
        analyzed_mainframe: DataFrame containing chain analysis data.
        log_intercept: Intercept from the log-log fit used for normalization.
        log_slope: Slope from the log-log fit used for normalization.
        lengths: Optional list of chain lengths to filter the data. Default is an empty list (no filtering).
        show: Boolean to control whether to display the plot. Default is True.
    
    Outputs:
        condensed_arr: A 2D array with unique hinge numbers and corresponding averaged normalized Rg values.
        stds: A list of standard deviations for the normalized Rg values at each unique hinge number.
    """

    # Deep copy the input DataFrame to avoid modifying the original
    anaframe = copy.deepcopy(analyzed_mainframe)

    # Extract raw Rg values and chain lengths from the DataFrame
    Rgs_raw = anaframe.loc[:, 'avg Rg'].to_numpy()
    chain_len = anaframe.loc[:, 'length'].to_numpy()
    total_len = (anaframe.loc[:,'avg e2e'] + anaframe.loc[:,'avg de2e']).to_numpy()

    # Scale raw Rg values by the chain length
    Rgs = Rgs_raw * chain_len

    # Normalize Rg values using the log-log fit parameters
    if total_length == False:
        norm_Rgs = Rgs / (np.exp(log_intercept) * chain_len**log_slope)
    else:
        norm_Rgs = Rgs / (np.exp(log_intercept) * total_len**log_slope)

    # Extract hinge numbers from the DataFrame
    N_hinges = anaframe.loc[:, 'hinge number'].to_numpy()

    # Combine hinge numbers, normalized Rg values, and chain lengths into a single array
    plot_arr = np.vstack((N_hinges, norm_Rgs, chain_len))

    # Filter the data by the specified lengths, if provided
    if len(lengths) != 0:
        # Identify indices of rows with chain lengths not in the specified list
        indices_to_remove = [i for i, val in enumerate(plot_arr[2]) if val not in lengths]

        # Remove unwanted rows in reverse order to maintain indexing
        for i in reversed(indices_to_remove):
            plot_arr = np.delete(plot_arr, i, axis=1)

    # Sort the array by hinge numbers (first row)
    sorted_indices = np.argsort(plot_arr[0])
    sorted_arr = plot_arr[:, sorted_indices]

    # Identify unique hinge numbers and their starting indices
    unique_values, indices = np.unique(sorted_arr[0], return_index=True)

    # Compute averages and standard deviations for normalized Rg values at each unique hinge number
    averages = [np.mean(sorted_arr[1, np.where(sorted_arr[0] == val)]) for val in unique_values]
    stds = [np.std(sorted_arr[1, np.where(sorted_arr[0] == val)]) for val in unique_values]

    # Create a condensed array with unique hinge numbers and corresponding averaged values
    condensed_arr = np.array([unique_values, averages])

    # Plot the normalized Rg values against hinge numbers if `show` is True
    if show:
        plt.figure(figsize=(6, 5))
        plt.errorbar(
            condensed_arr[0, :], condensed_arr[1, :], stds, fmt='o',
            color='black',  # Line color (if connecting markers)
            ecolor='black',  # Error bar color
            markerfacecolor='#08d1f9',  # Marker fill color
            markeredgecolor='black',  # Marker outline color
            markersize=8,  # Marker size
            capsize=5  # Error bar cap size
        )
        plt.xlabel('Number of Hinges', fontsize=20)  # X-axis label
        plt.xlim([0, 20.5])  # X-axis limits
        plt.ylabel(r'$\frac{Rg}{Chain Length^{3/4}}$', rotation=90, fontsize=20)  # Y-axis label
        plt.show()

    return condensed_arr, stds

def find_degree_fit(reduced_df):
    # Extract the x and y values from the reduced DataFrame, excluding the last two columns
    x = reduced_df[0, :-2]
    y = reduced_df[1, :-2]
    
    # Define the polynomial degrees to be tested
    degrees = [1, 2, 3, 4]  # Linear to quartic polynomial fits
    fits = {}  # Dictionary to store polynomial fits for each degree
    
    # Loop through each degree and fit a polynomial to the data
    for degree in degrees:
        coeffs = np.polyfit(x, y, degree)  # Fit polynomial and get coefficients
        poly = np.poly1d(coeffs)  # Create polynomial function from coefficients
        fits[degree] = poly  # Store the polynomial function in the dictionary

    # Create the colormap (reversed viridis)
    cmap = plt.get_cmap('viridis_r')
    
    # Generate a list of colors, one for each polynomial degree
    colors = cmap(np.linspace(0, 1, len(degrees)))

    # Plot the polynomial fits
    plt.figure(figsize=(6, 5))  # Set figure size
    plt.scatter(
        x, y, color='blue', label='Data Points', 
        edgecolor='black',  # Marker outline color (black)
        facecolor='#08d1f9',  # Marker fill color (light blue)
        s=80, zorder=5  # Marker size and z-order
    )  # Scatter plot of original data points
    
    # Generate x values for plotting the polynomial fits
    x_fit = np.linspace(min(x), max(x), 100)
    
    # Plot each polynomial fit with a corresponding color from the colormap
    for i, (degree, poly) in enumerate(fits.items()):
        plt.plot(
            x_fit, poly(x_fit), 
            label=f'Polynomial Degree {degree}', 
            color=colors[i], linewidth=2, zorder=i
        )

    # Add plot labels and legend
    plt.xlabel('x')  # Label for x-axis
    plt.ylabel('y')  # Label for y-axis
    plt.legend(fontsize=12)  # Add a legend with font size 12
    plt.show()  # Display the plot

    # Calculate R-squared values for each polynomial fit
    r_squared = {}  # Dictionary to store R-squared values for each degree
    rs = []  # List to store R-squared values for plotting
    for degree, poly in fits.items():
        y_fit = poly(x)  # Predicted y values from the polynomial fit
        ss_res = np.sum((y - y_fit) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
        r_squared[degree] = 1 - (ss_res / ss_tot)  # Calculate R-squared
        rs.append(1 - (ss_res / ss_tot))  # Append to the list

    # Plot the R-squared values for each polynomial degree
    #plt.figure()  # Create a new figure
    #plt.scatter(degrees, rs)  # Scatter plot of degrees vs. R-squared values
    #plt.show()  # Display the plot

    return fits  # Return the dictionary of polynomial fits

def fitandscaledata(reduced_df, stds, degree):
    """
    Fits a polynomial to the given data, plots the fit, and scales the data and standard deviations 
    by the polynomial fit values.

    Parameters:
    reduced_df (numpy array): 2D array where the first row contains x-values and the second row contains y-values.
    stds (numpy array): Standard deviations corresponding to y-values in reduced_df.
    degree (int): Degree of the polynomial to be fit to the data.

    Returns:
    fit (numpy.poly1d): Polynomial function representing the fit.
    scaled_df (numpy array): The data scaled by the polynomial fit.
    scaled_stds (numpy array): The standard deviations scaled by the polynomial fit.
    """
    
    # Extract x and y values from the reduced DataFrame
    x = reduced_df[0, :]  # x-values
    y = reduced_df[1, :]  # y-values

    # Fit a polynomial of the specified degree to the data
    coeffs = np.polyfit(x, y, degree)  # Polynomial coefficients
    fit = np.poly1d(coeffs)  # Polynomial function

    # Generate x and y values for the polynomial fit curve
    x_fit = np.linspace(1, reduced_df[0, -1], 100)  # Create evenly spaced x-values for the fit curve
    y_fit = fit(x_fit)  # Compute corresponding y-values using the polynomial fit

    # Plot the polynomial fit and the original data with error bars
    plt.plot(x_fit, y_fit, label=f'Polynomial Fit (Degree {degree})', linestyle='--', color='#f1910c')  # Plot the fit curve
    plt.errorbar(
        reduced_df[0, :], reduced_df[1, :], stds, fmt='o', 
        label='Data with Error Bars',
        color='black',  # Line color (if connecting markers)
        ecolor='black',  # Error bar color
        markerfacecolor='#08d1f9',  # Marker fill color
        markeredgecolor='black',  # Marker outline color
        markersize=8,  # Marker size
        capsize=5  # Error bar cap size
    )  # Plot data points with error bars
    plt.legend()  # Add a legend to the plot
    plt.show()  # Display the plot

    # Scale the data and standard deviations by the polynomial fit values
    scaled_df = copy.deepcopy(reduced_df) / fit(reduced_df[0, :])  # Scale the data
    scaled_stds = copy.deepcopy(stds) / fit(reduced_df[0, :])  # Scale the standard deviations

    # Return the polynomial fit, scaled data, and scaled standard deviations
    return fit, scaled_df, scaled_stds

def simultaneous_scaling_shape_analysis(analyzed_mainframe, shape_list, param_list, log_intercept, log_slope, fit, hinge_list = []):
    """
    This function performs simultaneous scaling and shape analysis for a set of data,
    calculates the average, standard deviation, and standard error for a range of parameters, 
    and returns the results as DataFrames.

    Parameters:
    analyzed_mainframe (pandas DataFrame): A DataFrame containing the analyzed data, with columns for shape, length, and various parameters.
    shape_list (list): A list of shapes (e.g., triangle percentage) to be analyzed.
    param_list (list): A list of parameters to be considered (e.g., e2e, de2e, etc.).
    log_intercept (float): The logarithmic intercept used for scaling the data.
    log_slope (float): The logarithmic slope used for scaling the data.
    fit (function): A function used to fit the data based on hinge values.
    hinge_list (list, optional): A list of hinge numbers to filter the data. Default is an empty list, meaning no filtering by hinge.

    Returns:
    av_df (pandas DataFrame): DataFrame containing the average values for each parameter.
    std_df (pandas DataFrame): DataFrame containing the standard deviations for each parameter.
    sterr_df (pandas DataFrame): DataFrame containing the standard errors for each parameter.
    """

    # Initialize arrays to store average, standard deviation, and standard error values for the parameters
    av_array = np.zeros((0, len(param_list)))  # Stores average values
    std_array = np.zeros((0, len(param_list)))  # Stores standard deviations
    sterr_array = np.zeros((0, len(param_list)))  # Stores standard errors

    # Initialize lists to track shape, length, and hinge values
    track_len = []
    track_shape = []
    track_hinges = []

    # Loop through each shape in the provided shape_list
    for i in shape_list:
        # Filter the data for the current shape
        s_mainframe = analyzed_mainframe.loc[analyzed_mainframe['% triangles'] == i]
        
        # Get the unique lengths for the current shape
        length_list = sorted(list(set(s_mainframe.loc[:, 'length'])))

        # Loop through each length in the list of lengths
        for length in length_list:
            # Filter the data for the current length
            s_l_mainframe = s_mainframe.loc[s_mainframe['length'] == length]

            # If hinge_list is provided, further filter by hinge number
            if len(hinge_list) != 0:
                for k in hinge_list:
                    # Filter the data for the current hinge number
                    s_l_h_mainframe = s_l_mainframe.loc[s_l_mainframe['hinge number'] == k]
                    
                    # Extract the relevant parameter data
                    arr = s_l_h_mainframe.loc[:, 'avg e2e':'avg symmetry'].to_numpy()

                    # Calculate the total length (sum of avg e2e and avg de2e)
                    total_length = s_l_h_mainframe.loc[:, 'avg e2e'].to_numpy() + s_l_h_mainframe.loc[:, 'avg de2e'].to_numpy()

                    # Perform scaling of the third column (Rg or other parameter) based on length and other factors
                    arr[:, 2] = arr[:, 2] * length  # Scale the third column by the length
                    arr[:, 2] = arr[:, 2] / (np.exp(log_intercept) * total_length**log_slope)  # Apply logarithmic scaling
                    arr[:, 2] = arr[:, 2] / fit(k)  # Further scaling based on the fit function

                    # Compute the average, standard deviation, and standard error of the parameters
                    avs = np.mean(arr, axis=0)
                    if len(s_l_h_mainframe) == 1:
                        stds = np.zeros(np.shape(arr)[1])
                        sterrs = np.zeros(np.shape(arr)[1])
                    else:
                        stds = st.tstd(arr, axis=0)  # Standard deviation
                        sterrs = stds / np.sqrt(len(arr))  # Standard error

                    # Append the results to the corresponding arrays
                    av_array = np.vstack((av_array, avs))
                    std_array = np.vstack((std_array, stds))
                    sterr_array = np.vstack((sterr_array, sterrs))

                    # Track the shape, length, and hinge number
                    track_shape.append(i)
                    track_len.append(length)
                    track_hinges.append(k)
            else:
                # No hinge filtering, just scale based on length
                arr = s_l_mainframe.loc[:, 'avg e2e':'avg symmetry'].to_numpy()
                total_length = s_l_mainframe.loc[:, 'avg e2e'].to_numpy() + s_l_mainframe.loc[:, 'avg de2e'].to_numpy()
                arr[:, 2] = arr[:, 2] * length
                arr[:, 2] = arr[:, 2] / (np.exp(log_intercept) * total_length**log_slope)
                arr[:, 2] = arr[:, 2] / fit(k)
                avs = np.mean(arr, axis=0)
                stds = st.tstd(arr, axis=0)
                sterrs = stds / np.sqrt(len(arr))

                # Append the results to the arrays
                av_array = np.vstack((av_array, avs))
                std_array = np.vstack((std_array, stds))
                sterr_array = np.vstack((sterr_array, sterrs))

                # Track the shape and length without hinge
                track_shape.append(i)
                track_len.append(length)
                track_hinges.append(np.nan)

    # Convert the tracking lists to numpy arrays and combine them with the result arrays
    track_shape = np.array(track_shape)[:, None]
    track_len = np.array(track_len)[:, None]
    track_hinges = np.array(track_hinges)[:, None]
    av_array = np.hstack((track_shape, track_len, track_hinges, av_array))
    std_array = np.hstack((track_shape, track_len, track_hinges, std_array))
    sterr_array = np.hstack((track_shape, track_len, track_hinges, sterr_array))

    # Define the column names for the output DataFrames
    columnlist = ['triangles', 'length', 'hinges', 'e2e', 'de2e', 'Rg', 'dRg', 'tortuosity', 'symmetry']

    # Create DataFrames for average values, standard deviations, and standard errors
    av_df = pd.DataFrame(av_array, columns=columnlist)
    std_df = pd.DataFrame(std_array, columns=columnlist)
    sterr_df = pd.DataFrame(sterr_array, columns=columnlist)

    # Return the DataFrames containing the results
    return av_df, std_df, sterr_df

def lengthhingenormRg_vs_triangles(av_df, std_df, shape_list, sterr_df):
    """
    This function plots the normalized radius of gyration (Rg) against the percentage of triangles in a dataset, 
    and calculates the average Rg and standard deviation for each shape in the given shape list.

    Parameters:
    av_df (pandas DataFrame): DataFrame containing average values for different parameters.
    std_df (pandas DataFrame): DataFrame containing standard deviation values for different parameters.
    shape_list (list): A list of unique shapes (triangles percentages) to be analyzed.
    sterr_df (pandas DataFrame): DataFrame containing the standard error values for different parameters.

    Returns:
    plot_arr (numpy array): Array containing the original data for plotting.
    condensed_arr (numpy array): Condensed array containing the average and standard deviation values for each shape.
    """
    
    # Initialize an array to hold the data for plotting: 
    # The first row will hold triangle percentages, 
    # The second will hold Rg values, 
    # The third will hold Rg standard deviations, 
    # The fourth will hold Rg standard errors
    plot_arr = np.zeros((4, len(av_df)))
    plot_arr[0, :] = av_df['triangles']  # Triangle percentage
    plot_arr[1, :] = av_df['Rg']  # Radius of gyration (Rg)
    plot_arr[2, :] = std_df['Rg']  # Standard deviation of Rg
    plot_arr[3, :] = sterr_df['Rg']  # Standard error of Rg

    # Remove columns where any value in the array is NaN
    columns_with_nan = np.isnan(plot_arr).any(axis=0)
    plot_arr = plot_arr[:, ~columns_with_nan]

    # Initialize an array to hold the condensed data for each shape in shape_list
    condensed_arr = np.zeros((4, len(shape_list)))

    # Loop through each shape in the shape_list and condense the data
    for i in range(len(shape_list)):
        # Extract data for the current shape (percentage of triangles)
        work_arr = plot_arr[:, plot_arr[0, :] == shape_list[i]]
        
        # Store the shape percentage (triangles) in the condensed array
        condensed_arr[0, i] = shape_list[i] * 100  # Multiply by 100 for percentage
        # Store the average Rg for the current shape
        condensed_arr[1, i] = np.mean(work_arr[1, :])  # Average of Rg values
        # Store the standard error of Rg for the current shape
        condensed_arr[3, i] = np.linalg.norm(work_arr[3, :]) / np.shape(work_arr)[1]  # Norm of the standard errors, normalized by the number of samples
        # Store the standard deviation of Rg for the current shape
        condensed_arr[2, i] = condensed_arr[3, i] * np.sqrt(np.shape(work_arr)[1])  # Standard deviation from standard error

    # Plotting the data using error bars
    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.errorbar(condensed_arr[0, :], condensed_arr[1, :], yerr=condensed_arr[2, :],
                 fmt='o', capsize=5, color='black', ecolor='black',
                 markeredgecolor='black', markerfacecolor='#08d1f9', markersize=10, linewidth=1.5, markeredgewidth=1.5)

    # Labels, title, and aesthetics (commented out here, but available for customization)
    # plt.xlabel('Percentage Triangles', fontsize=20)
    # plt.ylabel('Rg Normalized to Length and Hinges', fontsize=20)
    # plt.title('Rg Sensitivity to Shape Composition', fontsize=30)
    
    # Customize the font size for ticks
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Set the linewidth for the plot spines (axes borders)
    ax = plt.gca()  # Get the current Axes instance
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Set the desired thickness here

    # Show the plot
    plt.show()

    # Return the original data array and the condensed results
    return plot_arr, condensed_arr

def simultaneous_scaling_patch_analysis(analyzed_mainframe, patches_list, param_list, log_intercept, log_slope, fit, hinge_list = []):
    """
    This function performs a simultaneous scaling and patch analysis on the given dataset. 
    It computes the average, standard deviation, and standard error for various parameters 
    (e.g., e2e, de2e, Rg, tortuosity, symmetry) based on patch numbers and hinge numbers.

    Parameters:
    analyzed_mainframe (pandas DataFrame): The DataFrame containing the data to be analyzed.
    patches_list (list): A list of patch numbers to be analyzed.
    param_list (list): A list of parameters to be calculated.
    log_intercept (float): Logarithmic intercept for scaling the data.
    log_slope (float): Logarithmic slope for scaling the data.
    fit (function): A function used to adjust the data based on the hinge number.
    hinge_list (list, optional): A list of hinge numbers to be considered. Default is an empty list.

    Returns:
    av_df (pandas DataFrame): DataFrame containing the average values for the parameters.
    std_df (pandas DataFrame): DataFrame containing the standard deviation values for the parameters.
    sterr_df (pandas DataFrame): DataFrame containing the standard error values for the parameters.
    """
    
    # Initialize arrays to store average values, standard deviations, and standard errors
    av_array = np.zeros((0, len(param_list)))  # Array for average values
    std_array = np.zeros((0, len(param_list)))  # Array for standard deviations
    sterr_array = np.zeros((0, len(param_list)))  # Array for standard errors
    
    # Arrays to track the patches, lengths, and hinges
    track_len = []
    track_hinges = []
    track_patches = []

    # Loop over each patch in the list of patches
    for i in patches_list:
        # Filter rows based on the current patch number
        m_mainframe = analyzed_mainframe[analyzed_mainframe['patch numbers'].apply(lambda x: x[0] == i)]
        
        # Get unique lengths for the current patch
        length_list = sorted(list(set(m_mainframe.loc[:, 'length'])))
        
        # Loop over each unique length for the current patch
        for length in length_list:
            # Filter the data based on the current length
            m_l_mainframe = m_mainframe.loc[m_mainframe['length'] == length]
            
            # If hinge numbers are specified, loop over them
            if len(hinge_list) != 0:
                for k in hinge_list:
                    # Filter data for the current hinge number
                    m_l_h_mainframe = m_l_mainframe.loc[m_l_mainframe['hinge number'] == k]
                    
                    # Extract relevant columns and perform scaling
                    arr = m_l_h_mainframe.loc[:, 'avg e2e':'avg symmetry'].to_numpy()
                    arr[:, 2] = arr[:, 2] * length  # Adjust values by length
                    arr[:, 2] = arr[:, 2] / (np.exp(log_intercept) * length ** log_slope)  # Apply scaling based on log intercept and slope
                    arr[:, 2] = arr[:, 2] / fit(k)  # Apply additional fitting function based on hinge number
                    
                    # Calculate the average values for each parameter
                    avs = np.mean(arr, axis=0)
                    
                    # Calculate the standard deviations and standard errors
                    if len(m_l_h_mainframe) == 1:
                        stds = np.zeros(np.shape(arr)[1])  # If only one entry, set std and sterr to zero
                        sterrs = np.zeros(np.shape(arr)[1])
                    else:
                        stds = st.tstd(arr, axis=0)  # Standard deviation for the parameters
                        sterrs = stds / np.sqrt(len(arr))  # Standard error based on the number of samples
                    
                    # Store the results in the respective arrays
                    av_array = np.vstack((av_array, avs))
                    std_array = np.vstack((std_array, stds))
                    sterr_array = np.vstack((sterr_array, sterrs))
                    
                    # Track the patch, length, and hinge numbers
                    track_patches.append(i)
                    track_len.append(length)
                    track_hinges.append(k)
            
            # If no hinge numbers are specified, analyze the data without filtering by hinge
            else:
                # Extract relevant columns and perform scaling
                arr = m_l_mainframe.loc[:, 'avg e2e':'avg symmetry'].to_numpy()
                arr[:, 2] = arr[:, 2] * length  # Adjust values by length
                arr[:, 2] = arr[:, 2] / (np.exp(log_intercept) * length ** log_slope)  # Apply scaling based on log intercept and slope
                
                # Calculate the average values for each parameter
                avs = np.mean(arr, axis=0)
                # Calculate the standard deviations and standard errors
                stds = st.tstd(arr, axis=0)
                sterrs = stds / np.sqrt(len(arr))
                
                # Store the results in the respective arrays
                av_array = np.vstack((av_array, avs))
                std_array = np.vstack((std_array, stds))
                sterr_array = np.vstack((sterr_array, sterrs))
                
                # Track the patch and length, but set hinge number as NaN since it's not used
                track_patches.append(i)
                track_len.append(length)
                track_hinges.append(np.nan)

    # Convert the tracking lists to numpy arrays and add them to the result arrays
    track_patches = np.array(track_patches)[:, None]
    track_len = np.array(track_len)[:, None]
    track_hinges = np.array(track_hinges)[:, None]
    
    # Concatenate the tracking information with the data arrays
    av_array = np.hstack((track_patches, track_len, track_hinges, av_array))
    std_array = np.hstack((track_patches, track_len, track_hinges, std_array))
    sterr_array = np.hstack((track_patches, track_len, track_hinges, sterr_array))

    # Define column names for the resulting DataFrames
    columnlist = ['weak patches', 'length', 'hinges', 'e2e', 'de2e', 'Rg', 'dRg', 'tortuosity', 'symmetry']
    
    # Convert the arrays to pandas DataFrames for better readability
    av_df = pd.DataFrame(av_array, columns=columnlist)
    std_df = pd.DataFrame(std_array, columns=columnlist)
    sterr_df = pd.DataFrame(sterr_array, columns=columnlist)

    # Return the DataFrames containing average, standard deviation, and standard error values
    return av_df, std_df, sterr_df

def analyze_patches(analyzed_mainframe, param_list, log_intercept, log_slope, fit):
    """
    This function analyzes patch data from a DataFrame, performs scaling analysis on different patches, 
    and returns the average, standard deviation, and standard error for each parameter across patches.

    Parameters:
    analyzed_mainframe (pandas DataFrame): DataFrame containing the analyzed data.
    param_list (list): List of parameters to be analyzed.
    log_intercept (float): Logarithmic intercept used in scaling.
    log_slope (float): Logarithmic slope used in scaling.
    fit (function): Function used to fit the data.

    Returns:
    av_df (pandas DataFrame): DataFrame containing average values for each parameter.
    std_df (pandas DataFrame): DataFrame containing standard deviation values for each parameter.
    sterr_df (pandas DataFrame): DataFrame containing standard error values for each parameter.
    """

    # Extract patch numbers from the 'patch numbers' column of the DataFrame
    patches = analyzed_mainframe.loc[:, 'patch numbers']
    patch_list = []

    # Loop through the 'patch numbers' column to create a list of unique patches
    for i in range(len(patches)):
        patch_list.append(patches[i][0])  # Get the first element in each patch (assuming it's a list)

    # Create a sorted list of unique patches
    patches_list = sorted(list(set(patch_list)))

    # Extract hinge numbers from the 'hinge number' column
    hinges = np.array(analyzed_mainframe.loc[:, 'hinge number'])
    
    # Create a sorted list of unique hinge numbers
    hinge_list = sorted(list(set(hinges)))

    # Call the function for simultaneous scaling and patch analysis
    av_df, std_df, sterr_df = simultaneous_scaling_patch_analysis(
        analyzed_mainframe, patches_list, param_list, log_intercept, log_slope, fit, hinge_list
    )

    # Return the average, standard deviation, and standard error dataframes
    return av_df, std_df, sterr_df, patches_list

def lengthhingenormRg_vs_weakmags(av_df, std_df, sterr_df, patches_list):
    """
    This function analyzes the relationship between normalized Rg (radius of gyration) and the percentage of weak patches,
    plots the results with error bars, and returns the processed data arrays.

    Parameters:
    av_df (pandas DataFrame): DataFrame containing average values for each parameter, including 'weak patches' and 'Rg'.
    std_df (pandas DataFrame): DataFrame containing standard deviation values for each parameter.
    sterr_df (pandas DataFrame): DataFrame containing standard error values for each parameter.
    patches_list (list): List of unique patch numbers for analysis.

    Returns:
    plot_arr (numpy array): Array containing raw data for plotting.
    condensed_arr (numpy array): Condensed data for analysis, including mean values and error bars.
    """
    
    # Initialize array to hold data for plotting (4 rows: weak patches, Rg, standard deviation, standard error)
    plot_arr = np.zeros((4, len(av_df)))
    plot_arr[0, :] = av_df.loc[:, 'weak patches']  # Weak patches data
    plot_arr[1, :] = av_df.loc[:, 'Rg']  # Rg (radius of gyration) values
    plot_arr[2, :] = std_df.loc[:, 'Rg']  # Standard deviation of Rg
    plot_arr[3, :] = sterr_df.loc[:, 'Rg']  # Standard error of Rg

    # Initialize array to hold condensed data by patch
    condensed_arr = np.zeros((4, len(patches_list)))

    # Remove columns that contain NaN values
    columns_with_nan = np.isnan(plot_arr).any(axis=0)
    plot_arr = plot_arr[:, ~columns_with_nan]

    # Condense data by patches
    for i in range(len(patches_list)):
        # Extract data for the current patch
        work_arr = plot_arr[:, plot_arr[0, :] == patches_list[i]]
        
        # Calculate average Rg and error values for the current patch
        condensed_arr[0, i] = patches_list[i]  # Patch number
        condensed_arr[1, i] = np.mean(work_arr[1, :])  # Mean Rg for the patch
        condensed_arr[3, i] = np.linalg.norm(work_arr[3, :]) / np.shape(work_arr)[1]  # Standard error for Rg
        condensed_arr[2, i] = condensed_arr[3, i] * np.sqrt(np.shape(work_arr)[1])  # Standard deviation for Rg

    # Plotting the condensed data with error bars
    plt.figure(figsize=(10, 5))  # Set figure size
    plt.errorbar(condensed_arr[0, :], condensed_arr[1, :] + 0.016, yerr=condensed_arr[2, :],
                 fmt='o', capsize=5, color='black', ecolor='black',
                 markeredgecolor='black', markerfacecolor='#08d1f9', markersize=10, linewidth=1.5, markeredgewidth=1.5)

    # Aesthetic adjustments for the plot
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Adjust the spine (border) thickness
    ax = plt.gca()  # Get the current Axes instance
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Set the desired thickness here
    
    # Display the plot
    plt.show()

    # Return the raw data array and the condensed data array
    return plot_arr, condensed_arr

def plot_mags(mag_dict):
    # Dictionary for mapping magnitude categories to colors
    color_dict = {'10': '#77DD77', '25': '#f1910c', '50': '#08d1f9'}
    # Alternative color dictionary (commented out)
    # color_dict = {'10':'#77DD77','50':'#08d1f9'}

    # Create a new figure with a specific size for the plot
    plt.figure(figsize=(10, 5))

    # Loop through each key-value pair in the input dictionary `mag_dict`
    # `mag_dict` is expected to have keys as magnitude categories (e.g., '10', '25', '50')
    # and values as arrays containing the data to be plotted
    for key in mag_dict.keys():
        # Extract the condensed data for the current key
        condensed_arr = mag_dict[key]
        
        # Plot the data points using error bars
        # condensed_arr[0, :] are the x-values (e.g., some category or measurement)
        # condensed_arr[1, :] are the y-values (e.g., corresponding measurements)
        # condensed_arr[2, :] are the y-error values (uncertainty in measurements)
        plt.errorbar(condensed_arr[0, :], condensed_arr[1, :]+.016, yerr=condensed_arr[2, :],
                     fmt='o',                  # Marker style (circle)
                     capsize=5,                # Size of the caps at the ends of error bars
                     color='black',            # Color of the error bars
                     ecolor='black',           # Color of the error bars
                     markeredgecolor='black',  # Edge color of the markers (circles)
                     markerfacecolor=color_dict[key],  # Face color of the markers (based on magnitude key)
                     markersize=10,            # Size of the markers
                     linewidth=1.5,            # Width of the error bars
                     markeredgewidth=1.5)      # Width of the marker edges

    # Adjust the font size for the x and y axis tick labels
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Add a legend showing the magnitude categories (keys of `mag_dict`)
    plt.legend(mag_dict.keys())

    # Adjust the line width of the plot's borders for better visibility
    ax = plt.gca()  # Get the current Axes instance
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Set the thickness of the plot borders

    # Display the plot
    plt.show()

def filter_df(length_list, analyzed_mainframe):
    # Filter the 'analyzed_mainframe' DataFrame to only include rows where the 'length' column
    # contains values present in the 'length_list'. The '.isin()' function checks if each value
    # in the 'length' column is in the provided 'length_list'.
    filtered_length = analyzed_mainframe[analyzed_mainframe['length'].isin(length_list)]
    
    # Return the filtered DataFrame containing only the rows with specified lengths
    return filtered_length

def Rg_histograms(filtered_length0, log_slope, log_intercept, fit):
    """
    Generates and visualizes histograms of radius of gyration (Rg) metrics using various normalization approaches.

    Parameters:
    - filtered_length0: DataFrame containing structural data, including 'dRg', 'length', and 'clusternumber'.
    - log_slope: Slope of the log-log fit used for normalization.
    - log_intercept: Intercept of the log-log fit used for normalization.
    - fit: Function to apply hinge-based normalization.

    Displays:
    - Four histograms visualizing different Rg metrics:
      1. Total Rg normalized by sequence length.
      2. Rg divided by sequence length.
      3. Rg normalized using a length-based fit.
      4. Rg normalized using both length and hinge corrections.
    """

    # Create a deep copy of the filtered_length DataFrame to avoid modifying the original data
    filtered_length = copy.deepcopy(filtered_length0)

    # Extract 'dRg' column and initialize arrays for Rg values and cluster counts
    frame = filtered_length.loc[:, 'dRg']
    Rg_vec = np.zeros(0)  # Array to store concatenated Rg values
    number = []  # List to store the number of clusters per sequence

    # Loop through the sequences to concatenate 'dRg' values and collect cluster counts
    for i in range(len(frame)):
        Rg_vec = np.hstack((Rg_vec, np.array(frame[i])))
        number.append(filtered_length.loc[:, 'clusternumber'][i])

    # Extract sequence lengths and hinge counts
    lengths = filtered_length.loc[:, 'length'].to_numpy()
    hinges = filtered_length.loc[:, 'hinges'].apply(lambda x: len(x)).to_numpy()

    # Repeat lengths and hinges to match the total number of clusters
    length_vec = np.repeat(lengths, number)
    hinges_vec = np.repeat(hinges, number)

    # Apply normalization factors
    length_fit_norm = length_vec / (log_intercept * length_vec**log_slope)  # Length normalization
    length_hinge_norm = length_vec / (log_intercept * length_vec**log_slope) / fit(hinges_vec)  # Length + hinge normalization

    # Compute normalized Rg metrics
    Rg_length_fit_norm = Rg_vec * length_fit_norm
    Rg_length_hinge_norm = Rg_vec * length_hinge_norm
    Rg_total = Rg_vec * length_vec

    # Plot histogram of the total Rg normalized by length
    plt.figure(figsize=(10, 8))
    bin_edges = np.histogram_bin_edges(Rg_total, bins='auto')
    plt.hist(Rg_total, bins=bin_edges, edgecolor='black', color='#51c1bf', linewidth=1)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.title('Total dRg')
    plt.show()

    # Plot histogram of Rg divided by sequence length
    plt.figure(figsize=(10, 8))
    bin_edges = np.histogram_bin_edges(Rg_vec, bins='auto')
    plt.hist(Rg_vec, bins=bin_edges, edgecolor='black', color='#51c1bf', linewidth=1)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.title('dRg/Length')
    plt.show()

    # Plot histogram of Rg normalized to the length fit
    plt.figure(figsize=(10, 8))
    bin_edges = np.histogram_bin_edges(Rg_length_fit_norm, bins='auto')
    plt.hist(Rg_length_fit_norm, bins=bin_edges, edgecolor='black', color='#51c1bf', linewidth=1)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.title('dRg Normalized to Length Fit')
    plt.show()

    # Plot histogram of Rg normalized to both length and hinges
    plt.figure(figsize=(10, 8))
    bin_edges = np.histogram_bin_edges(Rg_length_hinge_norm, bins='auto')
    plt.hist(Rg_length_hinge_norm, bins=bin_edges, edgecolor='black', color='#51c1bf', linewidth=1)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.title('Rg normalized to length and hinges')
    plt.show()

def show_unique_structures(hingenum, shapes, clustercenters, sequence_name, high_probs, high, parameter):
    '''
    Function displays the final conformation of the sequence in each cluster based on the minimum-energy centroid of that cluster.

    Inputs:
        hingenum : list
            List of moving hinge numbers.
        
        shapes : dict
            Dictionary containing all shapes associated with the sequence.
        
        clustercenters : numpy.ndarray
            Array of centroid points for each cluster.
        
        sequence_name : str
            Name or identifier of the sequence to be visualized.
        
        high_probs : list
            List of probabilities corresponding to each centroid/cluster.
        
        high : list
            List of values (e.g., scores) corresponding to each centroid/cluster.
        
        parameter : str
            The name of the parameter associated with the visualization (e.g., 'Rg', 'tortuosity').
    '''

    # Generate all shapes in the initial state and extract related data
    hinge_vec, shape_arr, sequence, linelist, patch_arr = generate(shapes)

    # Display the initial orientation of the sequence
    shapeplots(shape_arr, linelist, title=sequence_name)

    # Loop through each cluster (based on the number of cluster centroids)
    for i in range(len(clustercenters)):
        
        # Regenerate all shapes for each cluster iteration
        hinge_vec, shape_arr, sequence, linelist, patch_arr = generate(shapes)

        # Determine the number of hinges in the current cluster
        if len(hingenum) == 1:
            cluster_index = 1
        else:
            cluster_index = np.shape(clustercenters)[1]

        # Loop through each angle of the centroid point (for each cluster center)
        for j in range(cluster_index):
            # Select the hinge that corresponds to the current centroid point
            hingechoice = hingenum[j]

            # Calculate the angle between the current hinge angle (180 degrees) and the centroid angle
            angle = clustercenters[i, j] - hinge_vec[hingechoice]

            # Rotate the structure by the calculated angle around the selected hinge
            patch_arr, shape_arr, hinge_vec = rotate_once(patch_arr, shape_arr, linelist, hinge_vec, hingechoice, hingenum, angle)

        # Display the final conformation of the sequence, showing the score and probability of the current cluster
        shapeplots(
            shape_arr,
            linelist,
            title=sequence_name + ' folds like this with \n' + parameter + ' score ' + str(high[i]) +
                  '% with probability ' + str(round(high_probs[i]*100, 1)) + '%'
        )

def determine_unique_structures_range(analyzed_mainframe0, min_val, max_val, parameter, length_list = [], show_structures = False, random_number = 0):
    """
    Filters structures based on a specified parameter range and length, and optionally visualizes or randomly selects them.

    Parameters:
    analyzed_mainframe0 : pandas.DataFrame
        The DataFrame containing the shape data for all structures.
    
    min_val : float
        The minimum value of the parameter range for filtering structures.
    
    max_val : float
        The maximum value of the parameter range for filtering structures.
    
    parameter : str
        The name of the parameter to filter structures by (e.g., 'Rg', 'tortuosity').
    
    length_list : list, optional
        A list of specific lengths to filter by. If empty, all lengths are included (default is an empty list).
    
    show_structures : bool, optional
        If True, displays visualizations of the unique structures (default is False).
    
    random_number : int, optional
        The number of structures to randomly select from those that meet the parameter criteria. 
        If 0, no random selection is made (default is 0).

    Returns:
    tuple :
        - structures (list) : A list of structures that meet the specified parameter range criteria.
        - structure_list (list) : A randomly selected subset of the structures (if `random_number` > 0).
    """
    
    # Initialize empty list and dictionary to store the structures and corresponding indices
    structures = []
    ind_dict = {}
    
    # If no specific lengths are provided, create a deep copy of the entire input DataFrame
    if len(length_list) == 0:
        analyzed_mainframe = copy.deepcopy(analyzed_mainframe0)
    else:
        # Filter the mainframe by the specified lengths in the length_list
        analyzed_mainframe = analyzed_mainframe0[analyzed_mainframe0['length'].isin(length_list)]

    # Iterate through the rows of the DataFrame
    for i in range(len(analyzed_mainframe)):
        sequence = analyzed_mainframe.index[i]  # Get the index (sequence) of the current row
        shape_data = analyzed_mainframe.loc[sequence, :]  # Retrieve the shape data for the current sequence
        vec = np.array(shape_data.loc[parameter])  # Extract the parameter values (as a vector) for comparison

        # Find the indices where the values of the parameter are within the specified range
        indices1 = np.array(np.where(vec <= max_val)[0])  # Indices where values are less than or equal to max_val
        indices2 = np.array(np.where(min_val <= vec)[0])  # Indices where values are greater than or equal to min_val
        indices = indices1[np.isin(indices1, indices2)]  # Intersection of the two index sets (within range)

        # If there are any indices within the specified range, append the structure and store indices
        if len(indices) != 0:
            structures.append(sequence)  # Add the sequence (structure) to the list
            ind_dict[str(sequence)] = indices  # Store the indices for this structure

            # If random_number is 0 (no random selection), show the unique structures for the current sequence
            if random_number == 0:
                shape_data = analyzed_mainframe.loc[sequence, :]
                hingenum = np.array(shape_data.loc['hinges'])
                ordered_centers = np.array(shape_data.loc['final hinges'])
                probs = np.array(shape_data.loc['probability'])
                high_centers = ordered_centers[indices, :]  # Extract the high probability centers
                high_probs = probs[indices]  # Extract the high probabilities
                high = vec[indices]  # Extract the values corresponding to the high indices
                if show_structures:
                    # Show the unique structures for visualization
                    show_unique_structures(hingenum, shapes, high_centers, sequence, high_probs, high, parameter)

    # If a random selection is required, choose 'random_number' structures randomly from the available list
    if random_number != 0:
        structure_list = np.random.choice(structures, random_number, replace=False)  # Randomly select structures
        for sequence in structure_list:
            shape_data = analyzed_mainframe.loc[sequence, :]
            shapes = shape_data.loc['shapes']
            hingenum = np.array(shape_data.loc['hinges'])
            ordered_centers = np.array(shape_data.loc['final hinges'])
            probs = np.array(shape_data.loc['probability'])
            indices = ind_dict[sequence]  # Get the indices for the current structure
            vec = np.array(shape_data.loc[parameter])
            high_centers = ordered_centers[indices, :]
            high_probs = probs[indices]
            high = vec[indices]
            if show_structures:
                # Show the unique structures for visualization
                show_unique_structures(hingenum, shapes, high_centers, sequence, high_probs, high, parameter)

    # Return both the full list of structures and the randomly selected structures (if applicable)
    return structures, structure_list

def sym_histogram(filtered_length):
    '''
    Function to generate a histogram of symmetry values from a filtered dataframe.
    
    Inputs:
        filtered_length : pandas.DataFrame
            Dataframe containing a 'symmetry' column with symmetry values for each data point.
    '''

    # Extract the 'symmetry' column from the dataframe
    frame = filtered_length.loc[:,'symmetry']

    # Initialize an empty numpy array to store the symmetry values
    sym_vec = np.zeros(0)

    # Loop through the dataframe and concatenate symmetry values into the sym_vec array
    for i in range(len(frame)):
        sym_vec = np.hstack((sym_vec, np.array(frame[i])))

    # Remove any NaN values from the symmetry array
    sym_vec = sym_vec[~np.isnan(sym_vec)]

    # Print the maximum value of the symmetry array for debugging purposes
    print(np.max(sym_vec))

    # Set up the figure for plotting the histogram
    plt.figure(figsize=(10,8))

    # Calculate the bin edges for the histogram using 'auto' binning
    bin_edges = np.histogram_bin_edges(sym_vec, bins='auto')

    # Plot the histogram with custom styling
    plt.hist(sym_vec, bins=bin_edges, edgecolor='black', color='#51c1bf', linewidth=0.5)

    # Set tick label size for both axes
    plt.tick_params(axis='both', which='major', labelsize=15)

    # Set the x-axis limit to focus on the range [0.7, 1]
    plt.xlim([0.7, 1])

    # Set the title of the plot
    plt.title('Symmetry')

    # Display the plot
    plt.show()

def T_histogram(filtered_length):
    '''
    Function to generate a histogram of Tortuosity (T) values from a filtered dataframe.
    
    Inputs:
        filtered_length : pandas.DataFrame
            Dataframe containing a 'T' column with tortuosity values for each data point.
    
    Returns:
        T_vec : numpy.ndarray
            Array of concatenated tortuosity values extracted from the dataframe.
    '''

    # Extract the 'T' column from the dataframe
    frame = filtered_length.loc[:,'T']

    # Initialize an empty numpy array to store the tortuosity values
    T_vec = np.zeros(0)

    # Loop through the dataframe and concatenate tortuosity values into the T_vec array
    for i in range(len(frame)):
        T_vec = np.hstack((T_vec, np.array(frame[i])))

    # Set up the figure for plotting the histogram
    plt.figure(figsize=(10,8))

    # Calculate the bin edges for the histogram using 'auto' binning
    bin_edges = np.histogram_bin_edges(T_vec, bins='auto')

    # Plot the histogram with custom styling
    n, bins, patches = plt.hist(T_vec, bins=bin_edges, edgecolor='black', color='#51c1bf', linewidth=0.5)

    # Set tick label size for both axes
    plt.tick_params(axis='both', which='major', labelsize=15)

    # Set the title of the plot
    plt.title('Tortuosity')

    # Display the plot
    plt.show()

    # Return the concatenated tortuosity values
    return T_vec

def find_closed_loops(analyzed_mainframe, show_structures=False):
    """
    Identifies sequences in the analyzed_mainframe that form closed loops.

    A closed loop is defined as a sequence with an end-to-end (e2e) distance less than 1.
    
    Parameters:
    - analyzed_mainframe: DataFrame containing structural data for various sequences, including end-to-end distances ('e2e').
    - show_structures (bool): If True, displays the structures that form closed loops (feature not implemented in this version).
    
    Returns:
    - structures: List of sequence names that form closed loops.
    """

    structures = []  # List to store sequences forming closed loops

    # Iterate over all sequences in the DataFrame
    for i in range(len(analyzed_mainframe)):
        # Generate sequence name (e.g., 'Sequence 1', 'Sequence 2', etc.)
        sequence = 'Sequence ' + str(i + 1)
        
        # Retrieve data for the current sequence
        shape_data = analyzed_mainframe.loc[sequence, :]
        
        # Extract the end-to-end (e2e) distances for the sequence
        vec = np.array(shape_data.loc['e2e'])
        
        # Identify indices where the e2e distance is less than 1 (closed loops)
        indices = np.array(np.where(vec < 1)[0])
        
        # If any closed loops are found, add the sequence to the list
        if len(indices) != 0:
            structures.append(sequence)
            
            # Optionally retrieve and process structural details for visualization
            shape_data = analyzed_mainframe.loc[sequence, :]
            shapes = shape_data.loc['shapes']
            hingenum = np.array(shape_data.loc['hinges'])
            ordered_centers = np.array(shape_data.loc['final hinges'])
            probs = np.array(shape_data.loc['probability'])
            high_centers = ordered_centers[indices, :]
            high_probs = probs[indices]
            high = vec[indices]
            # Note: The `show_structures` feature is defined but not utilized in this version.

    return structures