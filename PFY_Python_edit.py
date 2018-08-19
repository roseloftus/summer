#!/usr/bin/python
import numpy as np
import math
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import gridspec

from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import interp1d
from scipy.special import wofz #for voigt profile
from lmfit.lineshapes import pvoigt, lorentzian, gaussian
import math
import sys
import os.path

#user input for centroids and ranges to be used in ev
ev = np.zeros((1,2))
help = input("For instructions on how to use this program, type 'yes'." +
             "\nOtherwise, type 'no': ")
if help.lower() == 'yes': 
      print("The following loop will ask how many samples you have." +
      "\nThen you will enter the centroids and energy ranges for each sample." +
      "\nPress 'Enter' after each input to continue." +
      "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

#while loop in case user gives bad input
restart = 'back'
while restart == 'back':

  #beginning of for loop to find out how many samples will be input
  if help.lower() == 'yes':
            print("The first step is entering the number of samples" +
                  "\nYou have. This is the same as the number of peaks")
  times = int(input("How many samples will you input?: "))
  print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

  for t in range(times):

      #get centroid (ele) for each element
      if help.lower() == 'yes':
            print("Now to enter the centroid for each of these peaks." +
                  "\nThese values must be between 0 and 4000 eV")
      ele = float(input("Enter the centroid for the element: "))

      #get energy range (e_rng) for each element
      if help.lower() == 'yes':
            print("Now for the energy ranges." +
                  "\nThis is the range of error for each peak")
      e_rng = float(input("Enter the energy range for the element: "))

      #make sure user gives reasonable input for centroid
      if ele <= 0 or ele >= 4000:
        print("Centroid must be between 0 and 4000!")
        ele = float(input("Re-enter the centroid for the element: "))

      #append centroid and energy range to ev
      else:
        ev = np.append(ev, [[ele, e_rng]], axis=0)

        #print("The centroid is", ele, " And the energy range is", e_rng)

      print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" +
            "\nThe centroid is", ele, " And the energy range is", e_rng)
      restart = input("If either of these values is incorrect, type 'back'" +
                      "\nIf they are both correct, type any letter: ")

      print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
      #explanation for restarting loop
      if restart == 'back':
        print("Values are incorrect. Restart:")

      #explanation for continuing program
      else:
        print("Values are correct. Continue program")

    
#===============================================================
#
# Function which reads in a two column file with incident energy
# and the I_Zero data
# 
#===============================================================  
def ReadEI0(fname):
  ene,i0 = np.loadtxt(fname,unpack=True)
  return ene,i0
  
#===============================================================
#
# Function which reads in the SDD energy values from a single
# column file
# 
#===============================================================  
def ReadESDD(fname):
  ene = np.loadtxt(fname,unpack=True)
  return ene
  
#===============================================================
#
# Function which reads in the SDD matrix from a file
# 
#===============================================================  
def ReadSDD(fname):
  map = np.loadtxt(fname,unpack=True)
  return map
  
#===============================================================
#
# Function which takes the I_Zero data and the SDD matrix and
# normalizes the SDD data by I_Zero
# 
#===============================================================  
def NormI0(I0,MapSDD):
  for p in range(1,len(MapSDD)):
    MapSDD[p,:] = np.divide(MapSDD[p,:],I0)
  return MapSDD
  
#===============================================================
#
# Function which can be used to correct for a single data point
# blip which is caused by injection at REIXS beamline. Simply
# replaces the data points for that incident energy by the
# average of the adjoining points (won't work for first/last
# point of scan)
# 
#===============================================================  
def CorrectInjection(I0,MapSDD,idx):
  I0[idx] = (I0[idx-1]+I0[idx+1])/2
  MapSDD[:,idx] = np.add(MapSDD[:,idx+1],MapSDD[:,idx-1])/2
  return I0,MapSDD
  
#===============================================================
#
# Calibrates on the incident energy axis by an arbitrary value
# dx. Uses linear interpolation to do so
# 
#===============================================================    
def EShift(dx,inEin,inMapSDD):
  newE = inEin - dx
  tempMap = np.copy(inMapSDD)
  for p in range(1,len(inMapSDD)):
    tempMap[p,:] = np.interp(newE,inEin,inMapSDD[p,:])
    #Scipy doesn't work for x outside range b/c my version is too old
    #but I recall having issues with installation for newer scipy, so using numpy linear interp instead
    #f = interp1d(inEin,inMapSDD[p,:],bounds_error=False,fill_value=(inMapSDD[p,0],inMapSDD[p,-1]))
    #inMapSDD[p,:] = f(newE)
  return tempMap
  
#===============================================================
#
#
# 
#===============================================================  
def LoadData(ELow, EHigh):
  #Scan 8, 25, 26, 27
  #27 needs correction for injection at point 81 (80 in zero based notation)
  ESDD = ReadESDD("/Users/roseloftus/Downloads/PFY_Python/Data/ESDD.dat")

  Ein1,I01 = ReadEI0("/Users/roseloftus/Downloads/PFY_Python/Data/S008_EI0.dat")
  MapSDD1 = ReadSDD("/Users/roseloftus/Downloads/PFY_Python/Data/S008_SDD.dat")
  MapSDD1 = NormI0(I01,MapSDD1)
  MapSDD1 = EShift(0.2,Ein1,MapSDD1) #calibration

  Ein2,I02 = ReadEI0("/Users/roseloftus/Downloads/PFY_Python/Data/S025_EI0.dat")
  MapSDD2 = ReadSDD("/Users/roseloftus/Downloads/PFY_Python/Data/S025_SDD.dat")
  MapSDD2 = NormI0(I02,MapSDD2)
  MapSDD2 = EShift(-0.15,Ein2,MapSDD2) #calibration

  Ein3,I03 = ReadEI0("/Users/roseloftus/Downloads/PFY_Python/Data/S026_EI0.dat")
  MapSDD3 = ReadSDD("/Users/roseloftus/Downloads/PFY_Python/Data/S026_SDD.dat")
  MapSDD3 = NormI0(I03,MapSDD3)
  MapSDD3 = EShift(-0.15,Ein3,MapSDD3) #calibration

  Ein4,I04 = ReadEI0("/Users/roseloftus/Downloads/PFY_Python/Data/S027_EI0.dat")
  MapSDD4 = ReadSDD("/Users/roseloftus/Downloads/PFY_Python/Data/S027_SDD.dat")
  I04,MapSDD4 = CorrectInjection(I04,MapSDD4,81-1)
  MapSDD4 = NormI0(I04,MapSDD4)
  MapSDD4 = EShift(0.2,Ein4,MapSDD4) #calibration

  Ein5,I05 = ReadEI0("/Users/roseloftus/Downloads/PFY_Python/Data/S055_EI0.dat")
  MapSDD5 = ReadSDD("/Users/roseloftus/Downloads/PFY_Python/Data/S055_SDD.dat")
  I05,MapSDD5 = CorrectInjection(I05,MapSDD5,196-1)  
  I05,MapSDD5 = CorrectInjection(I05,MapSDD5,357-1)  
  MapSDD5 = NormI0(I05,MapSDD5)

  Ein6,I06 = ReadEI0("/Users/roseloftus/Downloads/PFY_Python/Data/S056_EI0.dat")
  MapSDD6 = ReadSDD("/Users/roseloftus/Downloads/PFY_Python/Data/S056_SDD.dat")
  MapSDD6 = NormI0(I06,MapSDD6)
  MapSDD6 = EShift(0.06,Ein6,MapSDD6) #calibration
  
  
  MapSDD = np.add(MapSDD1,MapSDD2)
  MapSDD = np.add(MapSDD3,MapSDD)
  MapSDD = np.add(MapSDD4,MapSDD)
  MapSDD = np.add(MapSDD5,MapSDD)
  MapSDD = np.add(MapSDD6,MapSDD)/6

  #Now remove data where fluorescence is outside range
  while(ESDD[-1] > EHigh):
    #print(ESDD[-1])
    ESDD = np.delete(ESDD,-1,0)
    MapSDD = np.delete(MapSDD,-1,0)
  
  while(ESDD[0] < ELow):
    #print(ESDD[0])
    ESDD = np.delete(ESDD,0,0)
    MapSDD = np.delete(MapSDD,0,0)

    
  return Ein1,ESDD,MapSDD

#===============================================================
#
# A function which will integrate the SDD data within a certain
# energy window and return the resulting XAS spectrum (just
# intensity, not energy)
#
#===============================================================   
def EWindow(ESDD,MapSDD,ELow,EHigh):
  spec = np.zeros(len(MapSDD[1,:]))
  for p in range(1,len(MapSDD)):
    if(ESDD[p] < EHigh and ESDD[p] > ELow):
      spec = np.add(spec,MapSDD[p,:])
  return spec
  
#===============================================================
#
# A function like the previous one, but which first subtracts
# a straight line fitted on the fluorescence spectrum between
# the two endpoints of the window
#
#=============================================================== 
def EWindow2(ESDD,MapSDD,ELow,EHigh):
  spec = np.zeros(len(MapSDD[1,:]))
  #get low index
  idxL = np.searchsorted(ESDD,ELow,side="left")
  myX1 = ESDD[idxL]
  #get high index
  idxH = np.searchsorted(ESDD,EHigh,side="right")
  if(idxH >= len(ESDD)):
    idxH = len(ESDD)-1
    
  myX2 = ESDD[idxH]
  
  mydX = myX2 - myX1
  
  spec = np.zeros_like(MapSDD[0,:])
  
  for p in range(len(spec)): #for all incident energies
    myY1 = np.sum(MapSDD[idxL-1:idxL+2,p])/3
    myY2 = np.sum(MapSDD[idxH-1:idxH+2,p])/3
    myM = (myY2-myY1)/mydX
    myB = myY1 - myM*myX1
    spec[p] = np.sum(np.subtract(MapSDD[idxL:idxH+1,p],myM*ESDD[idxL:idxH+1]+myB))

  return spec
  
#===============================================================
#
# A simple smoothing function
# 
#===============================================================  
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
    
#===============================================================
#
# A voigt function used when fitting to the fluorescence spectra
# 
#===============================================================  
def voigt(x,G,L):
  #returns voigt with Lorentzian fwhm L and gaussian fwhm G,
  #centered at x

  sigma = G*0.424660900144
  gamma = L/2
  return np.real(wofz((x+1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)


def mypvoigt(x,amp,center,fwhm,frac):
  sig = fwhm*0.424660900144
  return ((1-frac)*gaussian(x, amp, center, sig) + frac*lorentzian(x, amp, center, sig))
  
#===============================================================
# 
# The code for the fitting starts now.  This function takes in 
# current fit parameters and returns the residual fluorescence
# data, which is the error between the current fit and the
# SDD data, summed over incident energy
# 
#===============================================================  
def ResFunctionLS(pars,inESDD,inSDD,NAmp):


  #The first two parameters in the parameter array are the 
  #Gaussian and Lorentzian widths for the peaks
  #Note that when using the pvoigt function, these aren't the actual
  #gaussian and lorentzian fwhm.  This would only be the case for
  #the true voigt function (also provided here, but slower)
  wG = pars[0]
  wL = pars[1]
  
  #The energy positions of the peaks are retrieved from another
  #function.  
  ev = GetEPeaks()

  #The starting energy positions are now shifted by the shifts
  #of the peaks which are being fitted (not necessarily all peaks)
  for p in range(len(pars)-2):
    ev[p,0] += pars[p+2]

  print(pars)
    
  #now fit according to amplitudes with the energy parameters fixed
  #defining this function within the other one so that it can access
  #the energies from the outer fitting loop
  def ResFunctionLSB(parsB,inESDDB,inSDDB):
    aB = np.zeros_like(inESDDB)
    for p in range(len(parsB)-1):
      #aB = aB + parsB[p]*voigt((inESDDB-ev[p,0]),wG,wL)
      aB += mypvoigt(inESDDB,parsB[p],ev[p,0],wG,wL/(wG+wL))
    aB += parsB[-1]*np.exp(-(inESDDB-bgndCen)**2/(bgndWid2)) #last parameter is background  
    #print(parsB)
    return (aB - inSDDB)  

  #Initialize the list of peak amplitudes that we will be fitting.
  x02 = np.zeros(NAmp+1) + 1e10 #extra amplitude is the background
  #Put very wide limits on the peak amplitudes (i.e. only constraint is they should be positive)
  lowLim = np.zeros(len(x02))
  highLim = np.zeros(len(x02)) + np.inf
  mb2 = (lowLim,highLim)

  #this will store the peak amplitudes for each incident energy (essentially the PFY spectra)
  specs = np.zeros((len(x02),len(inSDD[0,:])))

  #this will store the summed residuals along the fluorescence axis
  myRes = np.zeros_like(inESDD)
  
  
  #now fit the amplitudes according to the current peak energies, and return the total residual
  for p in range(len(inSDD[0,:])):
    ma2=(inESDD,inSDD[:,p])
    if(p==-1):
      res = least_squares(ResFunctionLSB,x02,method='trf',args=ma2,verbose=2)#bounds=mb2,
    else:
      res = least_squares(ResFunctionLSB,x02,method='trf',args=ma2)#bounds=mb2,
    specs[:,p] = np.copy(res.x)
    x02 = np.copy(res.x) #use previous as starting point for next
    myRes = myRes + res.fun

  print(pars)
  np.savetxt("/Users/roseloftus/Downloads/PFY_Python/Data/ampPars.txt",specs)

  return myRes/(1e10)


def FitMapLS(inEin,inESDD,inMapSDD,NdE,NAmp):
  

  x0 = np.zeros(NdE+2) #gauss width, lorentzian width,  and NdE dE  params
  x0[0] = 80
  x0[1] = 4
  x0[2:] = 0

  #print(np.shape(x0))
  
  ev = GetEPeaks()
  
  lowLim = np.zeros(len(x0))
  lowLim[0] = 20
  lowLim[1] = 1
  lowLim[2:] = -ev[0:NdE,1] #-20
  
  highLim = np.zeros(len(x0))
  highLim[0] = 100
  highLim[1] = 100
  highLim[2:] = ev[0:NdE,1]#20
  
  diff = np.zeros(NdE+2) + 0.0005 #1.0
  diff[0] = 0.005 #width needs different scale
  diff[1] = 0.005
  
  mb=(lowLim,highLim)
  ma = (inESDD,inMapSDD,NAmp) #these are passed to the residual function for the fitting

  #print(np.shape(x0))
  #exit()
  
  res = least_squares(ResFunctionLS,x0,bounds=mb,diff_step=diff,args=ma,verbose=2)
  ampPars = np.loadtxt("/Users/roseloftus/Downloads/PFY_Python/Data/ampPars.txt")
  
  return ampPars,res.x
#===============================================================
#
#
# 
#===============================================================  
def GetEPeaks():
  #each peak has a center and a dE for +/- allowable shift
 # ev = np.zeros((1,2)) #np.zeros(13)

  
  return ev


  """ev[0,0] = 528; ev[0,1] = 12.0          #Oxygen
  ev = np.append(ev,[[642.5,10.0]],axis=0)   #La 4p
  ev = np.append(ev,[[835.2,10.0]],axis=0)   #Sm 4p / La 3d
  ev = np.append(ev,[[860,10.0]],axis=0)   #Sm 4p / La 3d
  ev = np.append(ev,[[954,10.0]],axis=0)   #Cu ?
  ev = np.append(ev,[[1065,15.0]],axis=0)  #Sm M5
  ev = np.append(ev,[[1096,15.0]],axis=0)  #Sm M4
  return ev"""
  
#===============================================================
#
#
# 
#===============================================================  
def CalcMapLS(inAmps,inEner,inESDD,inMap):
  calcMap = np.zeros((len(inESDD),len(inAmps[1,:])))
  
  wG = inEner[0]  #gaussian fwhm

  wL = inEner[1] #lorentzian fwhm
  
  #wL = inEner[1]/2.35482
  #wLsq = 2*wL**2 #precalculating for speedup

  
  ev = GetEPeaks()

  for p in range(len(inEner)-2):
    ev[p,0] = ev[p,0] + inEner[p+2]  
  

  
  for p in range(len(inAmps[1,:])):
    for q in range(len(inAmps[:,1])-1): #The -1 accounts for the background being the last one 
      calcMap[:,p] = calcMap[:,p] + mypvoigt(inESDD,inAmps[q,p],ev[q,0],wG,wL/(wG+wL))
      #calcMap[:,p] = calcMap[:,p] + inAmps[q,p]*voigt(inESDD-ev[q,0],wG,wL)
    calcMap[:,p] = calcMap[:,p] + inAmps[-1,p]*np.exp(-(inESDD-bgndCen)**2/(bgndWid2))
    

    
  return calcMap
  
#===============================================================
# 
#
# 
#===============================================================  
def SliderPlot():

  inEin,inESDD,SDDMap,SDDCalcMap = GetSDDMaps()
  
  gs = gridspec.GridSpec(5, 1)
  
  axA1 = plt.subplot(gs[0:4, 0])
  l1, = plt.plot(inESDD, SDDMap[:,1], lw=2, color='red')
  m1, = plt.plot(inESDD, SDDCalcMap[:,1], lw=2, color='blue')
  axB1 = plt.subplot(gs[4, 0])
  slider1 = Slider(axB1, 'ECut', inEin[0], inEin[-1], valinit=inEin[0])  

  def updateP1(val):
    inEin,inESDD,SDDMap,SDDCalcMap = GetSDDMaps()
    idx = (np.abs(inEin-val)).argmin() #get index of inEin
    
    l1.set_ydata(SDDMap[:,idx])
    m1.set_ydata(SDDCalcMap[:,idx])
    #fig.canvas.draw_idle()
    axA1.set_ylim(min(0,np.min(SDDCalcMap[:,idx])),max(np.max(SDDMap[:,idx]),np.max(SDDCalcMap[:,idx])))

  slider1.on_changed(updateP1)

  plt.tight_layout()#pad=0.4, w_pad=0.5, h_pad=1.0)
  
  plt.show()  
#===============================================================
# 
#
# 
#===============================================================  
def GetSDDMaps():
    return Ein,ESDD,MapSDD,FitMap

    
#===============================================================
# 
#
#
#
#
#                     START OF MAIN PROGRAM
#
#
#
#
# 
#===============================================================  
#I also fit a "background", which is a broad peak centered at a
#particular energy.  Here you can set these two parameters (and
#then the background intensity is fitted). Can easily be changed
#to a constant background for the SDD detector, but usually it has
#some sort of peak structure due to pileup
bgndCen = 850 #centered at 850
bgndWid = 800 #width of background
bgndWid2 = 2*bgndWid*bgndWid/2.35482/2.35482 #precomputing for gaussian function


   
#Energy range to trim the fluorescence data to (to speed up fitting)
ELow = 300 
EHigh = 1300

NdE = 0 #number of peaks to allow the energy to fit
NAmp = times #number of peaks to include in the fitting (should be >= NdE)

if(NAmp < NdE):
  print("NAmp of ", NAmp, " should be >= NdE of ", NdE)
  exit()
  
#usage: python PFY_Python 

#Loading the data files (including calibrating/normalizing/summing)
Ein, ESDD, MapSDD = LoadData(ELow,EHigh)

#Example of extracting XAS via an integration windowc 
testspec = EWindow2(ESDD,MapSDD,510,560)#600,695)
np.savetxt("/Users/roseloftus/Downloads/PFY_Python/Results/EWindow_XAS.dat",testspec)

#Now fitting the data
Fits,dE = FitMapLS(Ein,ESDD,MapSDD,NdE,NAmp)

#Calculate a map using the fitted peak parameters
FitMap = CalcMapLS(Fits,dE,ESDD,MapSDD)

#Calculate a difference map between the experiment and fit
DiffMap = np.subtract(MapSDD,FitMap)

#Sum of the squares of the difference map to represent the final error in the fit  
myError = np.sum(np.divide(np.square(DiffMap),1e-10+np.square(MapSDD)))/MapSDD.size;
print("Error in fit: ", myError)

#Save results to file
np.savetxt("/Users/roseloftus/Downloads/PFY_Python/Results/Fits.dat",np.hstack((Ein[:,None],np.transpose(Fits))))
np.savetxt("/Users/roseloftus/Downloads/PFY_Python/Results/dE.dat", dE)
np.savetxt("/Users/roseloftus/Downloads/PFY_Python/Results/FitMap.dat",FitMap)


#Plot results
SliderPlot()

