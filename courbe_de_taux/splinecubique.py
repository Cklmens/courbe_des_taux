import pandas as pd 
import numpy as np 
import math as ma
from scipy.optimize import minimize,least_squares,curve_fit
import matplotlib.pyplot as plt
from datetime import date, datetime
from charger_donnee import Donnee
from scipy.optimize import NonlinearConstraint

class SplineCubique(Donnee):
    def __init__(self):
        super().__init__()
        self.pseudomaturite=self._maturite[0:4]
        self.pseudotaux= self._taux[0:4]
        self.B_actualisation= 1/(1+self._taux)**self._maturite
    

    def  splineCubique0(self, parms):
        t=self._maturite
        sc= 1 + parms[0]*t +parms[1]*t**2+ parms[2]*t**3 
        for i in  range(1,len(self._maturite)-1):
            c=self._maturite[i]
            sc=sc+(parms[i+2]-parms[i+1])*np.maximum((t-c),0)**3 
        error= np.sum((sc - self.B_actualisation)**2)  
        return error
    

    def  splineCubiqueTotal(self,t):
        parms=self.parmsValue()[0]
        sc= 1 + parms[0]*t +parms[1]*t**2+ parms[2]*t**3 
        for i in  range(1,len(self._maturite)-1):
            c=self._maturite[i]
            sc=sc+(parms[i+2]-parms[i+1])*np.maximum((t-c),0)**3 
        return ((1/sc)**(1/t))-1    

    
    def regressionSpline(self, parms,t,taux_):
       
        r2= t**3- np.maximum((t-t[1]),0)**3  
        r3=  np.maximum((t-t[1]),0)**3-np.maximum((t-t[2]),0)**3
        r4= np.maximum((t-t[2]),0)**3
        taux=1+parms[0]*t+parms[1]*t**2+parms[2]*r2+parms[3]*r3+parms[4]*r4
        error= np.sum((taux - taux_)**2)  
        return error                                      
    def parmsValue(self ):#,n,tau
        #op= minimize(self.regressionSpline,np.ones(5)*-0.01, tol=0.00001, args=(n,tau))
        #constraint = ({'type': 'eq', 'fun': lambda parms:parms[1]+3*parms[2]*self._maturite[0]}) \, constraints=constraint 
        op= minimize(self.splineCubique0,np.ones(len(self._maturite)+1)*-0.001, tol=0.00001)
        return op.x, op.fun, op.success, op.message

    
    def plageDeB(self,a):
        maturite=self._maturite.to_numpy()
        for i in range(0, len(self._taux),3):
            if i + 4 <= len(self._maturite):
               if  (maturite[i]<=a and maturite[i+3]>=a ):
                 return self.B_actualisation[i:i+4]
            else:
                return self.B_actualisation[len(self._maturite)-4:len(self._maturite)]

    def plageDeMaturite(self,a):
        maturite=self._maturite.to_numpy()
        for i in range(0, len(self._maturite),3):
          if i + 4 <= len(self._maturite):
             if  (maturite[i]<=a and maturite[i+3]>=a ):
                return  self._maturite[i:i+4]
          else:
                return  self._maturite[len(self._maturite)-4:len(self._maturite)]


    def  splineCubiquePartielle(self,t):
        m=self.plageDeMaturite(t).to_numpy()
        tau=self.plageDeB(t).to_numpy()
        constrainte=({'type': 'eq', 'fun': lambda parms:parms[1]+3*parms[2]*m[0]} 
                     ,{'type': 'eq', 'fun': lambda parms:parms[1]+3*parms[2]*m[1] -3*(m[1]-m[2])*parms[3]+3*(m[3]-m[2])*parms[4]})
        op= minimize(self.regressionSpline,np.ones(5)*0.1 , tol=0.000001, args=(m,tau), constraints=constrainte)
        parms=op.x
        r2= t**3- np.maximum((t-m[1]),0)**3  
        r3=  np.maximum((t-m[1]),0)**3-np.maximum((t-m[2]),0)**3
        r4= np.maximum((t-m[2]),0)**3
        ba=1+parms[0]*t+parms[1]*t**2+parms[2]*r2+parms[3]*r3+ parms[4]*r4
        return  (1/ba)**(1/t)-1 , op.success, op.fun, op.x
     
    def plotsplineCubique(self):
      plt.plot(self._maturite,self.splineCubiqueTotal(self._maturite),label="Interpolation spline cubique") 
      plt.xlabel("Maturité")
      plt.ylabel("Taux Zéro coupon")
      plt.legend()
      self.plotData() 

    def plotsplineCubiqueStepbystep(self):
      r=[i for i in self._maturite]
      plt.plot(self._maturite,r,label="spline cubique Step by Step")
      self.plotData() 
      plt.xlabel("Maturité")
      plt.ylabel("Taux Zéro coupon")
      plt.legend("Interpolation spline cubique")

class SplineExponentiel(Donnee):

    def __init__(self):
        super().__init__()
        self.B_actualisation= 1/(1+self._taux)**self._maturite

    def  splineExponentiel(self, parms):
        a=self._maturite
        u=parms[0]
        t=np.exp(-u*a)
        sc=  1 + parms[1]*(t-1) +parms[2]*(t**2 -1)+ parms[3]*(t**3-1) 
        for i in  range(1,len(self._maturite)-1):
            c= np.exp(-u*self._maturite[i])
            sc=sc+(parms[i+3]-parms[i+2])*np.minimum((t-c),0)**3 
        error= np.sum((sc - self.B_actualisation)**2)  
        return error
    
    def  splineExponentielTotal(self,a):
        parms=self.parmsValue()[0]
        u=parms[0]
        t=np.exp(-u*a)
        sc=  1 + parms[1]*(t-1) +parms[2]*(t**2 -1)+ parms[3]*(t**3-1) 
        for i in  range(1,len(self._maturite)-1):
            c= np.exp(-u*self._maturite[i])
            sc=sc+(parms[i+3]-parms[i+2])*np.minimum((t-c),0)**3 
        return ((1/sc)**(1/a))-1
    
    def contrainte(self,parms, i):
      return parms[1] + (parms[i + 4] - parms[i + 3]) * (-np.exp(-3 * parms[0] * self._maturite[i]))

    def parmsValue(self ):
        op= minimize(self.splineExponentiel, np.ones(len(self._maturite)+2)*0.01, tol=0.00001)
        return op.x, op.fun, op.success, op.message
    
    def plotsplineExponentiel(self):
      plt.plot(self._maturite, self.splineExponentielTotal(self._maturite), label="Interpolation spline exponentiel") 
      plt.xlabel("Maturité")
      plt.ylabel("Taux Zéro coupon")
      plt.legend()
      self.plotData() 




"""ntsp=SplineExponentiel()

print(ntsp.parmsValue())
ntsp.plotsplineExponentiel()"""