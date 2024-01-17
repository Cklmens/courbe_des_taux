import pandas as pd 
import numpy as np 
import math as ma
from scipy.optimize import minimize,minimize_scalar
import matplotlib.pyplot as plt
from datetime import date, datetime
from charger_donnee import Donnee
from  sklearn import linear_model

class NSS(Donnee):
    def __init__(self):
        super().__init__()
        
    
    #Nelson Siegel Svensson model
    def  NSS_taux(self, t):
      parms=self.parmsValue()[0]
      return parms[0]+parms[1]*((1-np.exp(-t/parms[4]))/(t/parms[4]))+parms[2]*(((1-np.exp(-t/parms[4]))/(t/parms[4]))-np.exp(-t/parms[4]))+parms[3]*(((1-np.exp(-t/parms[5]))/(t/parms[5]))-np.exp(-t/parms[5]))
    
    def NSS_model(self,parms):
        t=self._maturite
        taux = parms[0]+parms[1]*((1-np.exp(-t/parms[4]))/(t/parms[4]))+parms[2]*(((1-np.exp(-t/parms[4]))/(t/parms[4]))-np.exp(-t/parms[4]))+parms[3]*(((1-np.exp(-t/parms[5]))/(t/parms[5]))-np.exp(-t/parms[5]))
        error= np.sum((taux - self._taux)**2)  
        return error
    def parmsValue(self):
        constraint = ({'type': 'ineq', 'fun': lambda parms: parms[5] },
                       {'type': 'ineq', 'fun': lambda parms:parms[4]-1})
        op= minimize(self.NSS_model,np.ones(6)*0.1, constraints=constraint, tol=0.00001)
        return op.x , op.fun,  op.success, op.message
    
    def plotNSS(self):
      plt.plot(self._maturite,self.NSS_taux(self._maturite), label="Interpolation optimisation NSS")  
      plt.xlabel("Maturité")
      plt.ylabel("Taux Zéro coupon")
      plt.legend("Interpolation NSS")
      self.plotData()
 
    
    def NSSregession(self,tau):
        t=self._maturite
        r=(1-np.exp(-t/tau[0]))/(t/tau[0])
        r1= ((1-np.exp(-t/tau[0]))/(t/tau[0]))-np.exp(-t/tau[0])
        r2= ((1-np.exp(-t/tau[1]))/(t/tau[1]))-np.exp(-t/tau[1])
        Mat=pd.concat([r,r1,r2], axis=1, keys=["c","c1","c2"])
        b=self._taux
        reg= linear_model.LinearRegression()
        reg.fit(Mat,b)
        c=reg.coef_
        taux=reg.intercept_+c[0]*((1-np.exp(-t/tau[0]))/(t/tau[0]))+c[1]*(((1-np.exp(-t/tau[0]))/(t/tau[0]))-np.exp(-t/tau[0]))+c[2]*(((1-np.exp(-t/tau[1]))/(t/tau[1]))-np.exp(-t/tau[1]))
        error=np.sum((taux - self._taux)**2)
        return error #reg.score()
    def optimiser(self):
        constraint = ({'type': 'ineq', 'fun': lambda tau: tau[1] },
                       {'type': 'ineq', 'fun': lambda tau :tau[0]-1})
        op= minimize(self.NSSregession,[0.1,0.1], constraints=constraint, tol=0.00001)
        return op.x , op.fun, op.success, op.message
    
    def NSS_Taux(self, z):
        tau=self.optimiser()[0]
        t=self._maturite
        r=(1-np.exp(-t/tau[0]))/(t/tau[0])
        r1= ((1-np.exp(-t/tau[0]))/(t/tau[0]))-np.exp(-t/tau[0])
        r2= ((1-np.exp(-t/tau[1]))/(t/tau[1]))-np.exp(-t/tau[1])
        Mat=pd.concat([r,r1,r2], axis=1, keys=["c","c1","c2"])
        b=self._taux
        reg= linear_model.LinearRegression()
        reg.fit(Mat,b)
        c=reg.coef_
        return reg.intercept_+c[0]*((1-np.exp(-z/tau[0]))/(z/tau[0]))+c[1]*(((1-np.exp(-z/tau[0]))/(z/tau[0]))-np.exp(-z/tau[0]))+c[2]*(((1-np.exp(-z/tau[1]))/(z/tau[1]))-np.exp(-z/tau[1]))
    
    def plotNSSregression(self):
      plt.plot(self._maturite,self.NSS_Taux(self._maturite), label="Interpolation regression NSS")  
      plt.xlabel("Maturité")
      plt.ylabel("Taux Zéro coupon")
      plt.legend("Interpolation NSS")
      self.plotData()
   


    
class NS(Donnee):
    def __init__(self):
        super().__init__()
  
    
    #Nelson Siegel  model
    def  NS_taux(self,t): # parm[3]=taux
      parms=self.parmsValue()[0]
      return parms[0]+parms[1]*((1-np.exp(-t/parms[3]))/(t/parms[3]))+parms[2]*(((1-np.exp(-t/parms[4]))/(t/parms[3]))-np.exp(-t/parms[3]))
    
    def NS_model(self,parms):
        t=self._maturite
        taux = parms[0]+parms[1]*((1-np.exp(-t/parms[4]))/(t/parms[4]))+parms[2]*(((1-np.exp(-t/parms[4]))/(t/parms[4]))-np.exp(-t/parms[4]))
        error= np.sum((taux - self._taux)**2)  
        return error
    def parmsValue(self):
        constraint = ({'type': 'ineq', 'fun': lambda parms:parms[4]-1})
        op= minimize(self.NS_model,np.ones(5)*0.1, constraints=constraint, tol=0.00001)
        return op.x , op.fun,  op.success, op.message
    
    def NSregession(self,tau):
        t=self._maturite
        r=(1-np.exp(-t/tau))/(t/tau)
        r1= ((1-np.exp(-t/tau))/(t/tau))-np.exp(-t/tau)
        Mat=pd.concat([r,r1], axis=1, keys=["c","c1"])
        b=self._taux
        reg= linear_model.LinearRegression()
        reg.fit(Mat,b)
        c=reg.coef_
        taux=reg.intercept_+c[0]*((1-np.exp(-t/tau[0]))/(t/tau[0]))+c[1]*(((1-np.exp(-t/tau[0]))/(t/tau[0]))-np.exp(-t/tau[0]))
        error=np.sum((taux - self._taux)**2)
        return error 
    
    def NStauxRegression(self, z):
        tau=self.optimiser()[0]
        t=self._maturite
        r=(1-np.exp(-t/tau))/(t/tau)
        r1= ((1-np.exp(-t/tau))/(t/tau))-np.exp(-t/tau)
        Mat=pd.concat([r,r1], axis=1, keys=["c","c1"])
        b=self._taux
        reg= linear_model.LinearRegression()
        reg.fit(Mat,b)
        c=reg.coef_
        return reg.intercept_+c[0]*((1-np.exp(-z/tau[0]))/(z/tau[0]))+c[1]*(((1-np.exp(-z/tau[0]))/(z/tau[0]))-np.exp(-z/tau[0]))

    def optimiser(self):
        constraint = ({'type': 'ineq', 'fun': lambda tau : tau-1})
        op= minimize(self.NSregession, 0.1, constraints=constraint, tol=0.000001)
        return op.x , op.fun, op.success, op.message
    
    def plotNSoptimisation(self):
      plt.plot(self._maturite,self.NS_taux(self._maturite), label="Interpolation optimisation NS ")  
      plt.xlabel("Maturité")
      plt.ylabel("Taux Zéro coupon")
      plt.legend("Interpolation NS ")
      self.plotData()

    def plotNSregression(self):
      plt.plot(self._maturite,self.NStauxRegression(self._maturite), label="Interpolation Regression NS")  
      plt.xlabel("Maturité")
      plt.ylabel("Taux Zéro coupon")
      plt.legend("Interpolation NS")
      self.plotData()


class NSSparms(Donnee):
   def __init__(self):
      super().__init__()

   def plotNSSparms(self):
      l=0.67
      time=np.arange(0.1,15,0.1)
      lambda0=[(lambda x:1)(taux)  for taux in time]
      lambda1=[(lambda x:(1-np.exp(-x/l))/(x/l))(taux)  for taux in time]
      lambda2=[(lambda x:(1-np.exp(-x/l))/(x/l) -np.exp(-x/l))(taux)  for taux in time]
      plt.plot(time, lambda0, label="Beta 0")
      plt.plot(time, lambda1, label="Beta 1")
      plt.plot(time, lambda2, label="Beta 2")
      plt.xlabel("Maturité")
      plt.ylabel("Sensibilité de taux")
      plt.grid()
      plt.legend()
      plt.show()
   

#ntsp=NSSparms()
#print(ntsp.plotNSSparms())

