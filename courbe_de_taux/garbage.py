import pandas as pd 
import numpy as np 
import math as ma
from scipy.optimize import minimize,least_squares
import matplotlib.pyplot as plt
from datetime import date, datetime
from  sklearn import linear_model

import os
pre = os.path.dirname(os.path.realpath(__file__))
fname = 'taux_tresor_ex.xlsx'
path = os.path.join(pre, fname)
data = pd.read_excel(path)

data["days"]=(data["Date echeance"]-data["Date valeur"])
#convert days into int
data["days"]=data["days"].dt.total_seconds() /(3600*24)
data["days"]=data["days"].astype(int)



def predictionVasicek(self,t) :
        plt.plot(np.arange(1/365, t, 1/365), self.tauxZCVasicek(t), label="Prévision Vasicek" )
        plt.plot(np.arange(1/365, t, 1/365), self.setDonneeOptimisation(self.type,t=t), label="Taux ZC BAM" )
        #plt.plot(np.arange(1/365, t, 1/365), self.setDonneeOptimisation(self.type,t=t), label="Taux ZC BAM" )
        plt.xlabel("Maturité")
        plt.ylabel("Taux Zéro coupon")
        plt.legend()
        plt.show()

class CIRModel:
    def __init__(self, maturite, tauxcourt ):
        self.maturite=maturite
        self.taux=tauxcourt
       # self.time=(tauxcourt-tauxcourt[0]).dt.total_seconds().astype(int)/(3600*24)

    def plotData(self):
      plt.plot(self.maturite,self.taux, label="Donnée taux réel")
      plt.xlabel("Maturité")
      plt.ylabel("Taux court")
      plt.legend()
      plt.show()

    def setparameters(self):
       taux_1=np.array(self.taux[1::].dropna())
       Z= self.taux[:-1]/ np.sqrt(taux_1) 
       Y= np.sqrt(taux_1) 
       X=1/np.sqrt(taux_1) 
       dataregression=pd.DataFrame({"Y":Y,"X":X})
       reg=linear_model.LinearRegression(fit_intercept = False)
       reg.fit(dataregression,Z)
       sse=metrics.mean_squared_error(Z, reg.predict(dataregression))
       a=1- reg.coef_ [0]
       b=reg.coef_ [1]/a
       sigma=np.sqrt(sse)
       return a,b,sigma ,reg.coef_ [0], reg.coef_ [1]
    

    
    def backtesting(self):
       a=self.setparameters()[0]
       b=self.setparameters()[1]
       sigma=self.setparameters()[2]
       tauxCIR= np.zeros(len(self.maturite))
       tauxCIR[0]=self.taux[0]
       for i in range(1,len(tauxCIR)):
          tauxCIR[i]= a*(b-self.taux[i-1])+self.taux[i-1] #+ sigma*np.sqrt(tauxCIR[i-1])*np.random.normal()
       return tauxCIR
    
    def predTaux(self,m):
       a=self.setparameters()[0]
       b=self.setparameters()[1]
       sigma=self.setparameters()[2]
       tauxCIR= np.zeros(len(self.maturite))
       tauxCIR=self.taux[len(tauxCIR)-1]
       m=round(m*365)
       for _ in range(m):
          tauxCIR= a*(b-tauxCIR)+tauxCIR #+ sigma*np.sqrt(tauxCIR[i-1])*np.random.normal()
       return tauxCIR 
       
    def plotCIR(self):
        plt.plot(self.maturite, self.backtesting(), label="Taux prédit")
        
        plt.xlabel("Maturité")
        plt.ylabel("Taux court")
        plt.legend()
        self.plotData() 

    def ErrorModel(self,pi):
       a=self.setparameters()[0]
       b=self.setparameters()[1]
       sigma=self.setparameters()[2]
       nss=NSS()
       Pzc=np.zeros(len(nss._maturite))
       Rzc=np.zeros(len(nss._maturite))
       gamma=ma.sqrt((a)**2+2*sigma**2)

       for i in range(len(nss._maturite)):
         r=self.predTaux(nss._maturite[i])
         t=nss._maturite[len(nss._maturite)-1] -nss._maturite[i]
         A=((2*gamma*ma.exp(t*(gamma+a)/2))/((gamma+a)*(ma.exp(gamma*t)-1)+2*gamma))**(2*a*b/sigma**2)
         B=2*(ma.exp(gamma*t)-1)/((gamma+a )*(ma.exp(gamma*t)-1)+2*gamma)
         Pzc[i]=A*np.exp(-B*r)
         Rzc[i]=-ma.log(Pzc[i])/t
       return np.sum((Rzc-nss._taux)**2) # np.sum((Pzc-np.exp(-nss._taux*nss._maturite))**2) #

    def optimizationModel(self):
        op=minimize(self.ErrorModel,0)
        return op.x, op.fun, op.success
    
    def previsionCIR(self):
       pi= self.optimizationModel()[0]
       a=self.setparameters()[0]
       b=self.setparameters()[1]
       sigma=self.setparameters()[2]
       nss=NSS()
       Pzc=np.zeros(len(nss._maturite))
       Rzc=np.zeros(len(nss._maturite))
       gamma=ma.sqrt((a+b)**2+2*sigma**2)

       for i in range(len(nss._maturite)):
         r=self.predTaux(nss._maturite[i])
         t=nss._maturite[len(nss._maturite)-1] -nss._maturite[i]
         A=((2*gamma*ma.exp(t*(gamma+a +pi)/2))/((gamma+a +pi)*(ma.exp(gamma*t)-1)+2*gamma))**(2*a*b/sigma**2)
         B=2*(ma.exp(gamma*t)-1)/((gamma+a +pi)*(ma.exp(gamma*t)-1)+2*gamma)
         Pzc[i]=A*np.exp(-B*r)
         Rzc[i]=-ma.log(Pzc[i])/t
       return Pzc, Rzc
    
    def plotCIRPred(self):
        nss=NSS()
        plt.plot(nss._maturite, self.previsionCIR()[1], label="Taux prédit")

        plt.xlabel("Maturité")
        plt.ylabel("Taux court")
        plt.legend()
        nss.plotData()


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
        return ((1/sc)**(1/t))-1
    
    def contrainte(self,parms, i):
      return parms[1] + (parms[i + 4] - parms[i + 3]) * (-np.exp(-3 * parms[0] * self._maturite[i]))

    def parmsValue(self ):
        
        constraint = ({'type': 'eq', 'fun': lambda parms:parms[1]+4*parms[2]*np.exp(-parms[0]*self._maturite[0])+9*parms[3]*np.exp(-2*parms[0]*self._maturite[0])},
                      {"type": "eq", "fun": lambda parms: np.array([parms[1] + (parms[i + 3] - parms[i + 2]) * (-np.exp(-3 * parms[0] * self._maturite[i])) for i in range(1, len(self._maturite) - 1)])} )
        op= minimize(self.splineExponentiel, np.ones(len(self._maturite)+2)*0.01, constraints=constraint, tol=0.00001)
        return op.x, op.fun, op.success, op.message
    
    def plotsplineExponentiel(self):
      plt.plot(self._maturite, self.splineExponentielTotal(self._maturite), label="Interpolation spline cubique") 
      plt.xlabel("Maturité")
      plt.ylabel("Taux Zéro coupon")
      plt.legend()
      self.plotData() 




ntsp=SplineExponentiel()

print(ntsp.parmsValue())
ntsp.plotsplineExponentiel()

class SplineExponentiel(Donnee):

    def __init__(self):
        super().__init__()
        self.B_actualisation= 1/(1+self._taux)**self._maturite

    def  splineExponentiel(self, parms):
        a=self._maturite
        u=parms[0]
        t=np.exp(-u*a)
        sc=  parms[1] + parms[2]*t +parms[3]*t**2+ parms[4]*t**3 
        for i in  range(1,len(self._maturite)-1):
            c= np.exp(-u*self._maturite[i])
            sc=sc+(parms[i+4]-parms[i+3])*np.minimum((t-c),0)**3 
        error= np.sum((sc - self.B_actualisation)**2)  
        return error
    
    def  splineExponentielTotal(self,a):
        parms=self.parmsValue()[0]
        u=parms[0]
        t=np.exp(-u*a)
        sc=  parms[1] + parms[2]*t +parms[3]*t**2+ parms[4]*t**3 
        for i in  range(1,len(self._maturite)-1):
            c= np.exp(-u*self._maturite[i])
            sc=sc+(parms[i+4]-parms[i+3])*np.minimum((t-c),0)**3 
        return (1/sc)**(1/t)-1
    
    def contrainte(self,parms, i):
      return parms[1] + (parms[i + 4] - parms[i + 3]) * (-np.exp(-3 * parms[0] * self._maturite[i]))

    def parmsValue(self ):
        
        constraint = ( {'type': 'eq', 'fun': lambda parms:parms[1]+parms[2]+parms[3]+parms[4]-1}, 
                      {'type': 'eq', 'fun': lambda parms:parms[2]+4*parms[3]*np.exp(-parms[0]*self._maturite[0])+9*parms[4]*np.exp(-parms[0]*self._maturite[0])},
                      {"type": "eq", "fun": lambda parms: np.array([parms[1] + (parms[i + 4] - parms[i + 3]) * (-np.exp(-3 * parms[0] * self._maturite[i])) for i in range(1, len(self._maturite) - 1)])} )
        op= minimize(self.splineExponentiel, np.ones(len(self._maturite)+3)*0.01, constraints=constraint, tol=0.00001)
        return op.x, op.fun, op.success, op.message
    
    def plotsplineExponentiel(self):
      plt.plot(self._maturite, self.splineExponentielTotal(self._maturite), label="Interpolation spline cubique") 
      plt.xlabel("Maturité")
      plt.ylabel("Taux Zéro coupon")
      plt.legend()
      self.plotData() 




ntsp=SplineExponentiel()

print(ntsp.parmsValue())
ntsp.plotsplineExponentiel()






import os
pre = os.path.dirname(os.path.realpath(__file__))
fname = "tauxjour2123.xlsx"#'taux_tresor_ex.xlsx'
path = os.path.join(pre, fname)
data = pd.read_excel(path)
data["Days"]=(data["Date"]-data["Date"].iloc[0]).dt.total_seconds().astype(int)/(3600*24)

cir=CIRModel(data["Date"],data["Taux Moyen"])

#print(cir.predTaux(1/365))
print(cir.optimizationModel())
cir.plotCIRPred()
#plt.plot(cir.maturite, cir.backtesting())
#cir.plotData()


def setDonneeOptimisation(self,type,t):
        if(type=="NSS"):
           self.type=type     
           return NSS().NSS_Taux(np.arange(1/365, t, 1/365))
        if(type=="NS"):
            self.type=type   
            return NS().NS_taux(np.arange(1/365, t, 1/365))
        if(type=="spline cubique partielle"): 
            self.type=type 
            return SplineCubique().splineCubiquePartielle(np.arange(1/365, t, 1/365))[0]
        if(type=="spline cubique totale"): 
            self.type=type 
            return SplineCubique().splineCubiqueTotal(np.arange(1/365, t, 1/365))
        if(type=="Interpolation cubique"): 
            self.type=type 
            return InterpolationCubique().interpolationCubique(np.arange(1/365, t, 1/365))
        if(type=="Interpolation cubique Regression"): 
            self.type=type 
            return InterpolationCubique().regressionMultiple(np.arange(1/365, t, 1/365))
        if(type=="Interpolation simple"): 
            self.type=type 
            return InterpolationSimple().interpolationSimple(np.arange(1/365, t, 1/365))