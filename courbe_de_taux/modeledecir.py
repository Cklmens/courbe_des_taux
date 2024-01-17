import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si
import math as ma
from nelsonsiegelsvneson import NSS , NS
from splinecubique import SplineCubique
from interplolationpolynomial import InterpolationCubique,InterpolationSimple
from scipy.optimize import minimize
from sklearn import linear_model, metrics
from charger_donnee import tauxzc

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
       nss=tauxzc()
       Pzc=np.zeros(len(nss.temp))
       Rzc=np.zeros(len(nss.temp))
       gamma=ma.sqrt((a+b)**2+2*sigma**2)

       for i in range(len(nss.temp)):
         r=self.predTaux(nss.temp[i])
         t=nss.temp[len(nss.temp)-1] -nss.temp[i]
         A=((2*gamma*ma.exp(t*(gamma+a +pi)/2))/((gamma+a +pi)*(ma.exp(gamma*t)-1)+2*gamma))**(2*a*b/sigma**2)
         B=2*(ma.exp(gamma*t)-1)/((gamma+a +pi)*(ma.exp(gamma*t)-1)+2*gamma)
         Pzc[i]=A*np.exp(-B*r)
         Rzc[i]=-ma.log(Pzc[i])/t
       return np.sum((Rzc-nss._taux)**2)# np.sum((Pzc-np.exp(-nss._taux*nss._maturite))**2) #

    def optimizationModel(self):
        op=minimize(self.ErrorModel,0)
        return op.x, op.fun, op.success
    
    def previsionCIR(self):
       pi= self.optimizationModel()[0]
       a=self.setparameters()[0]
       b=self.setparameters()[1]
       sigma=self.setparameters()[2]
       nss=tauxzc()
       Pzc=np.zeros(len(nss.temp))
       Rzc=np.zeros(len(nss.temp))
       gamma=ma.sqrt((a+b)**2+2*sigma**2)

       for i in range(len(nss.temp)):
         r=self.predTaux(nss.temp[i])
         t=nss.temp[len(nss.temp)-1] -nss.temp[i]
         A=((2*gamma*ma.exp(t*(gamma+a +pi)/2))/((gamma+a +pi)*(ma.exp(gamma*t)-1)+2*gamma))**(2*a*b/sigma**2)
         B=2*(ma.exp(gamma*t)-1)/((gamma+a +pi)*(ma.exp(gamma*t)-1)+2*gamma)
         Pzc[i]=A*np.exp(-B*r)
         Rzc[i]=-ma.log(Pzc[i])/t
       return Pzc, Rzc
    
    def plotCIRPred(self):
        nss=tauxzc()
        plt.plot(nss.temp, self.previsionCIR()[1], label="Taux prédit")

        plt.xlabel("Maturité")
        plt.ylabel("Taux court")
        plt.legend()
        NSS().plotData()








"""import os
pre = os.path.dirname(os.path.realpath(__file__))
fname = "tauxjour2123.xlsx"#'taux_tresor_ex.xlsx'
path = os.path.join(pre, fname)
data = pd.read_excel(path)
data["Days"]=(data["Date"]-data["Date"].iloc[0]).dt.total_seconds().astype(int)/(3600*24)

cir=CIRModel(data["Date"],data["Taux Moyen"])

#print(cir.predTaux(1/365))
print(cir.predTaux(0))
#cir.plotCIRPred()
#plt.plot(cir.maturite, cir.backtesting())
#cir.plotData()"""