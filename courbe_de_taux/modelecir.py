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
from statsmodels.tsa.ar_model import AutoReg
from charger_donnee import tauxzc

class CIRModel:
    def __init__(self, maturite, tauxcourt ):
        self.maturite=maturite
        self.taux=tauxcourt
        self.tauxcentre=tauxcourt -np.mean(tauxcourt)
       # self.time=(tauxcourt-tauxcourt[0]).dt.total_seconds().astype(int)/(3600*24)

    def plotData(self):
      plt.plot(self.maturite,self.taux, label="Donnée taux réel")
      plt.xlabel("Maturité")
      plt.ylabel("Taux court")
      plt.legend()
      plt.show()

    def setparameters(self):
       b=np.mean(self.taux)
       phi=self.optimizationRSS()[0]
       a=-np.log(phi)
       sig=self.optimizationRSS()[1]/(len(self.taux)-1)
       sigma=np.sqrt((2*a*sig)/(1-phi**2))
       return a , b, sigma

    def calculSigma(self):
       a=self.setparameters()[0]
       b=self.setparameters()[1]
       taux_1=np.array(self.taux[1::].dropna())
       Z= np.array(self.taux[:-1])/ np.sqrt(taux_1) 
       Y= np.sqrt(taux_1) 
       X=1/np.sqrt(taux_1)

       return  np.sum((Z-(1-a)*Y-a*b*X)**2)/ len(taux_1)

    def ErrorRSS(self, phi):
       r=np.array(self.tauxcentre[1::])
       tab=((r-phi*np.array(self.tauxcentre[:len(r)]))**2 )/ (r+np.mean(self.taux))
       return  np.sum(tab)

    def optimizationRSS(self):
        op=minimize(self.ErrorRSS,0)
        return op.x, op.fun, op.success  
    

    
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
       r_1=self.taux.iloc[-1]
       np.random.seed(5)
       m=round(m*365)
       for _ in range(m):
         r_1=  r_1+  a*(b-r_1)/365 + sigma*np.sqrt(r_1/365)*np.random.normal()
      
       return r_1
    
    def previsionCIR(self, theta):
       a=self.setparameters()[0]
       b=self.setparameters()[1]
       sigma=self.setparameters()[2]
       tz=tauxzc()
       Pzc=np.zeros(len(tz.temp))
       Rzc=np.zeros(len(tz.temp))
       gamma=ma.sqrt((a)**2+2*sigma**2)

       for i in range(len(tz.temp)):
         r=tz.tauxzc[i] 
         t=  tz.temp[i]-theta #nss._maturite[len(nss._maturite)-1] -nss._maturite[i]
         A=((2*gamma*ma.exp(t*(gamma+a )/2))/((gamma+a )*(ma.exp(gamma*t)-1)+2*gamma))**(2*a*b/sigma**2)
         B=2*(ma.exp(gamma*t)-1)/((gamma+a )*(ma.exp(gamma*t)-1)+2*gamma)
         Pzc[i]=A*np.exp(-B*r)
         Rzc[i]=-ma.log(Pzc[i])/t
       return Pzc, Rzc

    def plotCIR(self):
        plt.plot(self.maturite, self.backtesting(), label="Taux prédit CIR") 
        plt.xlabel("Maturité")
        plt.ylabel("Taux court")
        plt.legend()
        self.plotData()


    
    def plotCIRPred(self, theta):
        tz=tauxzc()
        plt.plot(tz.temp, self.previsionCIR(theta)[1], label="Taux Zéro coupon CIR")

        plt.xlabel("Maturité")
        plt.ylabel("Taux court")
        plt.legend()
        NSS().plotData()

"""""
import os
pre = os.path.dirname(os.path.realpath(__file__))
fname = "tauxjour2123.xlsx"#'taux_tresor_ex.xlsx'
path = os.path.join(pre, fname)
data = pd.read_excel(path)
data["Days"]=(data["Date"]-data["Date"].iloc[0]).dt.total_seconds().astype(int)/(3600*24)

cir=CIRModel(data["Date"],data["Taux Moyen"])

#print(cir.plotCIR())
#print(cir.setparameters())
print(cir.plotData())

#plt.plot(cir.maturite, cir.backtesting())
#cir.plotData()"""