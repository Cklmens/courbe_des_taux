import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si
import math as ma
from nelsonsiegelsvneson import NSS , NS
from splinecubique import SplineCubique
from interplolationpolynomial import InterpolationCubique,InterpolationSimple
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from charger_donnee import tauxzc


class VasicekModel:
    def __init__(self, maturite, tauxcourt ):
        self.maturite=maturite
        self.taux=tauxcourt
       # self.time=(tauxcourt-tauxcourt[0]).dt.total_seconds().astype(int)/(3600*24)

    def plotData(self):
      plt.plot(self.maturite, self.taux, label="Donnée taux réel")
      plt.xlabel("Maturité")
      plt.ylabel("Taux court")
      plt.legend()
      plt.show()
    def setParameters(self):
        mod=AutoReg(self.taux, lags=1)
        reg=mod.fit()
        sse=mean_squared_error(self.taux[1::], reg.predict().dropna())
        a=-ma.log(reg.params[1])
        return a, reg.params[0]/(1-reg.params[1]), np.sqrt(sse/(1-reg.params[1]**2)/2*a), reg ,reg.params[1], reg.params[0]

    def modeleVasicek(self):
        a=self.setParameters()[0]
        b=self.setParameters()[1]
        sigma=self.setParameters()[2]
        tauxVasicek=np.zeros(len(self.maturite))
        tauxVasicek[0]= self.taux[0]
        for i in range(1,len(self.maturite)):
            tauxVasicek[i]= self.taux[i-1]*ma.exp(-a) + b*(1-ma.exp(-a))
        return tauxVasicek

    def plotVasicek(self):
       
        plt.plot(self.maturite, self.modeleVasicek(), label="Taux prédit")
        plt.xlabel("Maturité")
        plt.ylabel("Taux court")
        plt.legend()
        plt.show()
        self.plotData()
     

    def ErrorModel(self , pi ,theta):
        a=self.setParameters()[0]
        b=self.setParameters()[1]
        sigma=self.setParameters()[2]
        Rinf= b-(pi/a)-(sigma**2)/(2*a**2)
        tz=tauxzc()
        Pzc=np.zeros(len(tz.temp))
        Rzc=np.zeros(len(tz.temp))
        for i in range(len(tz.temp)):
            r= tz.tauxzc[i]
            t=tz.temp[i] - theta
            Rzc[i]= Rinf -(Rinf -r)*((1-np.exp(-a*t))/(a*t))- ((sigma**2)*(1-np.exp(-a*t))**2)/(4*t*a**2)
            Pzc[i]=(np.exp(-Rzc[i]*tz.temp[i]))
        return np.sum((Rzc-tz.tauxzc)**2)# np.sum((Pzc-np.exp(-nss._taux*nss._maturite))**2) # 
    
    def optimizationModel(self, theta):
        op=minimize(self.ErrorModel,0, args=(theta))
        return op.x, op.fun, op.success
    
    def tauxZCVasicek(self,theta):
        pi= self.optimizationModel(theta)[0]
        a=self.setParameters()[0]
        b=self.setParameters()[1]
        sigma=self.setParameters()[2]
        Rinf= b-(pi/a)-(sigma**2/(2*a**2))
        tz=tauxzc()
        Pzc=np.zeros(len(tz.temp))
        Rzc=np.zeros(len(tz.temp))
        for i in range(len(tz.temp)):
            r= tz.tauxzc[i]
            t=tz.temp[i]-theta
            Rzc[i]= Rinf -(Rinf -r)*((1-np.exp(-a*t))/(a*t))- ((sigma**2)*(1-np.exp(-a*t))**2)/(4*t*a**2)
            Pzc[i]=(np.exp(-Rzc[i]*tz.temp[i]))
        return Rzc , np.sum((Rzc-tz.tauxzc)**2) 

    def predictionVasicek(self, theta) :
        tz=tauxzc()
        plt.plot(tz.temp, self.tauxZCVasicek(theta)[0], label="Prévision Vasicek" )
        plt.xlabel("Maturité")
        plt.ylabel("Taux Zéro coupon")
        plt.legend()
        NSS().plotData()

"""""
if __name__=="__main__":
#Importer les fichiers excel
    import os
    pre = os.path.dirname(os.path.realpath(__file__))
    fname = "tauxjour2123.xlsx"#'taux_tresor_ex.xlsx'
    path = os.path.join(pre, fname)
    data = pd.read_excel(path)
    data["Days"]=(data["Date"]-data["Date"].iloc[0]).dt.total_seconds().astype(int)/(3600*24)

    #Set up les parametres


    vas=VasicekModel(data["Date"],data["Taux Moyen"])
    #print(vas.setParameters())
    #plt.plot(vas.maturite,vas.modeleVasicek_())
    #vas.plotVasicek()

    plt.show()
    print(vas.predictionVasicek(5/365))
    #print(data)
    print(vas.optimizationModel(5/365))"""
