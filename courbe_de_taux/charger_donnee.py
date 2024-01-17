import pandas as pd 
import numpy as np 
import math as ma
from scipy.optimize import minimize,least_squares
import matplotlib.pyplot as plt
from datetime import date, datetime
from datascarping import LoadData

class  Donnee:  #abstract
    def __init__(self):
        data=LoadData().loadInterpolationData()
        data["Maturite"]=(data["EndDate"]-data["StartDate"])
        #convert days into int
        data["Maturite"]=(data["Maturite"].dt.total_seconds() /(3600*24)).astype(int)/365
        self.date=data["StartDate"][0]
        self._taux=data["Taux"]
        self._maturite=data["Maturite"]

    def plotData(self):
      plt.plot(self._maturite,self._taux,label="Donnée Taux Zéro coupon")
      plt.xlabel("Maturité")
      plt.ylabel("Taux Zéro coupon")
      plt.legend()
      """f = plt.figure() 
      f.set_figwidth(4) 
      f.set_figheight(4) """
      #plt.figure(figsize=(5,5))
      plt.show()
    
    def data(self):
       return LoadData().loadInterpolationData()

class  tauxzc:
   def __init__(self):
    import os
    pre = os.path.dirname(os.path.realpath(__file__))
    fname = "taux16_11.xlsx"#'taux_tresor_ex.xlsx'
    path = os.path.join(pre, fname)
    data = pd.read_excel(path)
    self.tauxzc= data["taux "]
    self.temp=data["days"]/365


    
  
