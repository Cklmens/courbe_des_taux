import pandas as pd 
import numpy as np 
import math as ma
from scipy.optimize import minimize,least_squares
import matplotlib.pyplot as plt
from datetime import date, datetime
from charger_donnee import Donnee
from  sklearn import linear_model

class InterpolationSimple(Donnee):
    def __init__(self):
       super().__init__()

    def plotInterpolationSimple(self):
    # methode de lagrange
      r=np.ones((2*len(self._maturite)-1,2))
      for i in range((len(self._maturite)-1)):
        r[2*i][0]=self._maturite[i]
        r[2*i+1][0]= (self._maturite[i]+self._maturite[i+1])/2
        r[2*i][1]=self._taux[i]
        r[2*i+1][1]= (self._taux[i+1]*((-self._maturite[i]+self._maturite[i+1])/2) 
                      -self._taux[i]*((self._maturite[i]-self._maturite[i+1])/2))/ (self._maturite[i+1]- self._maturite[i])
      r[2*len(self._maturite)-2][0]=self._maturite[len(self._maturite)-1]
      r[2*len(self._maturite)-2][1]=self._taux[len(self._maturite)-1]
      rt=r.T
      plt.plot(rt[0],rt[1], label="Interpolation simple")  
      
      plt.xlabel("Maturité")
      plt.ylabel("Taux Zéro coupon")
      self.plotData()

    def interpolationSimple(self,t):
         taux=[]
         for j in range(len(t)):
            if(t[j]< min(self._maturite)or t[j]>max(self._maturite)):
               j=j+1#"ERROR rate not in the range", 
            else:
               i =np.where((self._maturite-t[j])>0)[0][0]
               m=(self._taux[i]*(-t[j]+self._maturite[i])
                        +self._taux[i-1]*(t[j]-self._maturite[i-1]))/ (self._maturite[i]- self._maturite[i-1])
               taux.append(round(m,4))
         return np.array(taux)
    
    def interpolationsimple(self,t):
       
            if(t< min(self._maturite)or t>max(self._maturite)):
               "ERROR rate not in the range", 
            else:
               i =np.where((self._maturite-t)>0)[0][0]
               m=(self._taux[i]*(-t+self._maturite[i])
                        +self._taux[i-1]*(t-self._maturite[i-1]))/ (self._maturite[i]- self._maturite[i-1])
               return round(m,4)
         
    
    def plotInterpolationsimple(self):
      t=np.arange(self._maturite[0],self._maturite[len(self._maturite)-1], 1/360)
      plt.plot(t,self.interpolationSimple(t), label="Interpolation simple")  
       
      plt.xlabel("Maturité")
      plt.ylabel("Taux Zéro coupon")
      self.plotData()
       
class InterpolationCubique(Donnee):

    def __init__(self):
       super().__init__()

    def interpolationCubique(self,a):
       taux=[]
       for j in range(len(a)):
         t=a[j]
         if(t< np.min(self._maturite)or t>np.max(self._maturite)):
            j=j+1#return "ERROR rate not in the range",  np.min(self._maturite),max(self._maturite)
         else:
            i =np.where((self._maturite-t)>0)[0][0] #Recherche de lindice de la premier valeur supérieur à a
            B=[self._taux[i-2],self._taux[i-1],self._taux[i],self._taux[i+1]] #construction de la matrice 4x1 taux
            A=[self._maturite[i-2],self._maturite[i-1],self._maturite[i],self._maturite[i+1]] #construction de la matrice 4x1

            a=b=c=d=0
      # method de gauss seidel
            for i in range(100):
               a =( B[3]-b*A[3]**2-c*A[3]-d)/ ma.pow(A[3],3)
               b=( B[2]-a*ma.pow(A[2],3)-c*A[2]-d)/ A[2]**2
               c=( B[1]-b*A[1]**2-a*ma.pow(A[1],3)-d)/ A[1]
               d=( B[0]-ma.pow(A[0],3)*a -b*A[0]**2-c*A[0])
            taux.append(a*t**3 +b*t**2 +c*t+d)
       return taux
    
    def regressionMultipleCoef(self):
      r= self._maturite
      r2= self._maturite**2
      r3=self._maturite**3
      Mat=pd.concat([r,r2,r3], axis=1, keys=["t","t^2","t^3"])
      b=self._taux
      regress=linear_model.LinearRegression()
      regress.fit(Mat,b)
      return regress
    
    def regressionMultiple(self,t):
       parm=self.regressionMultipleCoef()
       return parm.intercept_ + parm.coef_[0]*t+parm.coef_[1]*t**2+parm.coef_[2]*t**3
    
    def regressionScore(self):
      r= self._maturite
      r2= self._maturite**2
      r3=self._maturite**3
      Mat=pd.concat([r,r2,r3], axis=1, keys=["t","t^2","t^3"])
      b=self._taux
      regress=linear_model.LinearRegression()
      regress.fit(Mat,b)
      return regress.score(Mat,b)
    
    
    
    def plotRegressionMultiple(self):
      predict= self.regressionMultiple(self._maturite)
      plt.plot(self._maturite,predict, label="Interpolation cubique par regression")  
      self.plotData()
      plt.xlabel("Maturité")
      plt.ylabel("Taux Zéro coupon")
      plt.legend()
    

    
    def plotInterpolationCubique(self):
      A=self._maturite
      B=self._taux
      #step=int((max(A)-min(A))/100)
      step=0.5
      #A is a Pandas
      rt=np.zeros((len(np.arange(A[0],A.iloc[-1]+step,step)),2))
      # method de gauss seidel
      n=0
      for u in np.arange(0,len(A)-3,3):
            a=b=c=d=0
            for _ in range(100):
                a =( B[u+3]-b*A[u+3]**2-c*A[u+3]-d)/ ma.pow(A[u+3],3)
                b=( B[u+2]-a*ma.pow(A[u+2],3)-c*A[u+2]-d)/ A[u+2]**2
                c=( B[u+1]-b*A[u+1]**2-a*ma.pow(A[u+1],3)-d)/ A[u+1]
                d=( B[u]-ma.pow(A[u],3)*a -b*A[u]**2-c*A[u])
        
            for j in np.arange(A[u],A[u+3],step):
                #np.append(np.zeros(len(np.arange(A[u],A[u+3],step))),rt)
                rt[n][0]=j
                rt[n][1]=a*ma.pow(j,3) +b*ma.pow(j,2) + c*j +d 
                n=n+1
      r=rt.T
      a=r[0][0:len(r[0])-1]
      b=r[1][0:len(r[0])-1]
      print(a,b)
      plt.plot(a,b, label="Interpolation cubique")  
      
      plt.xlabel("Maturité")
      plt.ylabel("Taux Zéro coupon")
      plt.legend()
      self.plotData()
 

"""ntsp=InterpolationSimple()
#ntsp.plotData()
#print(ntsp.interpolationSimple(np.arange(1/365, 1, 1/365)))
print(np.arange(ntsp._maturite[0], ntsp._maturite[len(ntsp._maturite)-1] , 1/365))
#print(ntsp.plotRegressionMultiple())"""
#print(InterpolationCubique().plotInterpolationCubique())