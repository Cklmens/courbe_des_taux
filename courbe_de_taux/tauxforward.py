import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si
import math as ma
from nelsonsiegelsvneson import NSS


class TauxForward():
    def __init__(self) -> None:
        pass
       
        
    def tauxZC(self,t):
        return np.exp(-NSS().NSS_Taux(t)*t)
    
    def tauxForward(self, dateEval, dateDebut, dateFin):
        return (self.tauxZC(dateDebut-dateEval)-self.tauxZC( dateFin-dateEval))/(
            self.tauxZC(dateFin-dateEval)*(dateFin-dateDebut)
        )
    
        