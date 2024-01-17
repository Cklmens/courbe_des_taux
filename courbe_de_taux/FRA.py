import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si
import math as ma
from tauxforward import TauxForward

class FRA():
    def __init__(self, nominal, tauxFixe,dateEval, dateDebut, dateFin):
        self.nominal=nominal
        self.tauxFixe=tauxFixe
        self.dateEval=dateEval
        self.dateDebut=dateDebut
        self.dateFin=dateFin

    def valeurFRA(self):
        return self.nominal*(self.dateFin-self.dateDebut)*TauxForward().tauxZC(self.dateFin-self.dateEval)*(self.tauxFixe
                                -TauxForward().tauxForward(self.dateEval,self.dateDebut, self.dateFin))
    
    
    

    