import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si
import math as ma
from tauxforward import TauxForward

class Swap():
    def __init__(self, nominal, tauxFixe,dateEval, dateDebut, dateFin,periodicite):
        self.nominal=nominal
        self.tauxFixe=tauxFixe
        self.dateEval=dateEval
        self.dateDebut=dateDebut
        self.dateFin=dateFin
        self.period=periodicite
    
    
