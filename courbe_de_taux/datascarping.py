import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date, datetime

class LoadData:

  def loadInterpolationData(self): 
    url="https://www.bkam.ma/Marches/Principaux-indicateurs/Marche-obligataire/Marche-des-bons-de-tresor/Marche-secondaire/Taux-de-reference-des-bons-du-tresor"

    reponse=requests.get(url)

    if reponse.ok:
      soup=BeautifulSoup(reponse.text, "lxml")
      trs=soup.findAll("tbody")
      date1=[]
      date2=[]
      taux=[]
      dta=[]
      for tr in trs:
        tds= tr.findAll("tr")
        i=1
        for td in tds:
          tdas=td.findAll("td")
          for tda in tdas:
            if(tda.text!="\n" and tda.text!="-"):
                dta.append(tda.text)
      for i in range(0,len(dta)-2,3):
        date1.append(datetime.strptime(dta[i],"%d/%m/%Y"))
        taux.append(float(dta[i+1].replace(',', '').replace('%', ''))/100000)
        date2.append(datetime.strptime(dta[i+2],"%d/%m/%Y"))

      data= pd.DataFrame({"EndDate":date1,"Taux":taux,"StartDate":date2 })
      return data
   
  def loadModellingData(self):
    url="https://www.bkam.ma/Marches/Principaux-indicateurs/Marche-monetaire/Marche-monetaire-interbancaire"
    reponse=requests.get(url)

    if reponse.ok:
      soup=BeautifulSoup(reponse.text, "lxml")
      trs=soup.findAll("tbody")
      date1=[]
      date2=[]
      taux=[]
      dta=[]
      for tr in trs:
        tds= tr.findAll("tr")
        i=1
        for td in tds:
          tdas=td.findAll("td")
          for tda in tdas:
            if(tda.text!="\n" and tda.text!="-"):
                dta.append(tda.text)
          
        return 0

#dat=LoadData()
#print(dat.loadInterpolationData())

