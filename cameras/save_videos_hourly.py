#!/bin/python
import ssl
import urllib.request
from datetime import datetime

ssl._create_default_https_context = ssl._create_unverified_context

def download(loc,save_path):
    url = 'http://e-nav.ccg-gcc.gc.ca/nvss-svsn/sequences/'+loc+'.mp4'
    filedata = urllib.request.urlopen(url)
    datatowrite = filedata.read()

    now = datetime.now()
    dt_string ='%(year)4i%(month)s%(day)s-%(hour)s%(minute)s'%{"year": now.year, "month":str(now.month).rjust(2, '0'), "day":str(now.day).rjust(2, '0'),"hour":str(now.hour).rjust(2, '0'),"minute":str(now.minute).rjust(2, '0')}
    fname = loc+'-'+dt_string+'.mp4'

    with open(save_path+fname,'wb') as f:
        f.write(datatowrite)


s_path = './'
download('Longueuil','./2022-2023/Longueuil/')
download('PontJacquesCartier','./2022-2023/PontJacquesCartier/')
download('IleCharron','./2022-2023/IleCharron/')

