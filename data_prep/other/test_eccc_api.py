#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:57:36 2022

@author: Amelie
"""
# 5cebf1e03d0f4a073c4bbdd7

import requests
import json
response_API = requests.get('https://api-iwls.dfo-mpo.gc.ca/')
print(response_API.status_code)