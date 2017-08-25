#Imports needed
import numpy as np
import scipy as scp
import sklearn as skp
import os
import sys
import csv
import itertools

#Import data
src = os.path.realpath(
    os.path.join(os.getcwd(),
    os.path.dirname('treated_data.csv')))
with open(src+'/treated_data.csv', newline='') as f:
    reader = csv.reader(f,delimiter='|')
    data = list(reader);
print(data)

