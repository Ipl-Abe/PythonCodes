# author : Fumiaki Abe

import numpy as np
import matplotlib.pyplot as plt
import csv
import pprint
import sys

args = sys.argv

leftList = []
heightList = []
leftList2 = []
heightList2 = []
List = []
List2 = []

if len(sys.argv) == 1:
    print("add file name !")
    exit
elif len(sys.argv) == 2:
    filename = '/home/rel/Desktop/ContactLink_ex/' + args[1]
    with open(filename) as f:
        reader = csv.reader(f,delimiter=",")
        header = next(reader)
        for row in reader:
            print(row[1])
            List.append([row[0],row[1]])

    for row in List:
        leftList.append(row[0])
        heightList.append(row[1])
    plt.plot(leftList, heightList )
    plt.show()

