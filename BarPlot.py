# author : Fumiaki Abe

# TODO : sort big number
import numpy as np
import matplotlib.pyplot as plt
import csv
import pprint
import sys

args = sys.argv

leftList = []
heightList = []
List = []

if len(sys.argv) < 2:
    print("add file name !")
    exit
else:
    filename = '/home/rel/Desktop/ContactLink_ex/' + args[1]
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            List.append([row[0],int(row[1])])

    sortsecond = lambda val: val[1]
    List.sort(key=sortsecond)
    for row in List:
        leftList.append(row[0])
        heightList.append(int(row[1]))
    plt.barh(leftList, heightList )
    plt.show()


