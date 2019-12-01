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

elif len(sys.argv) == 3:
    filename1 = '/home/rel/Desktop/ContactLink_ex/' + args[1]
    filename2 = '/home/rel/Desktop/ContactLink_ex/' + args[2]
    # file1
    with open(filename1) as f:
        reader = csv.reader(f)
        for row in reader:
            List.append([row[0],int(row[1])])
    for row in List:
        leftList.append(row[0])
        heightList.append(int(row[1]))
    # file2
    with open(filename2) as f2:
        reader2 = csv.reader(f2)
        for row in reader2:
            List2.append([row[0],int(row[1])])
    for row in List2:
        leftList2.append(row[0])
        heightList2.append(int(row[1]))

    left = np.arange(len(heightList))
    height = 0.3

    plt.barh(left, heightList, height=height, align='center')
    plt.barh(left+height, heightList2, height=height, align='center')
    plt.yticks(left + height/2, leftList)


    plt.show()
