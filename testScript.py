import os
from cnoid.Util import *
from cnoid.Base import *
from cnoid.BodyPlugin import *
from time import sleep
import time
import sys
import numpy as np


#def position_update:



itv = ItemTreeView.instance
pm = ProjectManager.instance

world = WorldItem()
world.name = "world"
RootItem.instance.addChildItem(world)


aistSim = AISTSimulatorItem()
world.addChildItem(aistSim)

bodyfile = "./Tank.body"
#os.path.join()

# Load floor
floor = BodyItem()
floor.load(floor,"/home/rel/choreonoid/share/model/misc/floor.body")
world.addChildItem(floor)
itv.checkItem(floor,True)

# Load Tank 
model = os.path.join("/home/rel/choreonoid/share/model/Tank","Tank.body")
body = BodyItem()
body.load(model)
bodyLink = body.body.link(0)
#LinkMat = np.array([[1.0],[0.0],[0.106]])
#LinkMat = [1,2,1]
#LinkMat = np.array([[1.0,0.0,0.0,2.0],[0.0,1.0,0.0,2.0],[1.0,0.0,1.0,0.16]])
#bodyLink.setPosition(LinkMat)
#body.body.defaultPosition = np.array([[1.0,0.0,0.0,2.0],[0.0,1.0,0.0,2.0],[1.0,0.0,1.0,0.16]])
world.addChildItem(body)
itv.checkItem(body,True)

# Load Controller for Tank
controller = SimpleControllerItem()
controller.setController("TankJoystickController.so")
body.addChildItem(controller)


#robot = lo(model, world)

#world.addChildItem(robot)



SimulationBar.instance.startSimulation(True)
startTime = time.time()

simlationFlag = 0



#print (time.time())
while 1:
    time.time()- startTime
    print("a")
    #print (time.time() - startTime)
    
    if (time.time() - startTime) > 5.0 and simlationFlag == 0:
        SimulationBar.instance.stopSimulation(aistSim)
        SimulationBar.instance.update()
        body.storeInitialState()
        startTime = time.time()
        simlationFlag = 1
    if (time.time() - startTime) > 1.0 and simlationFlag == 1:
        SimulationBar.instance.startSimulation(True)
        SimulationBar.instance.update()
        startTime = time.time()
        simlationFlag = 0
        #sys.exit()
#    if (time.time() - startTime) > 5.0 and simlationFlag == 1:
#        SimulationBar.instance.startSimulation(True)
#        simlationFlag = 0
#sleep(5)

#SimulationBar.instance.stopSimulation(aistSim)



