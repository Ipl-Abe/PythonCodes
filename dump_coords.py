from seq_pair02 import simulated_annealing, instance_generator
import _pickle as pickle 
 
if __name__ == '__main__':
    for i in range(1, 31):
        sa = simulated_annealing(instance_generator(40))
        sa.sim()
        with open(str(i).zfill(3) + '.txt', 'w') as f:
            pickle.dumps(sa.result(), f)