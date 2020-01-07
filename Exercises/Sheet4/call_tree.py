from tree import *
from random import *
import time
from copy import *

#
# Create a set of randomly positioned particles
# For convenience we assume all masses to be 1.
# If we have in reality another mass, we can simply
# rescale our answers.
#
results = dict()

for nparticles in [20000, 40000]:  # add 40000 by hand
    # Begin tree algorithm
    particles = []
    for i in range(nparticles):
        x = random()
        y = random()
        z = random()
        particles.append([x, y, z])
    q = TreeClass(particles)
    q.insertallparticles()
    # Compute N^2 gravity
    print("starting N^2 gravity")
    t0 = time.time()
    q.all_exact_forces()
    t1 = time.time()
    fullgrav_dt = t1 - t0
    print("done in " + str(fullgrav_dt) + " seconds\n")
    fexact = deepcopy(q.forces)
    #
    # Compute tree algorithm for different angles and compare to exact solution
    #
    q.computemultipoles(0)
    results_angle = dict()
    for angle in [0.8, 0.4, 0.2]:
        print("starting tree gravity with angle", angle)
        t0 = time.time()
        q.allgforces(angle)
        t1 = time.time()
        treegrav_dt = t1-t0
        print("done in "+str(treegrav_dt)+" seconds\n")
        fapprox = deepcopy(q.forces)
        #
        # Now compare the approximate and exact versions
        #
        avgerror = 0

        # Calculate the distance of two points
        def distance(particle1, particle2):
            squared_dist = (particle1[0] - particle2[0]) ** 2 \
                           + (particle1[1] - particle2[1]) ** 2 \
                           + (particle1[2] - particle2[2]) ** 2
            return np.sqrt(squared_dist)


        for i in range(0, len(particles)):
            avgerror = avgerror + distance(fapprox[i], fexact[i]) / distance(fexact[i], [0., 0., 0.])

        avgerror = avgerror / nparticles
        print("The average error was computed to be:", avgerror)
        results_angle[angle] = [treegrav_dt, avgerror]
    results[nparticles] = [fullgrav_dt, results_angle]

print(results)

