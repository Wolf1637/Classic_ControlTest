import gym
import numpy as np

dudes = 64
steps = 500
generation = 0
keep = 4  # 1/keep = part of old generation kept without mutation
rewMax = -10000
rewMaxTotal = -10000
timeAvg = 0
timeAvgTotal = 0
avgRew = 0
input1 = 24
output1 = 8
input2 = output1
output2 = 4

brain1Arr = np.zeros((dudes, input1, output1))
for i in range(dudes):
    brain1Arr[i] = np.random.rand(input1, output1)
brain1ArrSorted = brain1Arr

brain2Arr = np.zeros((dudes, input2, output2))
for i in range(dudes):
    brain2Arr[i] = np.random.rand(input2, output2)
brain2ArrSorted = brain2Arr

biasArr = np.zeros((dudes, output1))
for i in range(dudes):
    biasArr[i] = np.random.rand(input2)
biasArrSorted = biasArr

rewArr = np.zeros(dudes)
rewArr[:] = -10000
env = gym.make('BipedalWalker-v2')
while True:
    timeAvg = 0
    avgRew = 0
    rewMax = -10000

    for mutation in range(dudes):
        if mutation > dudes / keep:
            brain1 = brain1ArrSorted[mutation]
            brain2 = brain2ArrSorted[mutation]
            bias = biasArrSorted[mutation]
        else:
            brain1 = np.add(brain1ArrSorted[int(np.ceil(mutation / keep))],
                            np.random.normal(0.0, 1, (input1, output1)))
            brain2 = np.add(brain2ArrSorted[int(np.ceil(mutation / keep))],
                            np.random.normal(0.0, 1, (input2, output2)))
            bias = np.add(biasArrSorted[int(np.ceil(mutation / keep))], np.random.normal(0.0, 0.2, input2))
        rewSum = 0
        observation = env.reset()
        render = False

        if generation % (30) == 0:
            if mutation % 50 == 0:
                render = True
                print("Rendering generation {0} mutation {1}".format(generation, mutation))

        for t in range(steps):
            if render:
                env.render()
            # print(observation)
            action = np.matmul(np.tanh(np.add(np.matmul(observation, brain1), bias)), brain2)
            # if decision[0]<decision[1]:
            #    action = 0
            # else:
            #    action = 1

            # action = env.action_space.sample()
            # print(action)
            observation, reward, done, info = env.step(action)
            rewSum += reward

            if done:
                brain1Arr[mutation] = brain1
                rewArr[mutation] = rewSum
                # print("Mutation {0} finished after {1} timesteps".format(mutation+1,t+1))
                timeAvg += (t + 1) / dudes
                avgRew += rewSum / dudes
                break

    for i in range(dudes):
        brain1ArrSorted[i] = brain1Arr[rewArr.argmax()]
        brain2ArrSorted[i] = brain2Arr[rewArr.argmax()]
        biasArrSorted[i] = biasArr[rewArr.argmax()]
        if rewArr[rewArr.argmax()] > rewMax:
            rewMax = rewArr[rewArr.argmax()]
        rewArr[rewArr.argmax()] = -10000
    print("Generation {0} finished with max Reward {1}/{2} and {3}/{4} avg timesteps.".format(generation + 1,
                                                                                              np.round(rewMax, 2),
                                                                                              np.round(rewMaxTotal, 2),
                                                                                              np.round(timeAvg, 2),
                                                                                              np.round(timeAvgTotal,
                                                                                                       2)))

    if timeAvg > timeAvgTotal:
        timeAvgTotal = timeAvg
    if rewMax > rewMaxTotal:
        rewMaxTotal = rewMax
    generation += 1
