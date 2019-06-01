from agent import Agent
from monitor import interact
import gym
import numpy as np


def run(params):
    alpha, epsilon, gamma = params
    env = gym.make('Taxi-v2')
    agent = Agent(epsilon=epsilon, alpha=alpha, gamma=gamma)
    avg_rewards, best_avg_reward = interact(env, agent, 10000)
    return best_avg_reward


def twiddle_max(params, dparams, tol=0.01):  # Make this tolerance bigger if you are timing out!

    print('params ', params)
    print('dparams', dparams)
    best_best = run(params)
    n = 0
    # sum = integral of the cross track error
    while sum(dparams) > tol:
        for i in range(len(params)):
            params[i] += dparams[i]
            print('params ', params)
            print('dparamas', dparams)
            best = run(params)
            if best > best_best:
                # best is better than our best best
                best_best = best
                dparams[i] *= 1.1
            else:
                # error is not better than before
                # try other directions
                params[i] -= 2.0 * dparams[i]
                print('params ', params)
                best = run(params)
                if best > best_best:
                    best_best = best
                    dparams[i] *= 1.1
                else:
                    # did not succeed - decrease increments
                    params[i] += dparams[i]
                    dparams[i] *= 0.9
                    print('params ', params)

        n += 1
    print('Twiddle #', n, params, ' -> ', best_best)


    # print ' '
    return params

p = [0.05, 0.05, 0.8]
dp = [0.01, 0.01, 0.1]
params = twiddle_max(p, dp)

# err = run(params)
print
'\nFinal parameters: ', params, '\n -> ', err