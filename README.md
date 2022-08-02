# Reinforcement Learning Notes

My notes for learning RL.

## Setup

```
python -m venv env

.\env\Scripts\activate

pip install -r requirements.txt
```

## Test

```
python -m unittest
```

## Algorithms

### GLIE Monte-Carlo Control

Implementing the GLIE (Greedy in the Limit with Infinite Exploration) Monte-Carlo control algorithm on the [student MDP example](#student-mdp). GLIE refers to the policy converging to a greedy policy as the episodes increase. In this algorithm implementation we are using an epsilon-greedy policy, where epsilon is updated to 1 / the _k_ th episode.

Within each episode, every time a state-action pair is visited, the total discounted reward from that point onwards (or from that time step _t_ onwards) to the end of the episode is stored for each state-action pair.

![total discounted reward](./docs/total-discounted-reward.png)

That reward is then used to incrementally update the action-value function for all state-action pairs taken in that episode towards the mean of all the rewards seen for each state-action. To calculate the incremental mean, we keep track of the number of state-action visits across all episodes.

![action value incremental mean](./docs/action-value-incremental-mean.png)

At the end of each episode, we update the policy so as to converge on the optimal policy. 
1. We update epilson to 1/k, where _k_ is the episode; and 
2. We update the state-action mapping to be the action with the max action-value for each state.

![GLIE monte-carlo control policy update](./docs/glie-monte-carlo-control-policy-update.png)

#### Test GLIE Monte-Carlo Control

```
python -m unittest tests.test_monte_carlo.TestMonteCarlo.test_algorithm
```

**Note:** The resulting policy is deterministic (as the number of episodes goes to infinity) but the resulting action-value function is not, because of the stochastic nature of picking an action from a state combined with the policy converging to a greedy policy. This results in the values for each state-action pair to be farther from the true value as the state gets farther from the terminal state. For example, the state-action value of _State 3_ always equals the true state-action value, _State 2_ is typically (although not always) within +/-0.5 of the true state-action value, but _State 0_ is consistently outside +/-1 of the true state-action value. A couple ways to fix this (not yet explored) could be to converge to a greedy policy more slowly while increasing the number of total episodes.

#### GLIE Monte-Carlo Control References

1. Chapter 5.3 in [reference #1](#references).
2. Lecture 4 in [reference #2](#references) at [5:06](https://youtu.be/PnHCvfgC_ZA?t=306).
2. Lecture 5 in [reference #2](#references) at [25:37](https://youtu.be/0g4j2k_Ggc4?t=1537).

### Value Iteration

Implementing the value iteration algorithm on the [student MDP example](#student-mdp).

![optimal value iteration](./docs/algorithm-optimal-value-iteration.png)

![backup diagram value iteration](./docs/backup-diagram-value-iteration.png)

#### Test Value Iteration

```
python -m unittest tests.test_value_iteration.TestValueIteration.test_algorithm
```

#### Value Iteration References

1. Chapter 4.4 in [reference #1](#references).
2. Lecture 3 in [reference #2](#references) at [1:01:45](https://youtu.be/Nd1-UUMVfz4?t=3705).

## MDPs

### Student MDP

An example MDP for testing algorithms based off of the example in Lecture 2 in [reference #2](#references) at [1:21:47](https://youtu.be/lfHX2hHRMVQ?t=4907).

![Student MDP Example](./docs/example-mdp-student.png)

#### Solved Student MDP

The optimal value function, action-value function and policy for this student MDP example. The algorithm test on this MDP should target these functions.

![Solved Student MDP Example](./docs/example-mdp-student-solved.png)

## References

1. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf)
2. [DeepMind x UCL | Introduction to Reinforcement Learning 2015](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

