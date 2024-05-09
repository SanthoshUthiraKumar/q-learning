# Q Learning Algorithm

## AIM
To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Develop a Python program to derive the optimal policy using Q-Learning and compare state values with Monte Carlo method.

## Q LEARNING ALGORITHM
### Step 1:
Initialize Q-table and hyperparameters.
### Step 2:
Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.
### Step 3:
After training, derive the optimal policy from the Q-table.
### Step 4:
Implement the Monte Carlo method to estimate state values.
### Step 5:
Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.

## Q LEARNING FUNCTION
```python
#NAME : Santhosh U
#REG NO : 212222240092
def q_learning(env, 
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    alphas = decay_schedule(
        init_alpha,min_alpha,
        alpha_decay_ratio,n_episodes)
    epsilons=decay_schedule(
        init_epsilon,min_epsilon,epsilon_decay_ratio,
        n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      while not done:
        action=select_action(state,Q,epsilons[e])
        next_state,reward,done,_=env.step(action)
        td_target = reward + gamma * np.max(Q[next_state]) * (not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state=next_state
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
### Optimal policy, optimal value function , success rate for the optimal policy.
![329318207-7f9eeefd-01c8-44ca-ac3c-41e2640cf6ef](https://github.com/SanthoshUthiraKumar/q-learning/assets/119477975/de471fdb-8e77-4105-86a7-e5895137f1dc)

### State value functions of Monte Carlo method:
![329318245-b82e1a59-dfe1-45ca-891c-cb279a94cdc8](https://github.com/SanthoshUthiraKumar/q-learning/assets/119477975/479d323c-6e27-484b-9508-b1e5f413809c)

### State value functions of Qlearning method:
![329318307-32cc376e-d309-4fa1-ac66-40a89cf6cb93](https://github.com/SanthoshUthiraKumar/q-learning/assets/119477975/25e367d9-a7f4-4608-bdcd-3570d1fdf5aa)

## RESULT:
Thus, Q-Learning outperformed Monte Carlo in finding the optimal policy and state values for the RL problem.
