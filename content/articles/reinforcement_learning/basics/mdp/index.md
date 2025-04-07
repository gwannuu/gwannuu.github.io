+++
date = '2025-04-07T20:09:13+09:00'
draft = false
title = 'Markov Decision Process (MDP)'
tag = ['Markov Decision Process', 'MDP', 'Markov Proerty', 'Markov Process']
+++

# Markov Property and Markov Process
__Markov property__ describes a stochastic process in which the future state is independent of the past states, given the present state.
Let assume that $x_1, x_2, \dots$ are elements of stochastic process, and the following condition holds, which is called as _Markov property_:
$$
P(x_n \mid x_{n-1}) = P(x_n \mid x_{n-1}, x_{n-2}, \dots, x_1)
$$
Then, The stochastic process $x_1, x_2, \dots$ is called a __markov process__.

# Markov Decision Process
## Example
Consider the grid world environment in which agent can moves to four directions, $\textit{left, right, up, down}$, and agent gets reward $+1$ if agent arrives in goal state. Unless the agent don't arrives in goal states, everytime it gets $0$ reward.

In this situation, the agent can takes one of four action, $\textit{left, right, up, down}$ at each timestep $t$.
Let denote the chosen action as $a_t$ and position of agent in current timestep as $s_t$.
Then in next timestep, position of user $s_{t+1}$ information will becomes different with $s_t$ and agent will get reward $r_t$ from environment.

Situation like this is called **Markov Decision Process**, in which agent can choose corresponding actions from current state $s_t$ and gets next state $s_{t+1}$ and reward $r_t$.

## definition

Markov Decision process is uniquely defined by a tuple $\langle \mathcal{S}, \mathcal{A}, R, P \rangle$ , that consists of state space $\mathcal{S}$, action space $\mathcal{A}$, transition probability $P : \mathcal{S} \times \mathcal{S} \times \mathcal{A} \rightarrow \[0, \infty \)$ and reward function $R : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$.

