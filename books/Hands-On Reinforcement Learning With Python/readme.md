# Hands-On Reinforcement Learning With Python
```
官方程式碼
https://github.com/sudharsan13296/Hands-On-Reinforcement-Learning-With-Python
```
```
中譯本
Python强化学习实战：应用OpenAI Gym和TensorFlow精通强化学习和深度强化学习

用Python實作強化學習：使用TensorFlow與OpenAI Gym
Hands-On Reinforcement Learning with Python
作者： Sudharsan Ravichandiran  
碁峰出版社： 出版日期：2019/05/29
```
```
Master reinforcement and deep reinforcement learning using OpenAI Gym and TensorFlow

Reinforcement Learning with Python will help you to master basic reinforcement learning algorithms 
to the advanced deep reinforcement learning algorithms.

The book starts with an introduction to Reinforcement Learning followed by OpenAI and Tensorflow. 
You will then explore various RL algorithms and concepts such as the Markov Decision Processes, 
Monte-Carlo methods, and dynamic programming, including value and policy iteration. 

This example-rich guide will introduce you to deep learning, covering various deep learning algorithms. 

You will then explore deep reinforcement learning in depth, 
which is a combination of deep learning and reinforcement learning. 

You will master various deep reinforcement learning algorithms 
such as DQN, Double DQN. Dueling DQN, DRQN, A3C, DDPG, TRPO, and PPO. 

You will also learn about recent advancements in reinforcement learning 
such as imagination augmented agents, learn from human preference, DQfD, HER and many more.
```
```
第一章｜認識強化學習
介紹何謂強化學習以及其運作原理。介紹強化學習的各種元素，如代理、環境、策略與模型，
並帶領讀者認識用於強化學習的各種環境、平台與函式庫，以及強化學習的一些應用。

第二章｜認識OpenAI與TensorFlow
建置使用強化學習的電腦環境，包括Anaconda、Docker、OpenAI Gym、Universe與TensorFlow的安裝設定，
並說明如何在OpenAI Gym中來模擬代理，以及如何建置一個會玩電玩遊戲的機器人程式。
另外也會解說TensorFlow的基礎觀念以及如何使用TensorBoard來進行視覺化操作。

第三章｜Markov決策過程與動態規劃
從介紹何謂Markov鍊與Markov流程開始，說明如何使用Markov決策流程來對強化學習問題來建模。
接著是一些重要的基本概念，例如價值函數、Q函數與Bellman方程式。
然後介紹動態規劃以及如何運用價值迭代與策略迭代來解決凍湖問題。

第四章｜使用Monte Carlo方法來玩遊戲
介紹了Monte Carlo法與不同類型的 Monte Carlo預測法，如首次拜訪MC與每次拜訪MC，
並說明如何使用Monte Carlo法來玩二十一點這項撲克牌遊戲。最後會介紹現時與離線這兩種不同的Monte Carlo控制方法。

第五章｜時間差分學習
介紹時間差分（TD）學習、TD預測與TD的即時/離線控制法，如Q學習與SARSA。
並說明如何使用Q學習與SARSA來解決計程車載客問題。

第六章｜多臂式吃角子老虎機問題
要討論的是強化學習的經典問題：多臂式吃角子老虎機（MAB）問題，也稱為k臂式吃角子老虎機（MAB）問題。
介紹如何使用各種探索策略來解決這個問題，例如epsilon-貪婪、softmax探索、UCB與湯普森取樣。
本章後半也會介紹如何運用MAB來對使用者顯示正確的廣告橫幅。

第七章｜深度學習的基礎概念
介紹深度學習的重要觀念。首先，說明何謂神經網路，接著是不同類型的神經網路，如RNN、LSTM與CNN等。
本章將實作如何自動產生歌詞與分類時尚產品。

第八章｜使用深度Q網路來玩Atari遊戲
介紹了一套最常用的深度強化學習演算法：深度Q網路（DQN）。接著介紹DQN的各個元件，
並說明如何運用DQN來建置代理來玩Atari遊戲。最後介紹一些新型的DQN架構，如雙層DQN與競爭DQN。

第九章｜使用深度循環Q網路來玩毀滅戰士
介紹深度循環Q網路（DRQN），並說明它與DQN的差異。本章會運用DRQN來建置代理來玩毀滅戰士遊戲。
同時介紹深度專注循環Q網路，它在DRQN架構中加入了專注機制。

第十章｜非同步優勢動作評價網路
介紹了非同步優勢動作評價網路（A3C）的運作原理。我們將帶領你深入了解A3C的架構並學會如何用它來建置會爬山的代理。

第十一章｜策略梯度與最佳化
說明策略梯度如何在不需要Q函數的前提下，幫助我們找到正確的策略。同時還會介紹深度確定性策略梯度法，
以及最新的策略最佳化方法，如信賴域策略最佳化與近端策略最佳化。

第十二章 使用DQN來玩賽車遊戲
本章將帶領你運用競爭DQN來建置代理，讓它學會玩賽車遊戲。

第十三章 近期發展與下一步
介紹強化學習領域中的各種最新發展，例如想像增強代理、從人類偏好來學習、
由示範來進行的深度Q學習以及事後經驗回放等等，然後談到了不同的強化學習方法，如層次強化學習與逆向強化學習。
```
### Table of Contents
```
1. Introduction to Reinforcement Learning
1.1. What is Reinforcement Learning?
1.2. Reinforcement Learning Cycle
1.3. How RL differs from other ML Paradigms?
1.4. Elements of Reinforcement Learning
1.5. Agent Environment Interface
1.6. Types of RL Environments
1.7. Reinforcement Learning Platforms
1.8. Applications of Reinforcement Learning


2. Getting Started with OpenAI and Tensorflow
2.1. Setting Up Your Machine
2.2. Installing Anaconda
2.3. Installing Docker
2.4. Installing OpenAI Gym and Universe
2.5. Common Error Fixes
2.6. OpenAI Gym
2.7. Basic Simulations
2.8. Training a Robot to walk
2.9. Building a Video Game Bot
2.10. Tensorflow Fundamentals
2.11. Tensorboard


3. Markov Decision Process and Dynamic Programming
3.1. Markov Chain and Markov Process
3.2. Markov Decision Process
3.3. Rewards and Returns
3.4. Episodic and Continous Tasks
3.5. Policy Function
3.6. State Value Function
3.7. State-Action Value Function (Q Function)
3.8. Bellman Equation and Optimality
3.9. Deriving Bellman Equation for Value and Q functions
3.10. Solving the Bellman Equation
3.11. Dynamic Programming
3.12. Solving Frozen Lake Problem using Value Iteration
3.13. Solving Frozen Lake Problem using Policy Iteration


4. Gaming with Monte Carlo Methods
4.1. Monte Carlo Methods
4.2. Estimating Value of Pi Using Monte Carlo
4.3. Monte Carlo Prediction
4.4. First visit Monte Carlo
4.5. Every visit Monte Carlo
4.6. BlackJack with Monte Carlo
4.7. Monte Carlo Control
4.8. Monte Carlo Exploration Starts
4.9. On Policy Monte Carlo Control
4.10. Off Policy Monte Carlo Control


5. Temporal Difference Learning
5.1. Temporal Difference Learning
5.2. TD Prediction
5.3. TD Control
5.4. Q Learning
5.5. Solving the Taxi Problem using Q learning
5.6. SARSA
5.7. Solving the Taxi Problem using SARSA
5.8. Difference Between Q learning and SARSA


6. Multi-Armed Bandit Problem
6.1. Multi-armed Bandit Problem
6.2. Epsilon-Greedy Algorithm
6.3. Softmax Exploration Algorithm
6.4. Upper Confidence Bound Algorithm
6.5. Thompson Sampling Algorithm
6.6. Applications of MAB
6.7. Identifying Right Advertisement Banner Using MAB
6.8. Contextual Bandits


7. Deep Learning Fundamentals
7.1. Artificial Neurons
7.2. Artificial Neural Network
7.3. Activation Functions
7.4. Deep Dive into ANN
7.5. Gradient Descent
7.6. Neural Networks in Tensorflow
7.7. Recurrent Neural Network
7.8. Backpropagation Through Time
7.9. Long Short Term Memory RNN
7.10. Generating Song Lyrics using LSTM RNN
7.11. Convolutional Neural Networks
7.12. CNN Architecture
7.13. Classifying Fashion Products Using CNN


8. Atari Games With Deep Q Network
8.1. What is Deep Q network
8.2. Architecture of DQN
8.3. Convolutional Network
8.4. Experience Replay
8.5. Target Network
8.6. Clipping Rewards
8.7. DQN Algorithm
8.8. Building an Agent to Play Atari Games
8.9. Double DQN
8.10. Dueling Architecture


9. Playing Doom With Deep Recurrent Q Network
9.1. Deep Recurrent Q Network
9.2. Partially Observable MDP
9.3. Architecture of DRQN
9.4. Basic Doom Game
9.5. Build an Agent to Play Doom Game using DRQN
9.6. Deep Attention Recurrent Q Network

10. Asynchronous Advantage Actor Critic Network
10.1. Asynchronous Actor Critic Algorithm
10.2. The three A's
10.3. Architecture of A3C
10.4. Working of A3C
10.5. Drive up the Mountain with A3C
10.6. Visualization in Tensorboard

11. Policy Gradients and Optimization
11.1. Policy Gradient
11.2. Lunar Lander Using Policy Gradient
11.3. Deep Deterministic Policy Gradient
11.4. Swinging up the Pendulum using DDPG
11.5. Trust Region Policy Optimizatio
11.6. Proximal Policy Optimization

12. Capstone Project: Car Racing using DQN
12.1. Environment Wrapper Functions
12.2. Dueling Network
12.3. Replay Buffer
12.4. Training the Network
12.5. Car Racing

13. Recent Advancements and Next Steps
13.1. Imagination Augmented Agents
13.2. Learning From Human Preference
13.3. Deep Q Learning From Demonstrations
13.4. Hindsight Experience Replay
13.5. Hierarchical Reinforcement Learning
13.6. Inverse Reinforcement Learning
```
