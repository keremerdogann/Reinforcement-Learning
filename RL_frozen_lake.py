import gymnasium as gym
import numpy as np
import random
import time
import pygame

env = gym.make("FrozenLake-v1", is_slippery=False,render_mode="human")

action_size = env.action_space.n
state_size = env.observation_space.n  # Düzeltildi
q_table = np.zeros((state_size, action_size))

total_episodes = 10000
learning_rate = 0.1
max_steps = 100
gamma = 0.95
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

for episode in range(total_episodes):
    state = env.reset()[0]
    done = False

    for step in range(max_steps): #her adım icin karar verme
        exp_exp_tradeoff = random.uniform(0, 1) #0-1 arasında rastgele sayı uretelim.

        if exp_exp_tradeoff > epsilon: #eğer 0-1 aralığındaki rastgele sayı , giderek azalan epsilon değerinden fazlaysa artık bildiğimiz yolları tercih etmeliyiz.
            action = np.argmax(q_table[state, :]) #en iyi eylemi sec - SÖMÜRÜ
        else:
            action = env.action_space.sample() #yeni bir eylem dene - KEŞİF

        new_state, reward, done, truncated, info = env.step(action)

        #tüm işin arkasındaki Q fonksiyonu :
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
        state = new_state

        if done:
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

for episode in range(5):
    state = env.reset()[0]
    done = False
    moves = []  # Hareketleri kaydetmek için liste
    print("********** EPISODE ", episode + 1, " **********\n")
    time.sleep(1)

    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state, :])
        moves.append((state, action))  # Hareketi kaydet
        new_state, reward, done, truncated, info = env.step(action)
        time.sleep(0.5)

        if done:
            env.render()
            print("Yapılan hareketler:")
            for s, a in moves:
                print(f"Durum: {s}, Eylem: {a}")

            if reward == 1:
                print("Hedefe ulaşıldı! 🏆")
            else:
                print("Dondan kayarak düştü! ❄️")
            time.sleep(2)
            break

        state = new_state

env.close()
