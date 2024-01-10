import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Estimation parameter of PF
Q = np.diag([0.4]) ** 2  # range error

# 模拟数据生成
np.random.seed(42)
num_steps = 100
NP = 1000  # Number of Particle
NTh = NP / 2.0  # Number of particle for re-sampling

# 真实状态转移模型
def true_state_transition(state, velocity, noise):
    return state + velocity + noise


def gauss_likelihood(x, sigma):
    p = 1.0 / np.sqrt(2.0 * np.pi * sigma ** 2) * \
        np.exp(-x ** 2 / (2 * sigma ** 2))
    return p


# def without_pf_state_transition(state, velocity, acceleration, noise):
#     return state + velocity + 0.5 * acceleration + noise

# 观测模型
def observation_model(state, noise):
    return state + noise

# 粒子滤波器类
class ParticleFilter():
    def __init__(self, num_particles):
        self.num_particles = num_particles
        # 初始粒子按真值初始化成一样的值就行！多个粒子模拟多种情况，不是体现在这里的初始化为多种随机值！
        # 是相同初始化，模拟多种不同的速度变化趋势
        # 不是模拟多种初始化，采用同一种速度变化趋势
        self.particles = np.zeros(num_particles)
        self.weights = np.ones(num_particles) / num_particles
        self.estimate_history = []  # 保存估计值的历史记录

    def resample(self):
        try:
            indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        except ValueError as e:
            self.weights = np.ones(self.num_particles) / self.num_particles
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def predict(self, velocity, noise):
        self.particles += velocity + noise

    def update(self, measurement, noise):
        dz = measurement - self.particles + noise
        a = np.exp(-0.5 * dz**2 / 0.5**2)
        p = gauss_likelihood(dz, math.sqrt(Q[0, 0]))
        self.weights *= p
        self.weights /= np.sum(self.weights)  # 归一化
        # self.weights = self.weights / self.weights.sum()  # 归一化
        self.estimate_history.append(np.sum(self.weights * self.particles))  # 保存当前时间步的估计值

# 模拟数据生成
# measurements 相当于 landmark (GPS观测)
def generate_data(num_steps):
    true_states = [0]
    dead_reckoning = [0]
    measurements = [0]
    velocities = [0]
    dr_vel = [0]

    for _ in range(1, num_steps):
        vel = 0.1
        true_states.append(true_state_transition(true_states[-1], velocities[-1], 0))
        dead_reckoning.append(true_state_transition(dead_reckoning[-1], dr_vel[-1], 0))
        velocities.append(vel)
        dr_vel.append(vel + np.random.normal(scale=0.2))
        measurements.append(observation_model(true_states[-1], np.random.normal(scale=0.1)))
    return dead_reckoning, true_states, measurements, dr_vel

# 动画更新函数
def update(frame):
    ax.clear()

    # 真实状态
    ax.plot(true_states, label='True States', color='green')
    # ax.plot(dead_reckoning, label='Dead Reckoning', color='black')

    # 测量值
    ax.scatter(range(num_steps), measurements, label='Measurements', color='red', marker='x')

    # 粒子滤波器估计
    ####################################### Key #####################################
    # 每个粒子的噪声都要是不一样的！要生成粒子数个噪声！随机性要体现在这里！即每个粒子都模拟一种速度变化，看看哪种更像！而不是体现在所有粒子初始化的时候！
    particle_filter.predict(velocity=velocities[frame], noise=np.random.normal(scale=0.2, size=NP))
    # 这种写法只生成一个噪声！若所有的粒子加的噪声都是相同的，就相当于只使用了一个粒子！
    # particle_filter.predict(velocity=velocities[frame], noise=np.random.normal(scale=0.2))
    ##################################################################################


    # without_pf.append(without_pf_state_transition(true_states[-1], velocities[-1], accelerations[-1], 0))
    particle_filter.update(measurement=measurements[frame], noise=np.random.normal(scale=0.0))
    particle_filter.resample()

    N_eff = 1.0 / (particle_filter.weights.T.dot(particle_filter.weights))  # Effective particle number
    if N_eff < NTh:
        particle_filter.resample()

    # 将估计值的历史记录绘制出来
    estimate_history = particle_filter.estimate_history
    if estimate_history:
        ax.plot(range(0, frame + 1), estimate_history[:frame+1], label='Particle Filter Estimate', color='blue')
        ax.plot(range(0, frame + 1), dead_reckoning[:frame+1], label='Dead Reckoning', color='black')

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Position')
    ax.legend()


dead_reckoning, true_states, measurements, velocities = generate_data(num_steps)

# 初始化粒子滤波器
particle_filter = ParticleFilter(num_particles=NP)

# 设置动画
fig, ax = plt.subplots(figsize=(10, 6))
animation = FuncAnimation(fig, update, frames=num_steps, repeat=False)

# 显示动画
plt.show()
