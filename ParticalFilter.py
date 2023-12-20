import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 真实状态转移模型
def true_state_transition(state, velocity, acceleration, noise):
    return state + velocity + 0.5 * acceleration + noise
def true_velocity_transition(velocity, acceleration, noise):
    return velocity + acceleration + noise

# 观测模型
def observation_model(state, noise):
    return state + noise

# 粒子滤波器类
class ParticleFilter():
    def __init__(self, num_particles):
        self.num_particles = num_particles
        self.particles = np.random.rand(num_particles) * 2 - 1  # 初始粒子分布在[-1, 1]之间
        self.weights = np.ones(num_particles) / num_particles
        self.estimate_history = []  # 保存估计值的历史记录

    def resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def predict(self, velocity, acceleration, noise):
        self.particles += velocity + 0.5 * acceleration + noise

    def update(self, measurement, noise):
        a = np.exp(-0.5 * (measurement - self.particles + noise)**2 / 0.5**2)
        self.weights *= a
        self.weights /= np.sum(self.weights)  # 归一化
        self.estimate_history.append(np.sum(self.weights * self.particles))  # 保存当前时间步的估计值

# 模拟数据生成
def generate_data(num_steps):
    true_states = [0]
    measurements = [observation_model(0, np.random.normal(scale=0.1))]
    velocities = [0.1]
    accelerations = [0]
    for _ in range(1, num_steps):
        accelerations.append(0.1 + np.random.normal(scale=0.5))
        true_states.append(true_state_transition(true_states[-1], velocities[-1], accelerations[-1], 0))
        velocities.append(true_velocity_transition(velocities[-1], accelerations[-1], np.random.normal(scale=0.05)))
        measurements.append(observation_model(true_states[-1], np.random.normal(scale=2)))
    return true_states, measurements, accelerations, velocities

# 动画更新函数
def update(frame):
    ax.clear()

    # 真实状态
    ax.plot(true_states, label='True States', color='green')

    # 测量值
    ax.scatter(range(num_steps), measurements, label='Measurements', color='red', marker='x')

    # 粒子滤波器估计
    particle_filter.predict(velocity=velocities[frame], acceleration=accelerations[frame], noise=np.random.normal(scale=0.1))
    particle_filter.update(measurement=measurements[frame], noise=np.random.normal(scale=0.0))
    particle_filter.resample()

    # 修正这里，将估计值的历史记录绘制出来
    estimate_history = particle_filter.estimate_history
    if estimate_history:
        ax.plot(range(0, frame + 1), estimate_history[:frame+1], label='Particle Filter Estimate', color='blue')

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Position')
    ax.legend()

# 模拟数据生成
np.random.seed(42)
num_steps = 50
true_states, measurements, accelerations, velocities = generate_data(num_steps)

# 初始化粒子滤波器
particle_filter = ParticleFilter(num_particles=100)

# 设置动画
fig, ax = plt.subplots(figsize=(10, 6))
animation = FuncAnimation(fig, update, frames=num_steps, repeat=False)

# 显示动画
plt.show()
