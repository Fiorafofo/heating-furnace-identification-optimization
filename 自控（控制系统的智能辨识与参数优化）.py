import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 中文、负号显示设置
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 读取温度数据表格
df = pd.read_excel('temperature.xlsx', engine='openpyxl')
time = df['time'].values
temperature = df['temperature'].values
volte = df['volte'].values

# 数据预处理，计算初始、稳态温度
y_0 = np.mean(temperature[:1000])
y_ss = np.mean(temperature[-1000])
delta_y = y_ss - y_0
delta_u = volte[0]
print(f"系统初始值：{y_0:.2f}℃，稳态值：{y_ss:.2f}℃，输入电压：{delta_u}")

# 两点法辨识一阶惯性纯滞后模型
K = delta_y / delta_u
y_632 = y_0 + delta_y * 0.632
y_865 = y_0 + delta_y * 0.865
t1 = time[np.argmin(np.abs(temperature - y_632))]
t2 = time[np.argmin(np.abs(temperature - y_865))]
T = 2 * (t2 - t1)
tau = 2 * t1 - t2
print("====两点法辨识模型参数====")
print(f"放大系数K={K:.3f}，时间常数T={T:.2f}s，滞后τ={tau:.2f}s")

# 最小二乘数据计算异常，统一使用两点法参数仿真
T_sys = T
K_sys = K
tau_sys = tau

# PID控制器类
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, process_value, dt):
        error = self.setpoint - process_value
        self.integral += error * dt
        deriv = (error - self.prev_error) / dt
        u = self.Kp * error + self.Ki * self.integral + self.Kd * deriv
        u = np.clip(u, 0, 10)
        self.prev_error = error
        return u

# 被控对象仿真函数
def plant_simulation(u_seq, dt, T, tau, K, y0):
    delay_steps = int(tau / dt)
    N = len(u_seq)
    y = np.full(N, y0)
    alpha = np.exp(-dt / T)
    for i in range(delay_steps, N):
        y[i] = alpha * y[i-1] + (1 - alpha) * K * u_seq[i-delay_steps]
    return y

# ITAE目标函数（DE适应度）
def objective(pid_params, setpoint, dt, T, tau, K, y0, sim_time):
    Kp, Ki, Kd = pid_params
    pid_obj = PIDController(Kp, Ki, Kd, setpoint)
    n = int(sim_time / dt)
    u = np.zeros(n)
    y = np.full(n, y0)
    itae = 0.0
    max_overshoot = 0.0
    for k in range(1, n):
        u[k] = pid_obj.update(y[k-1], dt)
    y_out = plant_simulation(u, dt, T, tau, K, y0)
    time_arr = np.arange(0, sim_time, dt)
    for idx, t in enumerate(time_arr):
        err = abs(setpoint - y_out[idx])
        itae += t * err
        over = y_out[idx] - setpoint
        if over > max_overshoot:
            max_overshoot = over
    return itae + 100 * max_overshoot

# 差分进化算法
def differential_evolution(objective_func, bounds, NP=10, F=0.6, CR=0.5, G_max=20, args=()):
    dim = len(bounds)
    pop = np.random.rand(NP, dim)
    for d_idx in range(dim):
        low, high = bounds[d_idx]
        pop[:, d_idx] = pop[:, d_idx] * (high - low) + low
    fit = np.array([objective_func(ind, *args) for ind in pop])
    best_idx = np.argmin(fit)
    best_ind = pop[best_idx].copy()
    best_fit = fit[best_idx]
    for gen in range(G_max):
        for i in range(NP):
            others = [k for k in range(NP) if k != i]
            r1, r2, r3 = np.random.choice(others, 3, replace=False)
            v = pop[r1] + F * (pop[r2] - pop[r3])
            for d in range(dim):
                v[d] = np.clip(v[d], bounds[d][0], bounds[d][1])
            trial = pop[i].copy()
            jr = np.random.randint(0, dim)
            for d in range(dim):
                if np.random.rand() < CR or d == jr:
                    trial[d] = v[d]
            f_trial = objective_func(trial, *args)
            if f_trial < fit[i]:
                pop[i] = trial
                fit[i] = f_trial
        cur_min = np.min(fit)
        if cur_min < best_fit:
            best_fit = cur_min
            best_ind = pop[np.argmin(fit)]
    return best_ind, best_fit

# ---------------------- 主程序入口 ----------------------
if __name__ == "__main__":
    set_temp = 35
    sim_dt = 0.5
    sim_total = 3000
    bounds = [(0, 10), (0, 0.1), (0, 100)]
    # DE传入参数列表
    arg_tuple = (set_temp, sim_dt, T_sys, tau_sys, K_sys, y_0, sim_total)
    # 执行差分进化寻优
    best_pid_params, best_J = differential_evolution(objective, bounds, NP=10, G_max=20, args=arg_tuple)
    Kp_opt, Ki_opt, Kd_opt = best_pid_params
    print("\n==== DE优化完成 PID最优参数 ====")
    print(f"比例 Kp = {Kp_opt:.2f}")
    print(f"积分 Ki = {Ki_opt:.4f}")
    print(f"微分 Kd = {Kd_opt:.2f}")
    print(f"ITAE综合指标 = {best_J:.2f}")

    # ========== 人工经验PID仿真（修复变量名pid未定义错误） ==========
    pid_exp = PIDController(2.5, 0.001, 60, set_temp)
    n_sim = int(sim_total / sim_dt)
    u_exp = np.zeros(n_sim)
    y_exp = np.full(n_sim, y_0)
    for i in range(1, n_sim):
        u_exp[i] = pid_exp.update(y_exp[i-1], sim_dt)
    y_exp_out = plant_simulation(u_exp, sim_dt, T_sys, tau_sys, K_sys, y_0)

    # ========== DE优化PID仿真 ==========
    pid_opt = PIDController(Kp_opt, Ki_opt, Kd_opt, set_temp)
    u_opt = np.zeros(n_sim)
    y_opt = np.full(n_sim, y_0)
    for i in range(1, n_sim):
        u_opt[i] = pid_opt.update(y_opt[i-1], sim_dt)
    y_opt_out = plant_simulation(u_opt, sim_dt, T_sys, tau_sys, K_sys, y_0)

    # 图1：原始加热炉阶跃实测曲线
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(time, temperature, c="black", label="实测炉温")
    plt.xlabel("时间 / s")
    plt.ylabel("温度 / ℃")
    plt.title("加热炉阶跃响应实测曲线")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show(block=False)

    # 图2：人工PID vs 优化PID对比曲线
    t_axis = np.arange(0, sim_total, sim_dt)
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(t_axis, y_exp_out, label="人工经验PID")
    plt.plot(t_axis, y_opt_out, label="DE差分进化优化PID")
    plt.axhline(y=35, c="red", linestyle="--", label="设定温度35℃")
    plt.xlabel("时间 / s")
    plt.ylabel("温度 / ℃")
    plt.title("两种PID控制响应对比图")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show(block=False)
