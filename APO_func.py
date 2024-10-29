import numpy as np
import pandas as pd
import time
from cec2022 import F12022, F22022, F32022, F42022, F52022, F62022, F72022, F82022, F92022, F102022, F112022, \
    F122022
from matplotlib import pyplot as plt

# Function mapping
problem_mapping = {
    1: F12022,
    2: F22022,
    3: F32022,
    4: F42022,
    5: F52022,
    6: F62022,
    7: F72022,
    8: F82022,
    9: F92022,
    10: F102022,
    11: F112022,
    12: F122022
}


def APO_func(Fid, dim, pop_size, iter_max, Xmin, Xmax):
    np.random.seed(int(time.time() * 1000) % 4294967296)
    targetbest = np.array([300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700])
    fitness_history = []

    problem = problem_mapping[Fid](ndim=dim)

    fname = f'APO_Fid_{Fid}_{dim}D.txt'
    with open(fname, 'w') as f_out:
        f_out.write("Iteration: Best Fitness = Best Fitness Value\n")

    # 随机初始化 protozoa，除了最后两个参数
    protozoa = np.random.uniform(low=Xmin, high=Xmax, size=(pop_size, dim))
    # 确保最后两位是学习率和 epsilon，并且它们在预定的范围内
    protozoa[:, -2] = np.random.uniform(0.001, 0.1, size=pop_size)  # 学习率通常在0.001到0.1之间
    protozoa[:, -1] = np.random.uniform(0.1, 0.9, size=pop_size)  # epsilon通常在0.1到0.9之间

    start_time = time.time()
    best_learning_rate = None
    best_epsilon = None
    best_solution = None
    for iteration in range(1, iter_max + 1):
        for i in range(pop_size):
            ri = np.random.choice(range(pop_size), size=int(np.ceil(pop_size * 0.1)), replace=False)
            for j in ri:
                pdr = 0.5 * (1 + np.cos((1 - j / pop_size) * np.pi))
                if np.random.rand() < pdr:
                    protozoa[j] = Xmin + np.random.rand(dim) * (Xmax - Xmin)
                else:
                    flag = np.random.choice([-1, 1])
                    Mr = np.zeros(dim)
                    Mr[np.random.choice(dim, size=int(np.ceil(np.random.rand() * dim)), replace=False)] = 1
                    # Ensure learning_rate is in (0, 1) range
                    learning_rate = np.random.rand()
                    new_learning_rate = max(min(learning_rate, 1), 0)
                    protozoa[j] += flag * new_learning_rate * (Xmin + np.random.rand(dim) * (Xmax - Xmin)) * Mr

        fitness = np.array([problem.evaluate(individual) for individual in protozoa])
        fitness = np.array([problem.evaluate(individual) for individual in protozoa])
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        fitness_history.append(best_fitness)

        for i in range(pop_size):
            for j in range(dim - 2):  # 不更新学习率和epsilon
                protozoa[i, j] += np.random.normal(0, 1)  # 添加高斯扰动

            # 更新学习率和epsilon
            protozoa[i, -2] = np.clip(protozoa[i, -2] + np.random.normal(0, 0.01), 0.001, 0.1)
            protozoa[i, -1] = np.clip(protozoa[i, -1] + np.random.normal(0, 0.1), 0.1, 0.9)

        if iteration % (iter_max // 50) == 0:
            print(f'Iteration {iteration}: Best Fitness = {best_fitness}')
            with open(fname, 'a') as f_out:
                f_out.write(f"Iteration {iteration}: Best Fitness = {best_fitness}\n")

        best_idx = np.argmin(fitness)
        current_best_solution = protozoa[best_idx]
        current_best_learning_rate = current_best_solution[-2]
        current_best_epsilon = current_best_solution[-1]

        if best_solution is None or fitness[best_idx] < problem.evaluate(best_solution):
            best_solution = current_best_solution
            best_learning_rate = current_best_learning_rate
            best_epsilon = current_best_epsilon


    record_time = time.time() - start_time
    f_out.close()

    best_fitness = np.min(fitness_history)

    return best_fitness, record_time, best_solution, best_learning_rate, best_epsilon, fitness_history


# Example usage
Fid = 1
dim = 2
pop_size = 100
iter_max = 200
Xmin = -100
Xmax = 100

bestFit, elapsedTime, best_solution, best_learning_rate, best_epsilon, fitness_history = APO_func(Fid, dim, pop_size, iter_max, Xmin, Xmax)

print("Best Fitness:", bestFit)
print("Elapsed Time:", elapsedTime)
print("Best Solution:", best_solution)
print("Best Learning Rate:", best_learning_rate)
print("Best Epsilon:", best_epsilon)

data = {'Best Fitness': [bestFit],
        'Best Solution': [best_solution],
        'Elapsed Time': [elapsedTime],
        'Best Learning Rate': [best_learning_rate],
        'Best Epsilon': [best_epsilon]}

df = pd.DataFrame(data)

excel_file = 'E:\迅雷下载\code3DDQN\APO-1.xlsx'
df.to_excel(excel_file, index=False)

print("Results saved to", excel_file)

