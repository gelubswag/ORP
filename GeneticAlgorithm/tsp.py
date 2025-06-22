import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np

# Параметры сетки
GRID_SIZE = 100
MAX_WEIGHT = 1000
MIN_WEIGHT = 0.1

# Параметры алгоритма
MAX_STEPS = 1000
GENERATIONS = 1000
POPULATION_SIZE = 100
MUTATION_RATE = 0.1
ELITISM = 0.1


def smooth_grid(grid: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Применяет размытие к сетке для плавных переходов"""
    new_grid = np.zeros_like(grid)
    pad = kernel_size // 2
    padded = np.pad(grid, pad, mode='edge')

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            new_grid[i, j] = np.mean(window)

    return new_grid


def generate_weight_grid(
    grid_size: int,
    weight_type: str = "random"
) -> np.ndarray:
    """Генерирует сетку весов"""
    def normalize_weights(weights):
        """Нормализует веса в заданный диапазон"""
        min_val = np.min(weights)
        max_val = np.max(weights)
        if max_val - min_val > 1e-6:
            weights = (weights - min_val) / (max_val - min_val)
            weights = weights * (MAX_WEIGHT - MIN_WEIGHT) + MIN_WEIGHT
        return weights

    if weight_type == "1":
        weights = np.random.uniform(
            MIN_WEIGHT,
            MAX_WEIGHT,
            (grid_size, grid_size)
            )

    elif weight_type == "2":
        weights = np.random.uniform(
            MIN_WEIGHT,
            MAX_WEIGHT,
            (grid_size, grid_size)
            )
        weights = smooth_grid(weights, kernel_size=5)

    elif weight_type == "3":
        weights = np.ones((grid_size, grid_size))
        # Прямоугольные зоны
        weights[
            grid_size//4:3*grid_size//4,
            grid_size//5:4*grid_size//5
            ] = MAX_WEIGHT * 0.6
        weights[grid_size//3:2*grid_size//3, :grid_size//2] = MAX_WEIGHT * 0.4

        # Диагональная зона
        for i in range(grid_size):
            for j in range(grid_size):
                if abs(i - j) < grid_size//5:
                    weights[i, j] = MAX_WEIGHT * 0.6

        # Случайные вариации
        weights += np.random.rand(grid_size, grid_size) * MAX_WEIGHT * 0.2
    elif weight_type == "4":
        # Вертикальный градиент
        weights = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            weights[i, :] = np.linspace(MIN_WEIGHT, MAX_WEIGHT, grid_size)

        # Добавление горизонтального градиента
        for j in range(grid_size):
            weights[:, j] += np.linspace(
                MAX_WEIGHT * 0.2,
                MIN_WEIGHT,
                grid_size
                )

        weights = normalize_weights(weights)

    elif weight_type == "5":
        weights = np.random.uniform(
            MIN_WEIGHT,
            MAX_WEIGHT,
            (grid_size, grid_size)
            )
        # Создание диагонального коридора
        for i in range(grid_size-1):
            weights[i, i] = MIN_WEIGHT
            weights[i+1, i] = MIN_WEIGHT * 2
            weights[i, i+1] = MIN_WEIGHT * 2
    elif weight_type == "6":
        weights = np.ones((grid_size, grid_size))
        weights[0, :] = MIN_WEIGHT
        weights[:, 0] = MIN_WEIGHT
        weights[1, :] = MIN_WEIGHT
        weights[:, 1] = MAX_WEIGHT
        weights[:, 2] = MAX_WEIGHT
        weights[2, :] = MAX_WEIGHT
        weights[GRID_SIZE - 1, :] = MAX_WEIGHT
        weights[GRID_SIZE - 2, :] = MAX_WEIGHT
        weights[GRID_SIZE - 3, :] = MAX_WEIGHT
        weights[:, GRID_SIZE - 3] = MAX_WEIGHT
        weights[:, GRID_SIZE - 2] = MAX_WEIGHT
        weights[:, GRID_SIZE - 1] = MAX_WEIGHT
        weights[:, GRID_SIZE // 2] = MAX_WEIGHT
        weights[GRID_SIZE // 2, :] = MAX_WEIGHT
        for i in range(grid_size - 3):
            weights[i, i] = MAX_WEIGHT
            weights[i+1, i+1] = MAX_WEIGHT
            weights[i+2, i+2] = MAX_WEIGHT
            weights[i+3, i+3] = MAX_WEIGHT
    else:  # uniform
        weights = np.ones((grid_size, grid_size))

    # Гарантируем, что старт и финиш легко проходимы
    weights[0, 0] = MIN_WEIGHT
    weights[-1, -1] = MIN_WEIGHT

    return weights


def simple_sols(
    grid_size: int,
    weight_grid: np.ndarray
) -> List[List[str]]:
    down = ["D" for _ in range(grid_size-1)]
    right = ["R" for _ in range(grid_size-1)]

    random_population = []
    for _ in range(POPULATION_SIZE - 6):
        path = down + right
        random.shuffle(path)
        random_population.append(path)

    diag_r = ["R", "D"] * (grid_size - 1)
    diag_d = ["D", "R"] * (grid_size - 1)

    zigzag_rd = ["R", "D", "R", "U"] * ((grid_size - 1) // 2)
    zigzag_ru = ["R", "U", "R", "D"] * ((grid_size - 1) // 2)
    zigzag_dr = ["D", "R", "D", "L"] * ((grid_size - 1) // 2)
    zigzag_dl = ["D", "L", "D", "R"] * ((grid_size - 1) // 2)

    simple_pops = [
        down + right,
        right + down,
        diag_r,
        diag_d,
        zigzag_rd + zigzag_dl,
        zigzag_dr + zigzag_ru,
    ] + random_population

    return simple_pops


class Animal:
    def __init__(
        self,
        genes: List[str],
        grid_size: int,
        max_steps: int,
        weight_grid: np.ndarray
    ):
        self.genes = genes
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.weight_grid = weight_grid
        self.fitness = 0.0
        self.completed_path = []
        self.effective_steps = 0
        self.total_weight = 0
        self.visited_cells = {}
        self.loop_penalty = 0
        self.win = False

    def calculate_fitness(self, generation: int) -> float:
        x, y = 0, 0
        self.effective_steps = 0
        self.completed_path = []
        self.total_weight = 0
        self.visited_cells = {(x, y): 1}
        self.loop_penalty = 0
        penalty = 0
        optimal_steps = 2 * (self.grid_size - 1)

        self.total_weight += self.weight_grid[y, x]

        for i in range(min(len(self.genes), self.max_steps)):
            step = self.genes[i]
            if x == self.grid_size-1 and y == self.grid_size-1:
                break

            new_x, new_y = x, y
            if step == "U" and y > 0:
                new_y -= 1
            elif step == "D" and y < self.grid_size-1:
                new_y += 1
            elif step == "L" and x > 0:
                new_x -= 1
            elif step == "R" and x < self.grid_size-1:
                new_x += 1

            if (new_x, new_y) != (x, y):
                visit_count = self.visited_cells.get((new_x, new_y), 0)
                if visit_count > 0:
                    revisit_penalty = 2 ** visit_count
                    self.loop_penalty += revisit_penalty

                    # Штраф за колебания
                    if len(self.completed_path) > 0:
                        prev_step = self.completed_path[-1]
                        if (
                            (step == "U" and prev_step == "D")
                            or (step == "D" and prev_step == "U")
                            or (step == "L" and prev_step == "R")
                            or (step == "R" and prev_step == "L")
                        ):
                            self.loop_penalty += MAX_WEIGHT

                x, y = new_x, new_y
                self.completed_path.append(step)
                self.visited_cells[(x, y)] = visit_count + 1
                self.total_weight += self.weight_grid[y, x]
            else:
                penalty += 5

        # Достройка пути
        remaining_right = (self.grid_size-1) - x
        remaining_down = (self.grid_size-1) - y
        auto_steps = min(optimal_steps, remaining_right + remaining_down)
        added = 0

        if x < self.grid_size-1 or y < self.grid_size-1:
            while x < self.grid_size-1 and added < auto_steps:
                new_x = x + 1
                visit_count = self.visited_cells.get((new_x, y), 0)
                if visit_count > 0:
                    self.loop_penalty += 2 ** visit_count
                self.completed_path.append("R")
                x = new_x
                self.visited_cells[(x, y)] = visit_count + 1
                self.total_weight += self.weight_grid[y, x]
                added += 1

            while y < self.grid_size-1 and added < auto_steps:
                new_y = y + 1
                visit_count = self.visited_cells.get((x, new_y), 0)
                if visit_count > 0:
                    self.loop_penalty += 2 ** visit_count
                self.completed_path.append("D")
                y = new_y
                self.visited_cells[(x, y)] = visit_count + 1
                self.total_weight += self.weight_grid[y, x]
                added += 1

        # Расчет фитнеса
        distance = (self.grid_size-1 - x) + (self.grid_size-1 - y)

        if distance == 0:
            self.fitness = (
                10000 / (1 + self.total_weight) +
                10000 / (1 + self.loop_penalty)
            )
            self.win = True
        else:
            self.fitness = max(
                0.001,
                1.0 / (1 + distance + self.total_weight) +
                1.0 / (1 + self.loop_penalty)
            )

        return self.fitness

    def mutate(self, mutation_rate: float) -> None:
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                if random.random() < 0.7:
                    self.genes[i] = random.choice(["R", "D"])
                else:
                    self.genes[i] = random.choice(["U", "L"])

    @staticmethod
    def crossover(parent1: 'Animal', parent2: 'Animal') -> 'Animal':
        crossover_point = random.randint(
            1,
            min(len(parent1.genes)-1, len(parent2.genes)-1)
        )
        child_genes = (
            parent1.genes[:crossover_point] +
            parent2.genes[crossover_point:]
        )
        return Animal(
            child_genes,
            parent1.grid_size,
            parent1.max_steps,
            parent1.weight_grid
        )


def generate_initial_population(
    pop_size: int,
    max_steps: int,
    grid_size: int,
    weight_grid: np.ndarray,
    is_start: bool = False
) -> List[Animal]:
    population = []
    if is_start:
        simple_genes = simple_sols(grid_size, weight_grid)
        for genes in simple_genes:
            population.append(Animal(genes, grid_size, max_steps, weight_grid))
        return population

    for _ in range(pop_size):
        genes = [
            random.choice(["R", "D"])
            if random.random() < 0.6
            else random.choice(["U", "L"])
            for _ in range(max_steps)
        ]
        population.append(Animal(genes, grid_size, max_steps, weight_grid))
    return population


def plot_path(
    animal: Animal,
    weight_grid: np.ndarray,
    title: str,
    generation: int
):
    plt.figure(figsize=(10, 10))
    grid_size = animal.grid_size

    plt.imshow(weight_grid, cmap='Blues', alpha=0.6)
    plt.colorbar(label='Вес клеток')

    x, y = 0, 0
    path_x = [x]
    path_y = [y]
    revisited_cells = set()

    for step in animal.completed_path:
        if step == "U":
            y -= 1
        elif step == "D":
            y += 1
        elif step == "L":
            x -= 1
        else:
            x += 1

        if (x, y) in zip(path_x[:-1], path_y[:-1]):
            revisited_cells.add((x, y))

        path_x.append(x)
        path_y.append(y)

    plt.plot(path_x, path_y, 'b-', linewidth=2)

    if revisited_cells:
        revisit_x, revisit_y = zip(*revisited_cells)
        plt.scatter(
            revisit_x,
            revisit_y,
            c='red',
            s=50,
            marker='o',
            label='Заново посещённые клетки'
        )

    plt.scatter([0], [0], c='green', s=100, marker='s', label='Старт')
    plt.scatter(
        [grid_size-1],
        [grid_size-1],
        c='red', s=100,
        marker='s',
        label='Финиш'
    )

    plt.title(
        (
            f"{title}\n"
            f"Вес: {animal.total_weight:.1f}\n"
            f"Штраф: {animal.loop_penalty:.1f}"
        )
    )
    plt.legend()
    plt.grid(True)
    plt.xlim(-0.5, grid_size-0.5)
    plt.ylim(-0.5, grid_size-0.5)
    plt.gca().invert_yaxis()

    filename = f"./output/path_gen_{generation}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Путь сохранен: {filename}")


def tournament_selection(population, tournament_size=3):
    candidates = random.sample(population, tournament_size)
    return max(candidates, key=lambda x: x.fitness)


def genetic_algorithm(
    grid_size: int = 10,
    population_size: int = 200,
    max_generations: int = 100,
    mutation_rate: float = 0.07,
    elitism_ratio: float = 0.2,
    max_steps: int = None,
    weight_type: str = "random",
    early_stopping_patience: int | None = None
) -> Animal:
    weight_grid = generate_weight_grid(grid_size, weight_type)

    if max_steps is None:
        max_steps = grid_size * grid_size

    population = generate_initial_population(
        population_size,
        max_steps,
        grid_size,
        weight_grid,
        is_start=True
    )
    best = None
    history = []
    optimal_steps = 2 * (grid_size - 1)

    best_fitness = -float('inf')
    generations_without_improvement = 0

    for generation in range(max_generations):
        for animal in population:
            animal.calculate_fitness(generation)

        population.sort(key=lambda x: x.fitness, reverse=True)
        current_best = population[0]

        if current_best.fitness > best_fitness:
            best_fitness = current_best.fitness
            generations_without_improvement = 0
            best = current_best
            history.append((generation, best))
        else:
            generations_without_improvement += 1

        if generation % 10 == 0:
            print(f"\nПоколение {generation}:")
            print(f"Лучший фитнес: {best.fitness:.2f}")
            print(f"Шаги: {len(best.completed_path)}")
            print(f"Вес: {best.total_weight:.1f}")
            print(f"Штраф за петлю: {best.loop_penalty:.1f}")

            if generation % 50 == 0:
                plot_path(
                    best,
                    weight_grid,
                    (
                        f"Поколение: {generation}\n"
                        f"Шагов: {len(best.completed_path)}"
                    ),
                    generation
                )

        if (
            early_stopping_patience
            and generations_without_improvement >= early_stopping_patience
        ):
            print(f"\nОстановка на поколении {generation}: нет улучшений")
            history.append((generation, best))
            break

        elite_size = int(elitism_ratio * population_size)
        new_population = population[:elite_size]

        while len(new_population) < population_size:
            parents = [tournament_selection(population) for _ in range(2)]
            child = Animal.crossover(parents[0], parents[1])
            child.mutate(mutation_rate)
            new_population.append(child)

        population = new_population

    print("\n##### Результат #####")
    print(f"Лучшее решение в поколении {history[-1][0]}")
    print(f"Фитнес: {best.fitness:.2f}")
    print(f"Шаги: {len(best.completed_path)}")
    print(f"Путь: {best.completed_path}")
    print(f"Вес: {best.total_weight:.1f}")
    print(f"Штраф за петлю: {best.loop_penalty:.1f}")

    plot_path(
        best,
        weight_grid,
        f"Итог (Поколение {history[-1][0]})\n"
        f"Шагов: {len(best.completed_path)}/{optimal_steps}\n"
        f"Вес: {best.total_weight:.1f}, Петли: {best.loop_penalty:.1f}",
        history[-1][0]
    )

    return best


if __name__ == "__main__":
    prompt = (
        "Выберите тип генерации сетки:\n"
        "  1 - Случайные веса\n"
        "  2 - Плавные случайные изменения\n"
        "  3 - Зоны с разной проходимостью\n"
        "  4 - Плавный градиент весов\n"
        "  5 - Со скрытым оптимальным путем\n"
        "  6 - С блокировкой начальных решений\n"
        "Ваш выбор: "
    )
    weight_type = input(prompt)

    best_solution = genetic_algorithm(
        grid_size=GRID_SIZE,
        population_size=POPULATION_SIZE,
        max_generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        elitism_ratio=ELITISM,
        weight_type=weight_type
    )
