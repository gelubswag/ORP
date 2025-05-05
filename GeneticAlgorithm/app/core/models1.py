import numpy as np
import matplotlib.pyplot as plt
from heapq import heappop, heappush
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation

class Grid:
    def __init__(self, width, height, default_weight=1):
        self.width = width
        self.height = height
        self.grid = np.full((height, width), default_weight, dtype=int)
        
    def set_weight(self, x, y, weight):
        if 0 <= x < self.height and 0 <= y < self.width:
            self.grid[x, y] = weight
            
    def random_generate(self, min_weight=1, max_weight=10):
        self.grid = np.random.randint(
            min_weight, max_weight+1, 
            size=(self.height, self.width)
        )
    
    def get_weight(self, x, y):
        return self.grid[x, y] if (0 <= x < self.height and 0 <= y < self.width) else None
    
    def plot(self, ax, path=None, points=None):
        norm = Normalize(vmin=np.min(self.grid), vmax=np.max(self.grid))
        im = ax.imshow(self.grid, cmap='viridis', origin='upper', norm=norm)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Рисуем сетку
        for i in range(self.height):
            ax.axhline(i - 0.5, color='white', lw=0.5, alpha=0.5)
        for j in range(self.width):
            ax.axvline(j - 0.5, color='white', lw=0.5, alpha=0.5)
            
        # Отображаем специальные точки
        if points:
            for idx, (x, y) in enumerate(points):
                color = 'green' if idx == 0 else ('red' if idx == len(points)-1 else 'blue')
                ax.text(y, x, str(idx), ha='center', va='center', 
                        fontsize=12, color='white', 
                        bbox=dict(facecolor=color, alpha=0.8, edgecolor='black', boxstyle='circle'))

class TravelingSalesmanGA:
    def __init__(self, grid, points, start_point, end_point,
                 pop_size=50, generations=100, mutation_rate=0.1):
        self.grid = grid
        self.points = points.copy()
        self.start_point = start_point
        self.end_point = end_point
        
        if start_point not in self.points:
            self.points.insert(0, start_point)
        if end_point not in self.points:
            self.points.append(end_point)
            
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        self.start_idx = self.points.index(start_point)
        self.end_idx = self.points.index(end_point)
        
        # Кэш путей и стоимостей
        self.cost_matrix, self.path_matrix = self._calculate_path_matrix()
        self.full_path_history = []

    def _dijkstra(self, start, end):
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        heap = [(0, start, [])]
        visited = set()
        
        while heap:
            cost, (x, y), path = heappop(heap)
            if (x, y) == end:
                return cost, path + [(x, y)]
            if (x, y) in visited:
                continue
            visited.add((x, y))
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid.height and 0 <= ny < self.grid.width:
                    new_cost = cost + self.grid.get_weight(nx, ny)
                    new_path = path + [(x, y)]
                    heappush(heap, (new_cost, (nx, ny), new_path))
        return float('inf'), []

    def _calculate_path_matrix(self):
        n = len(self.points)
        cost_matrix = np.zeros((n, n))
        path_matrix = {}
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    cost, path = self._dijkstra(self.points[i], self.points[j])
                    cost_matrix[i][j] = cost
                    path_matrix[(i, j)] = path
        return cost_matrix, path_matrix

    def _create_individual(self):
        middle = [i for i in range(len(self.points)) 
                 if i not in [self.start_idx, self.end_idx]]
        np.random.shuffle(middle)
        return [self.start_idx] + middle + [self.end_idx]

    def _get_full_path(self, individual):
        full_path = []
        for i in range(len(individual)-1):
            from_idx = individual[i]
            to_idx = individual[i+1]
            full_path += self.path_matrix[(from_idx, to_idx)]
        return full_path

    def _calculate_fitness(self, individual):
        """Вычисляем стоимость по порядку точек (индексам), а не по клеткам"""
        return sum(self.cost_matrix[individual[i]][individual[i+1]] 
               for i in range(len(individual)-1))


    def _crossover(self, parent1, parent2):
        middle1 = parent1[1:-1]
        middle2 = parent2[1:-1]
        
        start, end = sorted(np.random.choice(len(middle1), 2, replace=False))
        child1 = middle1[start:end]
        child2 = middle2[start:end]
        
        for p in middle2:
            if p not in child1:
                child1.append(p)
                
        for p in middle1:
            if p not in child2:
                child2.append(p)
                
        return (
            [parent1[0]] + child1 + [parent1[-1]],
            [parent2[0]] + child2 + [parent2[-1]]
        )

    def _mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            middle = individual[1:-1]
            idx1, idx2 = np.random.choice(len(middle), 2, replace=False)
            middle[idx1], middle[idx2] = middle[idx2], middle[idx1]
            return individual[:1] + middle + individual[-1:]
        return individual

    def _animate_path(self, full_path):
        fig, ax = plt.subplots(figsize=(10, 10))
        self.grid.plot(ax, points=self.points)
        
        scat = ax.scatter([], [], c='red', s=100, zorder=3)
        line, = ax.plot([], [], 'r-', lw=2)
        
        def init():
            scat.set_offsets(np.empty((0, 2)))
            line.set_data([], [])
            return scat, line
        
        def update(frame):
            x, y = full_path[frame]
            scat.set_offsets(np.array([[y, x]]))
            
            xs = [p[0] for p in full_path[:frame+1]]
            ys = [p[1] for p in full_path[:frame+1]]
            line.set_data(ys, xs)
            
            return scat, line
        
        ani = FuncAnimation(
            fig, update, frames=len(full_path),
            init_func=init, blit=True, interval=300
        )
        plt.show()

    def _plot_full_path(self, full_path_cells, individual):
        """Принимает два аргумента: путь по клеткам и порядок точек"""
        fig, ax = plt.subplots(figsize=(10, 10))
        self.grid.plot(ax, points=self.points)
        
        # Рисуем весь путь
        xs, ys = zip(*full_path_cells)
        ax.plot(ys, xs, 'r-', lw=2)
        ax.scatter(ys, xs, c='yellow', s=50, edgecolors='red', zorder=2)
        
        # Используем individual для расчета стоимости
        cost = self._calculate_fitness(individual)
        plt.title(f"Полный путь (стоимость: {cost})")
        plt.show()

    def run(self):
        population = [self._create_individual() for _ in range(self.pop_size)]
        
        for gen in range(self.generations):
            fitness = [self._calculate_fitness(ind) for ind in population]
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            
            # Сохраняем и путь по клеткам, и порядок точек
            full_path = self._get_full_path(best_individual)
            self.full_path_history.append( (full_path, best_individual) )
            
            if gen % 10 == 0:
                print(f"Поколение {gen}, Лучшая стоимость: {fitness[best_idx]}")
                self._plot_full_path(full_path, best_individual)  # Передаем оба аргумента
                self._animate_path(full_path)
            
            # Селекция и скрещивание
            selected = []
            for _ in range(self.pop_size):
                candidates = np.random.choice(len(population), 3)
                selected.append(population[min(candidates, key=lambda x: fitness[x])])
            
            new_population = []
            for i in range(0, self.pop_size, 2):
                parent1, parent2 = selected[i], selected[i+1]
                child1, child2 = self._crossover(parent1, parent2)
                new_population.extend([self._mutate(child1), self._mutate(child2)])
            
            population = new_population
        
        return self.full_path_history

if __name__ == "__main__":
    # Создание сетки 10x10 со случайными весами
    grid = Grid(10, 10)
    grid.random_generate(min_weight=1, max_weight=5)
    
    # Задание точек маршрута
    points = [(2,3), (5,7), (8,2), (4,5)]
    start_point = (0,0)
    end_point = (9,9)
    
    # Инициализация алгоритма
    tsp_ga = TravelingSalesmanGA(
        grid=grid,
        points=points,
        start_point=start_point,
        end_point=end_point,
        pop_size=30,
        generations=50,
        mutation_rate=0.2
    )
    
    paths = tsp_ga.run()