import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

class Grid:
    def __init__(self, width, height, default_weight=1):
        self.width = width
        self.height = height
        self.grid = np.full((height, width), default_weight, dtype=int)
        
    def random_generate(self, min_weight=1, max_weight=10):
        self.grid = np.random.randint(
            min_weight, max_weight + 1,
            size=(self.height, self.width)
        )
    def set_weight(self, x, y, weight):
        if 0 <= x < self.height and 0 <= y < self.width:
            self.grid[x, y] = weight
            
    def get_weight(self, x, y):
        return self.grid[x, y]
    
    def plot(self, ax, points=None):
        norm = Normalize(vmin=np.min(self.grid), vmax=np.max(self.grid))
        im = ax.imshow(self.grid, cmap='viridis', origin='upper', norm=norm)
        plt.colorbar(im, ax=ax)
        
        if points:
            for idx, (x, y) in enumerate(points):
                color = 'green' if idx == 0 else ('red' if idx == len(points)-1 else 'blue')
                ax.text(y, x, str(idx), ha='center', va='center', 
                        color='white', bbox=dict(facecolor=color, boxstyle='circle'))

class TravelingSalesmanGA:
    def __init__(self, grid, points, start_point, end_point,
                 pop_size=30, generations=30, mutation_rate=0.2):
        self.grid = grid
        self.points = self._validate_points(start_point, end_point, points)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        self.cost_matrix = self._calculate_cost_matrix()
        self.best_history = []
        self.animation_history = []
        self._validate_parameters()

    def _validate_parameters(self):
        if len(self.points) < 2:
            raise ValueError("Need at least 2 points (start and end)")
        if self.pop_size < 4:
            raise ValueError("Population size must be at least 4")

    def _validate_points(self, start, end, points):
        points = points.copy()
        if start not in points:
            points.insert(0, start)
        if end not in points:
            points.append(end)
        return points

    def _calculate_segment_cost(self, start, end):
        x1, y1 = start
        x2, y2 = end
        cost = 0
        
        dx = 1 if x2 > x1 else -1
        dy = 1 if y2 > y1 else -1
        
        # Horizontal movement
        for x in range(x1, x2, dx):
            cost += self.grid.get_weight(x, y1)
        
        # Vertical movement
        for y in range(y1, y2 + dy, dy):
            cost += self.grid.get_weight(x2, y)
            
        return cost

    def _calculate_cost_matrix(self):
        n = len(self.points)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = self._calculate_segment_cost(
                        self.points[i], 
                        self.points[j]
                    )
        return matrix

    def _create_individual(self):
        middle = list(range(1, len(self.points)-1))
        np.random.shuffle(middle)
        return [0] + middle + [len(self.points)-1]

    def _pmx_crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(np.random.choice(size, 2, replace=False))
        
        # Create mapping tables
        map1 = {parent1[i]: parent2[i] for i in range(start, end+1)}
        map2 = {parent2[i]: parent1[i] for i in range(start, end+1)}
        
        child1, child2 = parent1.copy(), parent2.copy()
        
        # Apply mapping
        for i in list(range(0, start)) + list(range(end+1, size)):
            while child1[i] in map1:
                child1[i] = map1[child1[i]]
            while child2[i] in map2:
                child2[i] = map2[child2[i]]
                
        return child1, child2

    def _mutate(self, individual):
        if np.random.rand() < self.mutation_rate and len(individual) > 4:
            idx1, idx2 = np.random.choice(len(individual)-2, 2, replace=False) + 1
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def run(self):
        population = [self._create_individual() for _ in range(self.pop_size)]
        
        for gen in range(self.generations):
            fitness = np.array([self._calculate_fitness(ind) for ind in population])
            best_idx = np.argmin(fitness)
            
            # Save best solution
            self.best_history.append({
                'gen': gen,
                'fitness': fitness[best_idx],
                'individual': population[best_idx].copy()
            })
            
            # Save animation data
            try:
                path = self._generate_full_path(population[best_idx])
                self.animation_history.append({
                    'gen': gen,
                    'path': path,
                    'cost': fitness[best_idx]
                })
            except Exception as e:
                print(f"Error saving frame {gen}: {str(e)}")
            
            # Selection
            elite = [population[i].copy() for i in fitness.argsort()[:2]]
            
            # Roulette wheel selection
            probs = 1 / (fitness + 1e-8)
            probs /= probs.sum()
            
            selected_idx = np.random.choice(
                len(population), 
                size=self.pop_size-2,
                p=probs
            )
            selected = [population[i].copy() for i in selected_idx]
            
            # Create new population
            new_pop = elite.copy()
            for i in range(0, len(selected), 2):
                if i+1 >= len(selected):
                    new_pop.append(selected[i])
                    continue
                
                p1, p2 = selected[i], selected[i+1]
                c1, c2 = self._pmx_crossover(p1, p2)
                new_pop += [self._mutate(c1), self._mutate(c2)]
            
            population = new_pop[:self.pop_size]
        
        return self.best_history

    def _calculate_fitness(self, individual):
        total = 0
        for i in range(len(individual)-1):
            total += self.cost_matrix[individual[i]][individual[i+1]]
        return total

    def _generate_full_path(self, individual):
        """Generate full path coordinates with error handling"""
        full_path = []
        try:
            for i in range(len(individual)-1):
                from_pt = self.points[individual[i]]
                to_pt = self.points[individual[i+1]]
                
                # Horizontal movement
                step_x = 1 if to_pt[0] > from_pt[0] else -1
                for x in range(from_pt[0], to_pt[0], step_x):
                    full_path.append((x, from_pt[1]))
                
                # Vertical movement
                step_y = 1 if to_pt[1] > from_pt[1] else -1
                for y in range(from_pt[1], to_pt[1] + step_y, step_y):
                    full_path.append((to_pt[0], y))
        except Exception as e:
            print(f"Path generation error: {str(e)}")
            return []
        return full_path

    def animate_solution(self):
        """Create evolution animation"""
        if not self.animation_history:
            print("No animation data available")
            return

        fig, ax = plt.subplots(figsize=(10, 10))
        self.grid.plot(ax, points=self.points)
        
        line, = ax.plot([], [], 'ro-', markersize=8, linewidth=2)
        title = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center")
        
        # Filter valid frames
        valid_frames = [f for f in self.animation_history if f['path']]
        
        def update(frame):
            data = valid_frames[frame]
            xs, ys = zip(*data['path']) if data['path'] else ([], [])
            line.set_data(ys, xs)
            title.set_text(f"Generation: {data['gen']} Cost: {data['cost']:.1f}")
            return line, title
        
        ani = FuncAnimation(
            fig, update, frames=len(valid_frames),
            init_func=lambda: (line.set_data([], [])), 
            interval=200, blit=True
        )
        
        plt.show()
        return ani

    def visualize_solution(self, solution):
        """Visualize final solution"""
        fig, ax = plt.subplots(figsize=(10, 10))
        self.grid.plot(ax, points=self.points)
        
        path = [self.points[i] for i in solution['individual']]
        xs, ys = zip(*path)
        ax.plot(ys, xs, 'ro-', markersize=8, linewidth=2)
        
        plt.title(f"Best Solution (Cost: {solution['fitness']:.1f})")
        plt.show()

if __name__ == "__main__":
    try:
        # Initialize grid and solver
        grid = Grid(10, 10)
        grid.random_generate(min_weight=1, max_weight=5)
        
        tsp_solver = TravelingSalesmanGA(
            grid=grid,
            points=[(2,3), (7,5), (5,8)],
            start_point=(0,0),
            end_point=(9,9),
            pop_size=30,
            generations=50,
            mutation_rate=0.25
        )
        
        # Run algorithm
        results = tsp_solver.run()
        
        # Show animation
        tsp_solver.animate_solution()
        
        # Show final result
        best_solution = min(results, key=lambda x: x['fitness'])
        print("\nBest solution found:")
        print(f"Generation: {best_solution['gen']}")
        print(f"Cost: {best_solution['fitness']:.1f}")
        print(f"Route: {best_solution['individual']}")
        
        tsp_solver.visualize_solution(best_solution)
        
    except Exception as e:
        print(f"Runtime error: {str(e)}")