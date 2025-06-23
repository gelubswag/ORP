from dataclasses import dataclass
from typing import Dict, List, Optional
import heapq
from collections import defaultdict


@dataclass
class Route:
    """Маршрут между аэропортами"""
    from_airport: str
    to_airport: str
    flight_time: int


@dataclass
class Airplane:
    """Самолет с уникальным идентификатором"""
    id: str
    current_location: str = None
    available_time: int = 0


@dataclass
class ScheduledFlight:
    """Запланированный рейс"""
    airplane_id: str
    from_airport: str
    to_airport: str
    departure_time: int
    arrival_time: int

    def __str__(self):
        return f"{self.airplane_id} | {self.from_airport} | {self.departure_time} | {self.to_airport} | {self.arrival_time}"


class Airport:
    """Класс аэропорта с маршрутами"""
    _map: Dict[str, 'Airport'] = {}

    def __init__(self, name: str, routes: List[Route]):
        self.name = name
        self.routes = routes
        self._map[name] = self

    @classmethod
    def get_airport(cls, name: str) -> Optional['Airport']:
        return cls._map.get(name)


class FlightScheduler:
    """Планировщик рейсов с оптимизацией"""
    def __init__(
        self,
        airports: List[Airport],
        airplanes: List[Airplane],
        central_hub: str,
        simulation_time: int = 1440
            ):  # 24 часа в минутах
        self.airports = {airport.name: airport for airport in airports}
        self.airplanes = airplanes
        self.central_hub = central_hub
        self.simulation_time = simulation_time

        # Ограничения
        self.service_time = 30  # время обслуживания
        self.departure_delta = 5  # минимальная дельта между вылетами
        self.min_flights_per_day = 1  # минимум рейсов в день для каждого аэропорта
        self.max_flights_per_day = 10  # максимум рейсов в день (кроме хаба)

        # Счетчики для отслеживания ограничений
        self.airport_departures = defaultdict(int)
        self.airport_arrivals = defaultdict(int)
        self.last_departure_time = defaultdict(lambda: -float('inf'))

        # Результат
        self.scheduled_flights: List[ScheduledFlight] = []

    def can_schedule_departure(self, airport: str, time: int) -> bool:
        """Проверяет, можно ли запланировать вылет из аэропорта в данное время"""
        # Проверка дельты между вылетами
        if time - self.last_departure_time[airport] < self.departure_delta:
            return False

        # Проверка максимального количества вылетов (кроме центрального хаба)
        if airport != self.central_hub and self.airport_departures[airport] >= self.max_flights_per_day:
            return False

        return True

    def can_schedule_arrival(self, airport: str) -> bool:
        """Проверяет, можно ли запланировать прилет в аэропорт"""
        # Проверка максимального количества прилетов (кроме центрального хаба)
        if airport != self.central_hub and self.airport_arrivals[airport] >= self.max_flights_per_day:
            return False
        return True

    def find_route(self, from_airport: str, to_airport: str) -> Optional[Route]:
        """Находит маршрут между аэропортами"""
        airport = self.airports.get(from_airport)
        if not airport:
            return None

        for route in airport.routes:
            if route.to_airport == to_airport:
                return route
        return None

    def schedule_flight(self, airplane: Airplane, route: Route, departure_time: int) -> Optional[ScheduledFlight]:
        """Планирует рейс для самолета"""
        arrival_time = departure_time + route.flight_time

        # Проверяем ограничения
        if not self.can_schedule_departure(route.from_airport, departure_time):
            return None

        if not self.can_schedule_arrival(route.to_airport):
            return None

        if arrival_time > self.simulation_time:
            return None

        # Создаем рейс
        flight = ScheduledFlight(
            airplane_id=airplane.id,
            from_airport=route.from_airport,
            to_airport=route.to_airport,
            departure_time=departure_time,
            arrival_time=arrival_time
        )

        # Обновляем счетчики и состояние
        self.airport_departures[route.from_airport] += 1
        self.airport_arrivals[route.to_airport] += 1
        self.last_departure_time[route.from_airport] = departure_time

        airplane.current_location = route.to_airport
        airplane.available_time = arrival_time + self.service_time

        return flight

    def optimize_schedule(self) -> List[ScheduledFlight]:
        """Оптимизирует расписание для максимизации количества рейсов"""
        # Используем жадный алгоритм с приоритетной очередью
        # Приоритет отдается самолетам, которые освобождаются раньше

        current_time = 0
        # Добавляем уникальный счетчик для разрешения конфликтов сравнения
        airplane_queue = [(airplane.available_time, i, airplane) for i, airplane in enumerate(self.airplanes)]
        heapq.heapify(airplane_queue)
        counter = len(self.airplanes)  # Счетчик для новых элементов в очереди

        while airplane_queue and current_time < self.simulation_time:
            # Берем самолет, который освобождается раньше всех
            available_time, _, airplane = heapq.heappop(airplane_queue)
            current_time = max(current_time, available_time)

            if current_time >= self.simulation_time:
                break

            # Ищем лучший маршрут для этого самолета
            best_flight = None
            best_departure_time = None

            current_airport = airplane.current_location
            if not current_airport:
                # Если местоположение не определено, начинаем с центрального хаба
                current_airport = self.central_hub
                airplane.current_location = current_airport

            airport = self.airports.get(current_airport)
            if not airport:
                continue

            # Перебираем все возможные маршруты
            for route in airport.routes:
                # Находим ближайшее возможное время вылета
                departure_time = max(current_time, available_time)

                # Выравниваем по дельте между вылетами
                last_dep = self.last_departure_time[route.from_airport]
                if last_dep != -float('inf'):
                    departure_time = max(departure_time, last_dep + self.departure_delta)

                # Пытаемся запланировать рейс
                flight = self.schedule_flight(airplane, route, departure_time)
                if flight:
                    best_flight = flight
                    break  # Берем первый подходящий рейс (жадный подход)

            if best_flight:
                self.scheduled_flights.append(best_flight)
                # Возвращаем самолет в очередь с новым временем доступности
                counter += 1
                heapq.heappush(airplane_queue, (airplane.available_time, counter, airplane))
            else:
                # Если не можем запланировать рейс, увеличиваем время
                current_time += 1
                if current_time < self.simulation_time:
                    airplane.available_time = current_time
                    counter += 1
                    heapq.heappush(airplane_queue, (current_time, counter, airplane))

        return self.scheduled_flights

    def print_schedule(self):
        """Выводит расписание рейсов"""
        print("Расписание рейсов:")
        print("Исполнитель(рейс) | Аэропорт отправления | Время отправления | Аэр. приб. | Время приб.")
        print("-" * 90)

        for flight in sorted(self.scheduled_flights, key=lambda f: f.departure_time):
            print(flight)

        print(f"\nВсего рейсов: {len(self.scheduled_flights)}")

        # Статистика по аэропортам
        print("\nСтатистика по аэропортам:")
        for airport_name in self.airports.keys():
            departures = self.airport_departures[airport_name]
            arrivals = self.airport_arrivals[airport_name]
            print(f"{airport_name}: {departures} вылетов, {arrivals} прилетов")

    def validate_constraints(self) -> bool:
        """Проверяет соблюдение всех ограничений"""
        valid = True

        # Проверяем минимальное количество рейсов для каждого аэропорта
        for airport_name in self.airports.keys():
            if airport_name != self.central_hub:
                total_flights = self.airport_departures[airport_name] + self.airport_arrivals[airport_name]
                if total_flights < self.min_flights_per_day:
                    print(f"ОШИБКА: Аэропорт {airport_name} имеет меньше минимального количества рейсов")
                    valid = False

        return valid


def create_example_network():
    """Создает пример сети аэропортов"""

    # Определяем маршруты
    routes_moscow = [
        Route("Москва", "СПб", 90),
        Route("Москва", "Казань", 120),
        Route("Москва", "Екатеринбург", 180),
        Route("Москва", "Новосибирск", 240),
    ]

    routes_spb = [
        Route("СПб", "Москва", 90),
        Route("СПб", "Казань", 150),
    ]

    routes_kazan = [
        Route("Казань", "Москва", 120),
        Route("Казань", "СПб", 150),
        Route("Казань", "Екатеринбург", 120),
    ]

    routes_ekb = [
        Route("Екатеринбург", "Москва", 180),
        Route("Екатеринбург", "Казань", 120),
        Route("Екатеринбург", "Новосибирск", 120),
    ]

    routes_nsk = [
        Route("Новосибирск", "Москва", 240),
        Route("Новосибирск", "Екатеринбург", 120),
    ]

    # Создаем аэропорты
    airports = [
        Airport("Москва", routes_moscow),
        Airport("СПб", routes_spb),
        Airport("Казань", routes_kazan),
        Airport("Екатеринбург", routes_ekb),
        Airport("Новосибирск", routes_nsk),
    ]

    # Создаем самолеты
    airplanes = [
        Airplane("SU001", "Москва"),
        Airplane("SU002", "Москва"),
        Airplane("SU003", "СПб"),
        Airplane("SU004", "Казань"),
        Airplane("SU005", "Москва"),
    ]

    return airports, airplanes


if __name__ == "__main__":
    # Создаем пример сети
    airports, airplanes = create_example_network()

    # Создаем планировщик
    scheduler = FlightScheduler(
        airports=airports,
        airplanes=airplanes,
        central_hub="Москва",
        simulation_time=1440  # 24 часа
    )

    # Оптимизируем расписание
    schedule = scheduler.optimize_schedule()

    # Выводим результат
    scheduler.print_schedule()

    # Проверяем ограничения
    if scheduler.validate_constraints():
        print("\nВсе ограничения соблюдены")
    else:
        print("\nНекоторые ограничения нарушены")
