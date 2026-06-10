"""
Simulación de Tráfico - Av. Luis Elizondo
Onda Verde en tres cruces — AgentPy + Matplotlib Animation

Escala: 1 unidad ≈ 100 metros reales  (4 m reales = 0.04 unidades)

García Roel — lógica detallada:
  N→S outer (x≈-3.616): sigue recto, cruza Elizondo hacia y=-1
  N→S inner (x≈-3.578): gira izquierda → se incorpora a Elizondo al ESTE (carril derecho)
  S→N left  (x≈-3.541): sigue recto hacia y=+1
  S→N right (x≈-3.504): gira derecha → se incorpora a Elizondo al ESTE (carril izquierdo)

Elizondo desde x=-4: puede girar S (cruce1_south), N (cruce1_north), o seguir recto.
  cruce1_south → usa carril derecho (LANE_RIGHT),  gira al inner N→S de García Roel
  cruce1_north → usa carril izquierdo (LANE_LEFT), gira al carril S→N left de García Roel
"""

import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import random
import os

# ══════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════

SPEED            = 0.025
MIN_GAP          = 0.06
LANE_CHANGE_RATE = 0.004
ANTICIPATION     = 0.40

LANE_RIGHT  = -0.045
LANE_CENTER =  0.000
LANE_LEFT   =  0.045
LANES_ELIZONDO = [LANE_RIGHT, LANE_CENTER, LANE_LEFT]

X_CRUCE1 = -3.56
X_CRUCE2 =  0.00
X_CRUCE3 =  4.69
INTER_HALF = 0.12

GREEN_DURATION = 80
RED_DURATION   = 60
CYCLE          = GREEN_DURATION + RED_DURATION

OFFSET_1_TO_2 = int((X_CRUCE2 - X_CRUCE1) / SPEED)
OFFSET_2_TO_3 = int((X_CRUCE3 - X_CRUCE2) / SPEED)

X_ENTRY = -4.0
X_EXIT  =  5.0

# ── Centros de carriles García Roel ──
GR_NS_INNER_X = -3.578   # N→S inner: gira izq → Elizondo
GR_NS_OUTER_X = -3.616   # N→S outer: sigue recto
GR_SN_RIGHT_X = -3.504   # S→N right: gira der → Elizondo
GR_SN_LEFT_X  = -3.541   # S→N left:  sigue recto

# ── Definición completa de carriles transversales ──
# action:
#   'straight'              → cruza Elizondo y sigue
#   'turn_to_elizondo_east' → se incorpora a Elizondo yendo al ESTE
TRANSVERSAL_LANES = {
    # ── García Roel ──
    'gr_ns_inner': {   # N→S, gira izq → Elizondo ESTE
        'x': GR_NS_INNER_X, 'dir': 'south',
        'cruce': X_CRUCE1,  'y_start': 1.0,
        'action': 'turn_to_elizondo_east',
        'join_lane': LANE_RIGHT,   # entra a Elizondo por carril derecho
    },
    'gr_ns_outer': {   # N→S, sigue recto
        'x': GR_NS_OUTER_X, 'dir': 'south',
        'cruce': X_CRUCE1,  'y_start': 1.0,
        'action': 'straight',
    },
    'gr_sn_right': {   # S→N, gira der → Elizondo ESTE
        'x': GR_SN_RIGHT_X, 'dir': 'north',
        'cruce': X_CRUCE1,  'y_start': -1.0,
        'action': 'turn_to_elizondo_east',
        'join_lane': LANE_LEFT,    # entra a Elizondo por carril izquierdo
    },
    'gr_sn_left': {    # S→N, sigue recto
        'x': GR_SN_LEFT_X, 'dir': 'north',
        'cruce': X_CRUCE1,  'y_start': -1.0,
        'action': 'straight',
    },
    # ── Junco de la Vega ──
    # N→S: auto de Elizondo gira derecha → baja por x=-0.030 (entre -0.045 y -0.015)
    'junco_ns_1': {
        'x': -0.030, 'dir': 'south',
        'cruce': X_CRUCE2, 'y_start': -0.07,  # arranca justo bajo Elizondo
        'action': 'straight',                  # sale por y=-1
    },
    # S→N: dos carriles entre x=-0.015 y x=0.045, ambos se incorporan a Elizondo
    'junco_sn_1': {
        'x':  0.015, 'dir': 'north',
        'cruce': X_CRUCE2, 'y_start': -1.0,
        'action': 'turn_to_elizondo_east',
        'join_lane': LANE_LEFT,
    },
    'junco_sn_2': {
        'x':  0.040, 'dir': 'north',           # centro entre -0.015 y 0.045 → ≈0.040
        'cruce': X_CRUCE2, 'y_start': -1.0,
        'action': 'turn_to_elizondo_east',
        'join_lane': LANE_LEFT,
    },
    # ── Garza Sada ──
    'garza_ns_1': {
        'x':  4.665, 'dir': 'south',
        'cruce': X_CRUCE3, 'y_start': 1.0,
        'action': 'straight',
    },
}
# junco_ns_1 NO se usa para spawn: solo lo crean autos que giran desde Elizondo
TRANSVERSAL_IDS = [k for k in TRANSVERSAL_LANES if k != 'junco_ns_1']

# Destinos de Elizondo  (cruce1_south=gira S, cruce1_north=gira N, cruce2, cruce3, exit)
ELIZ_DESTS   = ['cruce1_south', 'cruce1_north', 'cruce2', 'cruce3', 'exit']
ELIZ_WEIGHTS = [0.12, 0.12, 0.22, 0.18, 0.36]


# ══════════════════════════════════════════════════
#  Q-LEARNING PARA APRENDER OFFSETS DE ONDA VERDE
# ══════════════════════════════════════════════════

# En este modelo el agente aprende una configuración global de offsets:
#   acción = (offset_cruce1, offset_cruce2, offset_cruce3)
# El primer cruce se mantiene en 0 y se aprenden los offsets de los cruces 2 y 3.
# Los offsets están en pasos de simulación, igual que GREEN_DURATION, RED_DURATION y CYCLE.

N_CROSSES = 3
PLATOON_SIZE = 40
PLATOON_HEADWAY = 2.0

CROSS_POSITIONS = np.array([X_CRUCE1, X_CRUCE2, X_CRUCE3], dtype=float)
CROSS_NAMES = ['Cruce 1 (García Roel)', 'Cruce 2 (Junco)', 'Cruce 3 (Garza Sada)']
IDEAL_ARRIVAL_STEPS = np.array([
    0,
    OFFSET_1_TO_2,
    OFFSET_1_TO_2 + OFFSET_2_TO_3,
], dtype=float)

# Opciones discretas para que Q-learning pueda explorar un espacio finito.
# Con paso de 5, el agente puede aproximar mejor el offset ideal sin hacer lento el entrenamiento.
OFFSET_OPTIONS = list(range(0, CYCLE, 5))
ACTIONS = [(0, o2, o3) for o2 in OFFSET_OPTIONS for o3 in OFFSET_OPTIONS]


def is_green_for_elizondo_at(t, offset, cycle=CYCLE, green_duration=GREEN_DURATION):
    """Determina si Elizondo está en verde en el tiempo global t con cierto offset."""
    local_time = (t - offset) % cycle
    return 0 <= local_time < green_duration


def theoretical_offsets():
    """Offsets ideales aproximados según tiempo de viaje entre cruces."""
    return tuple(int(round(v % CYCLE)) for v in IDEAL_ARRIVAL_STEPS)


def simulate_platoon_offsets(offsets, platoon_size=PLATOON_SIZE, headway=PLATOON_HEADWAY):
    """
    Simula un pelotón Oeste→Este y evalúa si cruza los 3 semáforos en verde.

    Retorna:
      - vehículos que pasaron los 3 cruces en verde
      - paradas totales
      - índice de coordinación
    """
    vehicles_success = 0
    total_stops = 0
    green_crossings = 0

    for car_id in range(platoon_size):
        start_time = car_id * headway
        car_success = True

        for i in range(N_CROSSES):
            arrival_time = start_time + IDEAL_ARRIVAL_STEPS[i]
            if is_green_for_elizondo_at(arrival_time, offsets[i]):
                green_crossings += 1
            else:
                total_stops += 1
                car_success = False

        if car_success:
            vehicles_success += 1

    coordination_index = green_crossings / (platoon_size * N_CROSSES)
    return vehicles_success, total_stops, coordination_index


def q_reward(success, stops, coordination_index):
    """Misma estructura de recompensa del notebook, ajustada al corredor de esta simulación."""
    return 10 * success - 2 * stops + 50 * coordination_index


def choose_q_action(Q, epsilon):
    """Política epsilon-greedy."""
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    best_value = max(Q.values())
    best_actions = [a for a, q in Q.items() if q == best_value]
    return random.choice(best_actions)


def train_q_learning(episodes=800, alpha=0.15, gamma=0.90, epsilon=0.30, epsilon_decay=0.995, min_epsilon=0.05, seed=42):
    """
    Entrena Q-learning para escoger los offsets de onda verde.

    Aunque se declara gamma para mantener la forma estándar de Q-learning,
    este problema se modela como una sola decisión global por episodio.
    """
    random.seed(seed)
    np.random.seed(seed)

    Q = {a: 0.0 for a in ACTIONS}
    reward_history = []
    success_history = []
    stop_history = []
    coordination_history = []
    best_action_history = []
    epsilon_history = []

    for ep in range(episodes):
        action = choose_q_action(Q, epsilon)
        success, stops, coord_index = simulate_platoon_offsets(action)
        reward = q_reward(success, stops, coord_index)

        old_q = Q[action]
        Q[action] = old_q + alpha * (reward - old_q)

        reward_history.append(reward)
        success_history.append(success)
        stop_history.append(stops)
        coordination_history.append(coord_index)
        best_action_history.append(max(Q, key=Q.get))
        epsilon_history.append(epsilon)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    best_offsets = max(Q, key=Q.get)

    return {
        'Q': Q,
        'best_offsets': best_offsets,
        'reward_history': reward_history,
        'success_history': success_history,
        'stop_history': stop_history,
        'coordination_history': coordination_history,
        'best_action_history': best_action_history,
        'epsilon_history': epsilon_history,
        'episodes': episodes,
        'alpha': alpha,
        'gamma': gamma,
    }


def evaluate_offsets(label, offsets):
    success, stops, coord_index = simulate_platoon_offsets(offsets)
    return {
        'label': label,
        'offsets': tuple(offsets),
        'successful_vehicles': success,
        'stops': stops,
        'coordination_index': coord_index,
        'reward': q_reward(success, stops, coord_index),
    }


# ══════════════════════════════════════════════════
#  SEMÁFORO
# ══════════════════════════════════════════════════

class TrafficLight(ap.Agent):
    def setup(self, x_pos=0.0, offset=0):
        self.x = x_pos
        # El cruce N debe ponerse en verde exactamente cuando llega el auto,
        # es decir en el step = offset desde t=0.
        # El verde empieza cuando timer % CYCLE == 0, así que el timer inicial
        # debe ser CYCLE - (offset % CYCLE) para que el primer verde coincida
        # con la llegada del vehículo.
        self.timer = (CYCLE - (offset % CYCLE)) % CYCLE
        self._recalc()

    def _recalc(self):
        self.elizondo_green = (self.timer % CYCLE) < GREEN_DURATION

    def step(self):
        self.timer += 1
        self._recalc()

    def is_green_for_elizondo(self):    return self.elizondo_green
    def is_green_for_transversal(self): return not self.elizondo_green


# ══════════════════════════════════════════════════
#  AUTO
# ══════════════════════════════════════════════════

class Car(ap.Agent):

    def setup(self, x=0.0, y=0.0, destination='exit',
              road='elizondo', direction='east', lane_id=None):
        self.x           = x
        self.y           = y
        self.target_y    = y
        self.destination = destination
        self.speed       = SPEED
        self.active      = True
        self.waiting     = False
        self.road        = road
        self.direction   = direction
        self.lane_id     = lane_id
        self.turned      = False

    # ── Planificación de carril (Elizondo) ──

    def _dest_x(self):
        return {
            'cruce1_south': X_CRUCE1, 'cruce1_north': X_CRUCE1,
            'cruce2': X_CRUCE2, 'cruce3': X_CRUCE3, 'exit': X_EXIT,
        }.get(self.destination, X_EXIT)

    def _plan_lane(self):
        if self.road != 'elizondo':
            return
        dx   = self._dest_x()
        dist = dx - self.x
        if 0 < dist < ANTICIPATION:
            if self.destination == 'cruce1_north':
                self.target_y = LANE_LEFT      # giro izq → carril izquierdo
            elif self.destination in ('cruce1_south', 'cruce2', 'cruce3'):
                self.target_y = LANE_RIGHT     # giro der → carril derecho
        if self.x > dx + INTER_HALF and self.destination != 'exit':
            self.destination = 'exit'

    def _change_lane(self):
        diff = self.target_y - self.y
        if abs(diff) < LANE_CHANGE_RATE:
            self.y = self.target_y
        else:
            self.y += LANE_CHANGE_RATE * np.sign(diff)

    # ── Colisiones ──

    def _blocked_ahead(self, all_cars):
        for o in all_cars:
            if o is self or not o.active:
                continue
            if self.road == 'elizondo':
                if o.road != 'elizondo':
                    continue
                if abs(o.y - self.y) < 0.03 and 0 < (o.x - self.x) < MIN_GAP:
                    return True
            else:
                if o.road != 'transversal':
                    continue
                if abs(o.x - self.x) > 0.03:
                    continue
                if self.direction == 'north' and 0 < (o.y - self.y) < MIN_GAP:
                    return True
                if self.direction == 'south' and 0 < (self.y - o.y) < MIN_GAP:
                    return True
        return False

    # ── Semáforos ──

    def _red_elizondo(self, lights):
        """Auto en Elizondo: rojo si hay semáforo cercano en rojo para Elizondo."""
        for light in lights:
            dist = light.x - self.x
            if 0 < dist < MIN_GAP * 2.0 and not light.is_green_for_elizondo():
                return True
        return False

    def _red_transversal(self, lights):
        """Auto transversal: rojo si está cerca de cruzar Elizondo y transversal en rojo."""
        info = TRANSVERSAL_LANES.get(self.lane_id, {})
        cruce_x = info.get('cruce', 999)
        for light in lights:
            if abs(light.x - cruce_x) > 0.1:
                continue
            # distancia en Y hasta la zona de Elizondo (y ≈ 0)
            if self.direction == 'north':
                dist_to_cross = -self.y          # cuánto falta para llegar a y=0
            else:
                dist_to_cross = self.y           # cuánto falta (viene desde +1)
            if 0 < dist_to_cross < MIN_GAP * 2.5 and not light.is_green_for_transversal():
                return True
        return False

    # ── Maniobra de incorporación a Elizondo ──

    def _try_join_elizondo(self, info):
        """Intenta incorporarse a Elizondo cuando el auto transversal llega a la zona."""
        join_y = info.get('join_lane', LANE_LEFT)
        if self.direction == 'north':
            arrived = self.y >= join_y - SPEED * 1.5
        else:
            arrived = self.y <= join_y + SPEED * 1.5
        if arrived:
            self.road        = 'elizondo'
            self.y           = join_y
            self.target_y    = join_y
            self.direction   = 'east'
            self.destination = 'exit'
            self.turned      = True

    # ── Step principal ──

    def step_move(self, all_cars, lights):
        if not self.active:
            return

        # ╔══════════════════════════╗
        # ║  TRANSVERSAL             ║
        # ╚══════════════════════════╝
        if self.road == 'transversal':
            info   = TRANSVERSAL_LANES.get(self.lane_id, {})
            action = info.get('action', 'straight')

            blocked = self._blocked_ahead(all_cars)
            red     = self._red_transversal(lights)
            self.waiting = blocked or red

            if not self.waiting:
                self.y += self.speed if self.direction == 'north' else -self.speed

            # ── Maniobra en zona de cruce ──
            if not self.turned:
                cruce_x = info.get('cruce', 999)
                near    = abs(self.x - cruce_x) < INTER_HALF * 1.8

                if near and action == 'turn_to_elizondo_east':
                    self._try_join_elizondo(info)

                # 'straight': nada especial, el auto simplemente cruza

            # Salir del mapa
            if self.y > 1.15 or self.y < -1.15:
                self.active = False
            return

        # ╔══════════════════════════╗
        # ║  ELIZONDO                ║
        # ╚══════════════════════════╝
        self._plan_lane()
        self._change_lane()

        blocked = self._blocked_ahead(all_cars)
        red     = self._red_elizondo(lights)
        self.waiting = blocked or red

        if not self.waiting:
            self.x += self.speed

        # ── Giros desde Elizondo ──
        if not self.turned:
            dx      = self._dest_x()
            at_turn = abs(self.x - dx) < SPEED * 2.5

            # Giro a la derecha → SUR  (cruce1_south, cruce2, cruce3)
            if self.destination in ('cruce1_south', 'cruce2', 'cruce3') and at_turn:
                if abs(self.y - LANE_RIGHT) < 0.025:
                    for light in lights:
                        if abs(light.x - dx) < 0.05 and light.is_green_for_elizondo():
                            self.road      = 'transversal'
                            self.direction = 'south'
                            self.turned    = True
                            if self.destination == 'cruce1_south':
                                self.x       = GR_NS_INNER_X
                                self.lane_id = 'gr_ns_inner'
                            elif self.destination == 'cruce2':
                                self.x       = -0.030          # carril N→S de Junco
                                self.y       = LANE_RIGHT - 0.02  # justo bajo Elizondo
                                self.lane_id = 'junco_ns_1'
                            else:
                                self.x       = 4.665
                                self.lane_id = 'garza_ns_1'
                            break

            # Giro a la izquierda → NORTE en García Roel (cruce1_north)
            elif self.destination == 'cruce1_north' and at_turn:
                if abs(self.y - LANE_LEFT) < 0.025:
                    for light in lights:
                        if abs(light.x - X_CRUCE1) < 0.05 and light.is_green_for_elizondo():
                            self.road      = 'transversal'
                            self.direction = 'north'
                            self.turned    = True
                            self.x         = GR_SN_LEFT_X
                            self.lane_id   = 'gr_sn_left'
                            # Empieza justo arriba de Elizondo
                            self.y         = LANE_LEFT + 0.02
                            break

        if self.x > X_EXIT:
            self.active = False


# ══════════════════════════════════════════════════
#  MODELO
# ══════════════════════════════════════════════════

class TrafficModel(ap.Model):

    def setup(self):
        # Semáforos
        # q_offsets permite usar los offsets aprendidos por Q-learning.
        # Si no se pasan, se usa la onda verde teórica original.
        q_offsets = self.p.get('q_offsets', theoretical_offsets())
        for_lights = [
            (X_CRUCE1, q_offsets[0]),
            (X_CRUCE2, q_offsets[1]),
            (X_CRUCE3, q_offsets[2]),
        ]
        light_objs = []
        for xp, off in for_lights:
            tl = TrafficLight(self)
            tl.setup(x_pos=xp, offset=off)
            light_objs.append(tl)
        self.lights = ap.AgentList(self, light_objs)

        # Autos
        n = self.p.get('n_cars', 32)
        car_objs = [Car(self) for _ in range(n)]
        self.cars = ap.AgentList(self, car_objs)

        for i, car in enumerate(self.cars):
            if i < 3:
                # Un auto por cada carril en la entrada x=-4
                self._spawn_entry(car, LANES_ELIZONDO[i])
            elif i < int(n * 0.62):
                self._spawn_elizondo(car)
            else:
                self._spawn_transversal(car)

        self.history = []
        self._record()

    # ── Spawns ──

    def _spawn_entry(self, car, lane):
        dest = random.choices(ELIZ_DESTS, weights=ELIZ_WEIGHTS)[0]
        car.setup(x=X_ENTRY, y=lane, destination=dest,
                  road='elizondo', direction='east')

    def _spawn_elizondo(self, car):
        x    = random.uniform(X_ENTRY + 0.15, X_EXIT - 0.5)
        y    = random.choice(LANES_ELIZONDO)
        dest = random.choices(ELIZ_DESTS, weights=ELIZ_WEIGHTS)[0]
        if dest in ('cruce1_south', 'cruce1_north') and x > X_CRUCE1:
            dest = random.choices(['cruce2', 'cruce3', 'exit'],
                                   weights=[0.3, 0.3, 0.4])[0]
        if dest == 'cruce2' and x > X_CRUCE2:
            dest = random.choices(['cruce3', 'exit'], weights=[0.4, 0.6])[0]
        if dest == 'cruce3' and x > X_CRUCE3:
            dest = 'exit'
        car.setup(x=x, y=y, destination=dest, road='elizondo', direction='east')

    def _spawn_transversal(self, car):
        lid  = random.choice(TRANSVERSAL_IDS)
        info = TRANSVERSAL_LANES[lid]
        off  = random.uniform(0.0, 0.40)
        ys   = info['y_start'] - off if info['dir'] == 'north' else info['y_start'] + off
        car.setup(x=info['x'], y=ys, destination='exit',
                  road='transversal', direction=info['dir'], lane_id=lid)

    def _respawn(self, car):
        r = random.random()
        if r < 0.55:
            self._spawn_entry(car, random.choice(LANES_ELIZONDO))
        elif r < 0.75:
            self._spawn_elizondo(car)
        else:
            self._spawn_transversal(car)
        car.active = True

    # ── Registro / Step ──

    def _record(self):
        self.history.append({
            'cars': [
                (c.x, c.y, c.active, c.destination, c.waiting, c.road, c.direction)
                for c in self.cars
            ],
            'lights': [(l.x, l.is_green_for_elizondo()) for l in self.lights],
            'step': self.t,
        })

    def step(self):
        cars_l  = list(self.cars)
        light_l = list(self.lights)
        for car in cars_l:
            car.step_move(cars_l, light_l)
        for light in self.lights:
            light.step()
        for car in self.cars:
            if not car.active:
                self._respawn(car)
        self._record()

    def run_simulation(self, steps=400):
        self.setup()
        for _ in range(steps):
            self.step()
        return self.history


# ══════════════════════════════════════════════════
#  VISUALIZACIÓN
# ══════════════════════════════════════════════════

DEST_COLORS = {
    'cruce1_south': '#FF6B6B',
    'cruce1_north': '#FF9F43',
    'cruce2':       '#FFD93D',
    'cruce3':       '#6BCB77',
    'exit':         '#4D96FF',
}

def draw_road_layout(ax):
    ax.set_facecolor('#2C2C2C')
    road_w = abs(LANE_LEFT) + abs(LANE_RIGHT) + 0.05

    # Elizondo
    ax.add_patch(mpatches.FancyBboxPatch(
        (X_ENTRY, LANE_RIGHT - 0.025), X_EXIT - X_ENTRY, road_w,
        boxstyle="square,pad=0", lw=0, color='#555555', zorder=1))
    for y in [LANE_RIGHT + 0.0225, LANE_CENTER + 0.0225]:
        ax.plot([X_ENTRY, X_EXIT], [y, y],
                color='#FFFF00', lw=0.5, ls='--', alpha=0.5, zorder=2)
    for y in [LANE_RIGHT - 0.025, LANE_LEFT + 0.025]:
        ax.plot([X_ENTRY, X_EXIT], [y, y],
                color='white', lw=0.8, alpha=0.7, zorder=2)



    # Calles transversales — fondo y líneas
    # Nota: Junco de la Vega solo existe visualmente hacia y negativas,
    # porque en esta simulación no cruza completamente la avenida.
    trans_layout = [
        (X_CRUCE1 - 0.10, X_CRUCE1 + 0.10, -1.05,  1.05, 'Fernando\nGarcía Roel'),
        (X_CRUCE2 - 0.07, X_CRUCE2 + 0.07, -1.05, -0.07, 'Junco de\nla Vega'),
        (X_CRUCE3 - 0.05, X_CRUCE3 + 0.05, -1.05,  1.05, 'Garza\nSada'),
    ]

    for x1, x2, y1, y2, label in trans_layout:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="square,pad=0", lw=0, color='#444444', zorder=1))
        ax.text((x1+x2)/2, max(y1 + 0.15, -0.84), label,
                color='#BBBBBB', fontsize=5.5, ha='center', va='top',
                zorder=6, style='italic')


    # Zona de intersección
    # García Roel y Garza Sada cruzan completo.
    for cx in [X_CRUCE1, X_CRUCE3]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - INTER_HALF, -1.05), 2*INTER_HALF, 2.1,
            boxstyle="square,pad=0", lw=0, color='#3A3A3A', zorder=3))

    # Junco de la Vega solo se dibuja en el tramo activo hacia y negativas.
    ax.add_patch(mpatches.FancyBboxPatch(
        (X_CRUCE2 - INTER_HALF, -1.05), 2*INTER_HALF, 1.05,
        boxstyle="square,pad=0", lw=0, color='#3A3A3A', zorder=3))

    # Líneas divisorias en García Roel (4 carriles)
    for xc in [GR_NS_OUTER_X, GR_NS_INNER_X, GR_SN_LEFT_X, GR_SN_RIGHT_X]:
        ax.plot([xc, xc], [-1.05, -0.07], color='#FFFF00', lw=0.3,
                ls='--', alpha=0.35, zorder=4)
        ax.plot([xc, xc], [0.07, 1.05], color='#FFFF00', lw=0.3,
                ls='--', alpha=0.35, zorder=4)

    # Líneas divisorias en Junco de la Vega (3 carriles en lado sur)
    # N→S: x=-0.030 (entre -0.045 y -0.015)  |  S→N: x=0.015 y x=0.040
    for xc in [-0.015, 0.015, 0.040]:
        ax.plot([xc, xc], [-1.05, -0.07], color='#FFFF00', lw=0.3,
                ls='--', alpha=0.35, zorder=4)

    # Flechas de dirección — solo García Roel
    ak = dict(arrowstyle='->', lw=0.7)
    # García Roel N→S outer (sigue recto)
    ax.annotate('', xy=(GR_NS_OUTER_X, -0.42), xytext=(GR_NS_OUTER_X, -0.72),
                arrowprops=dict(**ak, color='#FF8888'), zorder=7)
    # García Roel N→S inner (gira → Elizondo)
    ax.annotate('', xy=(GR_NS_INNER_X, 0.12), xytext=(GR_NS_INNER_X, 0.42),
                arrowprops=dict(**ak, color='#FFAA44'), zorder=7)
    # García Roel S→N left (sigue recto)
    ax.annotate('', xy=(GR_SN_LEFT_X, 0.72), xytext=(GR_SN_LEFT_X, 0.42),
                arrowprops=dict(**ak, color='#88FF88'), zorder=7)
    # García Roel S→N right (gira → Elizondo)
    ax.annotate('', xy=(GR_SN_RIGHT_X, -0.12), xytext=(GR_SN_RIGHT_X, -0.42),
                arrowprops=dict(**ak, color='#44FFAA'), zorder=7)
    # Junco y Garza: sin flechas (el flujo se aprecia por los autos)

    ax.text(X_ENTRY + 0.05, LANE_LEFT + 0.04, 'Av. Luis Elizondo  →',
            color='#AAAAAA', fontsize=6, va='bottom', zorder=6)


def create_animation(history, interval=50):
    fig, ax = plt.subplots(figsize=(15, 5.5))
    fig.patch.set_facecolor('#1A1A2E')
    ax.set_xlim(X_ENTRY - 0.15, X_EXIT + 0.15)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    draw_road_layout(ax)

    n_cars = len(history[0]['cars'])
    car_patches = []
    for _ in range(n_cars):
        r = mpatches.FancyBboxPatch(
            (0, 0), 0.04, 0.025,
            boxstyle="round,pad=0.002",
            lw=0.4, edgecolor='white', facecolor='#4D96FF', zorder=8)
        ax.add_patch(r)
        car_patches.append(r)

    # Dos semáforos por cruce
    light_eliz, light_trans = [], []
    for lx, _ in history[0]['lights']:
        ce = plt.Circle((lx,  LANE_LEFT + 0.09), 0.022, zorder=9)
        ct = plt.Circle((lx, -(LANE_LEFT + 0.09)), 0.022, zorder=9)
        ax.add_patch(ce); ax.add_patch(ct)
        light_eliz.append(ce); light_trans.append(ct)
        ax.text(lx,  LANE_LEFT + 0.14, '→E', color='#999999',
                fontsize=4, ha='center', va='bottom', zorder=10)
        ax.text(lx, -(LANE_LEFT + 0.14), '↕T', color='#999999',
                fontsize=4, ha='center', va='top', zorder=10)

    step_text = ax.text(X_ENTRY, 1.02, '', color='#AAAAAA', fontsize=7, zorder=10)

    legend_elems = [mpatches.Patch(color=c, label=d) for d, c in DEST_COLORS.items()]
    legend_elems += [
        mpatches.Patch(color='#888888', label='esperando'),
        mpatches.Patch(color='#FF8888', label='N→S recto'),
        mpatches.Patch(color='#88FF88', label='S→N recto'),
        mpatches.Patch(color='#FFAA44', label='N→S→Elizondo'),
        mpatches.Patch(color='#44FFAA', label='S→N→Elizondo'),
    ]
    ax.legend(handles=legend_elems, loc='upper right', fontsize=4,
              framealpha=0.35, facecolor='#222222', labelcolor='white',
              title='Estado / Destino', title_fontsize=4.5, ncol=2)

    def update(fidx):
        frame = history[fidx]

        for i, (lx, eg) in enumerate(frame['lights']):
            light_eliz[i].set_color('#00FF00' if eg  else '#FF3333')
            light_trans[i].set_color('#FF3333' if eg else '#00FF00')

        for i, (cx, cy, active, dest, waiting, road, direction) in enumerate(frame['cars']):
            p = car_patches[i]
            if active:
                p.set_visible(True)
                if road == 'elizondo':
                    p.set_width(0.04); p.set_height(0.025)
                    p.set_x(cx - 0.020); p.set_y(cy - 0.0125)
                    p.set_facecolor('#888888' if waiting else DEST_COLORS.get(dest, '#FFF'))
                else:
                    p.set_width(0.025); p.set_height(0.04)
                    p.set_x(cx - 0.0125); p.set_y(cy - 0.020)
                    if waiting:
                        p.set_facecolor('#888888')
                    else:
                        info = TRANSVERSAL_LANES.get(
                            frame['cars'][i][7] if len(frame['cars'][i]) > 7 else None, {}
                        ) if False else {}
                        if direction == 'south':
                            # Distinguir si gira o sigue recto (por color)
                            p.set_facecolor('#FFAA44' if abs(cx - GR_NS_INNER_X) < 0.02
                                            else '#FF8888')
                        else:
                            p.set_facecolor('#44FFAA' if abs(cx - GR_SN_RIGHT_X) < 0.02
                                            else '#88FF88')
            else:
                p.set_visible(False)

        step_text.set_text(f'Step: {frame["step"]:>4d}')
        return car_patches + light_eliz + light_trans + [step_text]

    ani = animation.FuncAnimation(
        fig, update, frames=len(history),
        interval=interval, blit=True, repeat=True)
    plt.title('Simulación Av. Luis Elizondo — Onda Verde',
              color='white', fontsize=9, pad=6)
    plt.tight_layout()
    return fig, ani



# ══════════════════════════════════════════════════
#  RESULTADOS DE Q-LEARNING
# ══════════════════════════════════════════════════

def save_q_learning_curves(training_results, output_path):
    """Guarda las curvas de aprendizaje como en el notebook."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor('white')

    axes[0].plot(training_results['reward_history'])
    axes[0].set_title('Recompensa')
    axes[0].set_xlabel('Episodio')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(training_results['success_history'])
    axes[1].set_title('Vehículos exitosos')
    axes[1].set_xlabel('Episodio')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(training_results['stop_history'])
    axes[2].set_title('Paradas')
    axes[2].set_xlabel('Episodio')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_q_comparison(evaluations, output_path):
    """Compara sin coordinación, onda teórica y Q-learning."""
    labels = [e['label'] for e in evaluations]
    x = np.arange(len(labels))
    width = 0.25

    success_values = [e['successful_vehicles'] for e in evaluations]
    stop_values = [e['stops'] for e in evaluations]
    coord_values = [e['coordination_index'] * 100 for e in evaluations]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, success_values, width, label='Vehículos exitosos')
    ax.bar(x, stop_values, width, label='Paradas')
    ax.bar(x + width, coord_values, width, label='Índice coordinación (%)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Valor')
    ax.set_title('Comparación de coordinación semafórica')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_time_space_diagram(offsets, output_path, horizon=400):
    """Diagrama tiempo-espacio con verdes, rojos y trayectorias del pelotón."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1A1A2E')
    ax.set_facecolor('#2C2C2C')

    # Distancia acumulada desde el primer cruce, en unidades de la simulación.
    distances = CROSS_POSITIONS - CROSS_POSITIONS[0]

    for i in range(N_CROSSES):
        y = distances[i]
        t = 0
        while t < horizon:
            local = (t - offsets[i]) % CYCLE
            if local < GREEN_DURATION:
                duration = min(GREEN_DURATION - local, horizon - t)
                ax.hlines(y, t, t + duration, colors='#2ECC71', linewidth=9, alpha=0.9)
            else:
                duration = min(CYCLE - local, horizon - t)
                ax.hlines(y, t, t + duration, colors='#E74C3C', linewidth=9, alpha=0.9)
            t += duration

        ax.text(-12, y, CROSS_NAMES[i], va='center', ha='right', color='white', fontsize=9)

    for car_id in range(PLATOON_SIZE):
        start_t = car_id * PLATOON_HEADWAY
        times = start_t + IDEAL_ARRIVAL_STEPS
        ax.plot(times, distances, color='#3498DB', alpha=0.22, linewidth=1.0)

    for t_cross, y_cross in zip(IDEAL_ARRIVAL_STEPS, distances):
        ax.plot(t_cross, y_cross, 'o', color='white', ms=4, zorder=6)
        ax.axvline(t_cross, color='white', lw=0.4, ls=':', alpha=0.25)

    ax.set_xlabel('Tiempo (pasos de simulación)', color='white')
    ax.set_ylabel('Distancia acumulada sobre Av. Luis Elizondo (unidades)', color='white')
    ax.set_title(f'Diagrama tiempo-espacio con offsets aprendidos por Q-learning: {offsets}', color='white')
    ax.set_xlim(0, horizon)
    ax.set_ylim(distances[0] - 0.3, distances[-1] + 0.4)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')
    ax.grid(True, alpha=0.18)

    green_patch = mpatches.Patch(color='#2ECC71', label='Verde Elizondo')
    red_patch = mpatches.Patch(color='#E74C3C', label='Rojo Elizondo')
    blue_patch = mpatches.Patch(color='#3498DB', label='Trayectorias del pelotón')
    ax.legend(handles=[green_patch, red_patch, blue_patch], facecolor='#1A1A2E', labelcolor='white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def save_results_csv(evaluations, output_path):
    """Guarda resultados principales sin depender de pandas."""
    import csv
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['escenario', 'offsets', 'vehiculos_exitosos', 'paradas', 'indice_coordinacion', 'recompensa'])
        for e in evaluations:
            writer.writerow([
                e['label'],
                e['offsets'],
                e['successful_vehicles'],
                e['stops'],
                round(e['coordination_index'], 4),
                round(e['reward'], 4),
            ])

# ══════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════

if __name__ == '__main__':
    print("Entrenando Q-learning para aprender offsets de onda verde...")
    print(f"  Ciclo: {CYCLE} steps | Verde: {GREEN_DURATION} | Rojo: {RED_DURATION}")
    print(f"  Offset teórico aproximado: {theoretical_offsets()}")
    print(f"  Acciones posibles: {len(ACTIONS)}")

    training = train_q_learning(episodes=800, alpha=0.15, gamma=0.90, epsilon=0.30, seed=42)
    best_offsets = training['best_offsets']

    print("Entrenamiento terminado")
    print(f"  Mejores offsets aprendidos: {best_offsets}")

    no_coord_offsets = (0, 0, 0)
    theory_offsets = theoretical_offsets()
    evaluations = [
        evaluate_offsets('Sin coordinación', no_coord_offsets),
        evaluate_offsets('Onda teórica', theory_offsets),
        evaluate_offsets('Q-learning', best_offsets),
    ]

    print("\nResultados:")
    for e in evaluations:
        print(
            f"  {e['label']}: offsets={e['offsets']} | "
            f"vehículos exitosos={e['successful_vehicles']} | "
            f"paradas={e['stops']} | "
            f"índice={e['coordination_index']:.3f} | "
            f"recompensa={e['reward']:.2f}"
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))

    save_results_csv(evaluations, os.path.join(script_dir, 'q_learning_resultados.csv'))
    save_q_learning_curves(training, os.path.join(script_dir, 'q_learning_curvas.png'))
    save_q_comparison(evaluations, os.path.join(script_dir, 'q_learning_comparacion.png'))
    save_time_space_diagram(best_offsets, os.path.join(script_dir, 'diagrama_onda_verde_qlearning.png'), horizon=400)

    print("\nIniciando simulación con offsets aprendidos...")
    model = TrafficModel({'n_cars': 32, 'q_offsets': best_offsets})
    history = model.run_simulation(steps=400)
    print(f"  Frames registrados: {len(history)}")

    print("Generando animación...")
    fig, ani = create_animation(history, interval=50)
    output_gif = os.path.join(script_dir, 'traffic_simulation_qlearning.gif')

    print("Guardando GIF...")
    ani.save(output_gif, writer=animation.PillowWriter(fps=20), dpi=150)

    print("\nArchivos generados:")
    print(f"  {os.path.join(script_dir, 'q_learning_resultados.csv')}")
    print(f"  {os.path.join(script_dir, 'q_learning_curvas.png')}")
    print(f"  {os.path.join(script_dir, 'q_learning_comparacion.png')}")
    print(f"  {os.path.join(script_dir, 'diagrama_onda_verde_qlearning.png')}")
    print(f"  {output_gif}")
    print("¡Listo!")
