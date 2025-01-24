# Parámetros globales del sistema
NUM_AGENTS = 5  # Número de prosumidores
NUM_STATES = 10  # Estados posibles en el entorno
NUM_ACTIONS = 5  # Acciones posibles por agente
GAMMA = 0.99  # Factor de descuento para aprendizaje
ALPHA = 0.1  # Tasa de aprendizaje
EPSILON = 0.2  # Probabilidad de exploración

# Inicialización de tablas Q para cada prosumidor
Q_tables = [np.zeros((NUM_STATES, NUM_ACTIONS)) for _ in range(NUM_AGENTS)]

# Parámetros del mercado local de energía
define_prices = {
    "import": 0.1,  # Precio por importar energía de la red
    "export": 0.05,  # Precio por exportar energía a la red
    "battery_cost": 0.05,  # Costo asociado al uso de baterías por kWh
    "carbon_cost": 0.01  # Costo por kg de emisiones de carbono evitadas
}

# Modelo de prosumidor con batería y producción PV
def environment_step(agent, state, action):
    """Simula la interacción de un prosumidor con el mercado local."""
    pv_production = random.uniform(0, 5)  # Producción solar (kWh)
    load_demand = random.uniform(2, 6)  # Demanda doméstica (kWh)
    battery_level = state  # Nivel de batería (estado actual)
    next_state = (state + action - int(load_demand - pv_production)) % NUM_STATES

    # Recompensas según el uso de energía
    if action > load_demand:  # Exportación a la red
        reward = define_prices["export"] * (action - load_demand)
    elif action < load_demand:  # Importación de la red
        reward = -define_prices["import"] * (load_demand - action)
    else:  # Uso interno con batería
        reward = -define_prices["battery_cost"] * abs(action)

    # Incentivo por reducción de emisiones
    carbon_savings = max(0, pv_production - load_demand)
    reward += define_prices["carbon_cost"] * carbon_savings

    return next_state, reward

# Optimización inicial con modelo centralizado
def centralized_optimization(agent):
    """Genera una política óptima inicial para cada agente mediante optimización convexa."""
    def objective(action):
        # Función objetivo para minimizar costos y maximizar beneficios
        return -action * define_prices["export"] + action * define_prices["import"]

    # Restricciones
    bounds = [(0, NUM_ACTIONS - 1)]
    result = minimize(objective, x0=random.uniform(0, NUM_ACTIONS - 1), bounds=bounds, method='SLSQP')
    return int(result.x[0])

# Entrenamiento centralizado inicial
def centralized_training():
    """Inicializa las tablas Q mediante políticas óptimas centralizadas."""
    global Q_tables
    for agent in range(NUM_AGENTS):
        for state in range(NUM_STATES):
            optimal_action = centralized_optimization(agent)
            Q_tables[agent][state, optimal_action] = 1.0  # Asignación alta a la acción óptima

# Entrenamiento descentralizado con recompensas marginales
def decentralized_training(num_episodes=1000):
    """Entrenamiento descentralizado donde cada prosumidor actualiza su política."""
    global_rewards = [[] for _ in range(NUM_AGENTS)]  # Guardar recompensas de cada agente por episodio

    for episode in range(num_episodes):
        states = [random.randint(0, NUM_STATES - 1) for _ in range(NUM_AGENTS)]
        episode_rewards = [0 for _ in range(NUM_AGENTS)]  # Recompensas totales por agente en el episodio

        for t in range(50):
            for agent in range(NUM_AGENTS):
                # Política epsilon-greedy para exploración y explotación
                if random.uniform(0, 1) < EPSILON:
                    action = random.randint(0, NUM_ACTIONS - 1)
                else:
                    action = np.argmax(Q_tables[agent][states[agent], :])

                # Simulación del entorno
                next_state, reward = environment_step(agent, states[agent], action)

                # Actualización de la tabla Q
                best_next_action = np.argmax(Q_tables[agent][next_state, :])
                Q_tables[agent][states[agent], action] += ALPHA * (
                    reward + GAMMA * Q_tables[agent][next_state, best_next_action] - Q_tables[agent][states[agent], action]
                )

                # Actualización del estado y acumulación de recompensas
                states[agent] = next_state
                episode_rewards[agent] += reward

        # Guardar recompensas por agente en este episodio
        for agent in range(NUM_AGENTS):
            global_rewards[agent].append(episode_rewards[agent])

    # Visualizar el progreso del entrenamiento por agente
    plt.figure(figsize=(12, 8))
    for agent in range(NUM_AGENTS):
        plt.plot(global_rewards[agent], label=f"Agente {agent + 1}")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa total")
    plt.title("Progreso del entrenamiento descentralizado por agente")
    plt.legend()
    plt.grid()
    plt.show()

# Evaluación de políticas finales
def evaluate_policies():
    """Evalúa las políticas aprendidas por los prosumidores."""
    total_rewards = [0 for _ in range(NUM_AGENTS)]
    states = [random.randint(0, NUM_STATES - 1) for _ in range(NUM_AGENTS)]

    for t in range(50):
        for agent in range(NUM_AGENTS):
            action = np.argmax(Q_tables[agent][states[agent], :])
            next_state, reward = environment_step(agent, states[agent], action)
            total_rewards[agent] += reward
            states[agent] = next_state

    # Visualización de recompensas totales
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, NUM_AGENTS + 1), total_rewards, color='skyblue')
    plt.xlabel("Agente")
    plt.ylabel("Recompensa total")
    plt.title("Recompensas totales por prosumidor")
    plt.grid(axis='y')
    plt.show()

    print("Recompensas totales por prosumidor:", total_rewards)

# Flujo principal
centralized_training()  # Entrenamiento inicial basado en optimización
decentralized_training(num_episodes=1000)  # Entrenamiento descentralizado
evaluate_policies()  # Evaluación final

