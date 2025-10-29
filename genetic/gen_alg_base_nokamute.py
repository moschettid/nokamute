import numpy as np
import random
import itertools
import argparse
from typing import List
from multiprocessing import Pool, cpu_count
import subprocess
from datetime import datetime

#TO DO: 
# - use float instead of int for params, possible normalized
# - make individual play with std nokamute engine for a couple of games
# - augment number of amtch
# - change tournament tipology
# - change crossover method
# - change mutation method
# REFACTOR 

#NOTES:
# remove random_param_sample flag from mutate if not working

# PROBLEMS: 
# - first batch wins with many points more than others, and it stays so for ever
# - first gen it happens something, from second one no more
# PROBABLE CAUSES: fitness, wrong matches, job creation, 1st-2nd gen not fitness reset

# CONFIG
NUM_PARAMS = 21
POPULATION_SIZE = 30
ELITISM = 6
TOURNAMENT_OPPONENTS = 5

GENERATIONS = 100
NUM_GAMES = 5 #2

starting_individual = np.array([200, 10, 40, 1, 40, 25, 3, 7, 6, 3, 2, 8, 4, 5, 2, 4, 2, 2, 200, 20, 8])
# Game settings
MAX_TURNS = 100
TIME_TOTAL_SEC = 1
TIME_H = TIME_TOTAL_SEC // 3600
TIME_M = TIME_TOTAL_SEC // 60
TIME_S = TIME_TOTAL_SEC % 60
OK = "ok\n"
DEPTH = 4 #2  # Depth for the game engine

# -------------------------------
# UTILS
# -------------------------------
def generate_jobs(num_teams: int, batch_size: int) -> list:
    """
    Generate a list of jobs for a tournament-style competition.
    Each job consists of a pair of teams that will compete against each other.
    """
    assert num_teams % batch_size == 0, "num_teams must be divisible by batch_size"
    assert batch_size % 2 == 0, "batch_size must be an even number"

    batches = list(itertools.batched(range(0, num_teams), batch_size))

    jobs = []
    for batch in batches:
        batch = list(batch)
        random.shuffle(batch)
        for i in range(0, len(batch), 2):
            job = (batch[i], batch[i+1])
            jobs.append(job)

    for batch_pair in itertools.combinations(batches, 2):
        batch1, batch2 = batch_pair
        batch1, batch2 = list(batch1), list(batch2)
        random.shuffle(batch1)
        random.shuffle(batch2)
        for match in zip(batch1, batch2):
            jobs.append(match)

    return jobs

# -------------------------------
# GAME INTERFACE
# -------------------------------
def send(p: subprocess.Popen, command: str) -> str:
    p.stdin.write(command + "\n")
    p.stdin.flush()
    return read_all(p)

def readuntil(p: subprocess.Popen, delim: str) -> str:
    output = []
    while True:
        line = p.stdout.readline()
        if not line:
            break
        output.append(line.strip())
        if line.endswith(delim):
            break
    return "\n".join(output)

def read_all(p: subprocess.Popen) -> str:
    return readuntil(p, OK)

def start_process(path, args=[]) -> subprocess.Popen:
    return subprocess.Popen(
        [path] + args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
    )

def end_process(p: subprocess.Popen) -> None:
    p.stdin.close()
    p.stdout.close()
    p.stderr.close()
    p.kill()

def play_step(p1: subprocess.Popen, p2: subprocess.Popen) -> str:
    if DEPTH > 0:
        move = send(p1, f"bestmove depth {DEPTH}")
    else:
        move = send(p1, f"bestmove time {TIME_H:02}:{TIME_M:02}:{TIME_S:02}")
    move = move.strip().split("\n")[0]
    #print(f"[Player] plays: {move}")
    send(p1, f"play {move}")
    return send(p2, f"play {move}")

def check_end_game(out: str) -> bool:
    return "InProgress" != out.split(";")[1]

def start_game(prompt_a, prompt_b) -> str:
    # MAX_TURNS = 100
    path = "./nokamute_opening"
    path_base = "./nokamute1.0"
    
    #print(f"Starting interaction with {path}...")
    if prompt_a is not None:
        player1 = start_process(path, ["--set-eval", prompt_a])#"--num-threads=1" now hardcoded in the engine
    else:
        player1 = start_process(path_base, ["--num-threads=1"])
    read_all(player1)
    if prompt_b is not None:
        player2 = start_process(path, ["--set-eval", prompt_b])#"--num-threads=1" now hardcoded in the engine
    else:
        player2 = start_process(path_base, ["--num-threads=1"])    
    read_all(player2)

    send(player1, "newgame Base+MLP")
    send(player2, "newgame Base+MLP")

    info = ["Base+MLP", "Draw"]  # Valore di default in caso di pareggio
    for _ in range(MAX_TURNS):
        out = play_step(player1, player2)
        if check_end_game(out):
            info = out.split("\n")[0].split(";")
            break

        out = play_step(player2, player1)
        if check_end_game(out):
            info = out.split("\n")[0].split(";")
            break

    send(player1, "exit")
    send(player2, "exit")
    end_process(player1)
    end_process(player2)
    # print("Game over.")
    print(f"Game result: {info[1]}")
    
    
    return info[1] # risultato della partita



###########################################

# -------------------------------
# ENGINE INTERFACE
# -------------------------------
def play_match(ind_a: 'Individual', ind_b: 'Individual') -> List[tuple]: #REFACTOR
    """
    Simulate a match between two individuals.
    This function should be replaced with the actual game engine logic.
    Return: List of tuples (winner, loser, white, draw)
    """
    params_a = ind_a.params
    params_b = None

    results = []
    # Play multiple games to get a more reliable result
    for _ in range(NUM_GAMES):
        result = play_game(params_a, params_b) #Return: 1 if A wins, -1 if B wins, 0 for draw
        if result == 0:
            results.append((ind_a, ind_b, ind_a, True)) #winner, loser, white, draw
        elif result == 1:
            results.append((ind_a, ind_b, ind_a, False))
        else:
            results.append((ind_b, ind_a, ind_a, False))

        result = play_game(params_b, params_a)
        if result == 0:
            results.append((ind_b, ind_a, ind_b, True))
        elif result == 1:
            results.append((ind_b, ind_a, ind_b, False))
        else:
            results.append((ind_a, ind_b, ind_b, False))

    print(f"Match between {ind_a.individual_id} and base nokamute finished")
    return results
    

def play_game(params_a, params_b) -> int:
    """
    Simulate a game between two parameter sets.
    You MUST replace this with your actual engine logic.
    Return: 1 if A wins, -1 if B wins, 0 for draw
    """
    if not params_a is None:
        params_a = np.round(params_a).astype(int)
        prompt_a = "queen_liberty_penalty:{},gates_factor:{},queen_spawn_factor:{},unplayed_bug_factor:{},pillbug_defense_bonus:{},ant_game_factor:{},queen_score:{},ant_score:{},beetle_score:{},grasshopper_score:{},spider_score:{},mosquito_score:{},ladybug_score:{},pillbug_score:{},mosquito_incremental_score:{},stacked_bug_factor:{},queen_movable_penalty_factor:{},opponent_queen_liberty_penalty_factor:{},trap_queen_penalty:{},placeable_pillbug_defense_bonus:{},pinnable_beetle_factor:{}".format(*list(params_a))
    else:
        params_a = None
        prompt_a = None
    if not params_b is None:
        params_b = np.round(params_b).astype(int)
        prompt_b = "queen_liberty_penalty:{},gates_factor:{},queen_spawn_factor:{},unplayed_bug_factor:{},pillbug_defense_bonus:{},ant_game_factor:{},queen_score:{},ant_score:{},beetle_score:{},grasshopper_score:{},spider_score:{},mosquito_score:{},ladybug_score:{},pillbug_score:{},mosquito_incremental_score:{},stacked_bug_factor:{},queen_movable_penalty_factor:{},opponent_queen_liberty_penalty_factor:{},trap_queen_penalty:{},placeable_pillbug_defense_bonus:{},pinnable_beetle_factor:{}".format(*list(params_b))
    else:
        params_b = None
        prompt_b = None

    result = start_game(prompt_a, prompt_b)
    if result == "Draw":
        return 0
    elif result == "WhiteWins":
        return 1
    elif result == "BlackWins":
        return -1
    else:
        raise ValueError(f"Unexpected game result: {result}")

# -------------------------------
# GENETIC INDIVIDUAL
# -------------------------------
class Individual:
    def __init__(self, batch_id=0, params=None, individual_id=None):
        # Generate random parameters between 1 and 500 if none provided
        if params is None:
            self.params = np.random.randint(1, 501, NUM_PARAMS)
        else:
            self.params = params
        self.batch_id = batch_id
        self.fitness = 0
        self.individual_id = individual_id

    def mutate(self, mutation_percent, random_param_sample=False):
        mutation_scale = mutation_percent / 100.0  # Convert percentage to scale factor

        # Create new parameter array by adding random noise scaled by the mutation percentage
        mutated_params = self.params.copy()
        for i in range(NUM_PARAMS):
            # Scale the mutation based on the current parameter value
            param_scale = abs(mutated_params[i]) * mutation_scale
            # If parameter is zero or very small, use a minimum scale
            if param_scale < 0.1:
                param_scale = 0.1
            # Apply random mutation within the range
            mutated_params[i] += np.random.uniform(-param_scale, param_scale)
            # Ensure parameters are bigger than 1
            mutated_params[i] = max(mutated_params[i], 1)
        # Return a new individual with the mutated parameters
        if random_param_sample:
            # Randomly sample a subset of 1 parameter to generate random
            index = np.random.choice(NUM_PARAMS, 1, replace=False)
            mutated_params[index] = np.random.randint(1, 501)
        return Individual(batch_id=self.batch_id, params=mutated_params, individual_id=self.individual_id)

# -------------------------------
# FITNESS EVALUATION
# -------------------------------

def evaluate_population(population: List[Individual], threads: int):
    # Define jobs
    # jobs = generate_jobs(POPULATION_SIZE, 6)
    #We do not need anymore the jobs, we just play matches against base nokamute
    jobs = []
    for i in range(POPULATION_SIZE):
        jobs.append((i, None))

    if threads > 1:
        with Pool(threads) as pool:
            results = pool.starmap(play_match, [(population[a], b) for a, b in jobs])
    else:
        results = [play_match(population[a], b) for a, b in jobs]

    # Update fitness based on results
    for match_result in results:
        for (winner, loser, white, draw) in match_result: #maybe refactoring
            
            winner_idx = winner.individual_id if winner is not None else None
            loser_idx = loser.individual_id if loser is not None else None
            white_idx = white.individual_id if white is not None else None
            if draw:
                if winner is not None:
                    if white is not None:
                        population[winner_idx].fitness += 0.4
                    else:
                        population[winner_idx].fitness += 0.6
                else:
                    if white is not None:
                        population[loser_idx].fitness += 0.4
                    else:
                        population[loser_idx].fitness += 0.6
            elif winner is not None:
                if winner_idx == white_idx:
                    population[winner_idx].fitness += 0.4 * 3
                else:
                    population[winner_idx].fitness += 0.6 * 3
        


# -------------------------------
# GENETIC LOOP
# -------------------------------
# 1-6 : Elitism (best individuals are preserved)
# 7-12: Crossover [+ slight mutation eventually] of elite individuals (1-6)
# 13-18: Crossover [+ slight mutation] of two individuals: 1 from elite, 1 7-12
# 19-24: Crossover [+ medium mutation] of two random individuals from 1-24
# 25-30: Something else pretty random, like strong mutations of the elite (i.e. +- 40% of each param) + random params sample
def evolve_population(population: List[Individual]) -> List[Individual]:
    batch_size = ELITISM
    new_population = population[:batch_size]

    # Crossover of elite individuals
    for _ in range(batch_size):
        parent1, parent2 = random.sample(new_population[:ELITISM], 2) # We could switch to refined version with probability proportional to fitness
        child_params = crossover(parent1.params, parent2.params)
        # Slight mutation
        child = Individual(params=child_params).mutate(5)
        new_population.append(child)
    
    # Crossover of elite with non-elite individuals
    for _ in range(batch_size):
        parent1 = random.choice(new_population[:ELITISM])
        parent2 = random.choice(population[ELITISM:2*ELITISM])
        child_params = crossover(parent1.params, parent2.params)
        # Slight mutation
        child = Individual(params=child_params).mutate(8)
        new_population.append(child)
    
    # Crossover of two random individuals from the first 24
    for _ in range(batch_size):
        parent1, parent2 = random.sample(population[:24], 2)
        child_params = crossover(parent1.params, parent2.params)
        # Medium mutation
        child = Individual(params=child_params).mutate(12)
        new_population.append(child)
    
    # Random mutations of elite individuals
    for i in range(batch_size):
        child = Individual(params=population[i].params).mutate(40, True)  # Strong mutation
        new_population.append(child)

    while len(new_population) < POPULATION_SIZE:
        parent = random.choice(population[:POPULATION_SIZE // 2])
        new_population.append(parent.mutate())
    
    for i in range(POPULATION_SIZE // batch_size):
        for j in range(batch_size):
            new_population[i * batch_size + j].batch_id = i

    for i in range(POPULATION_SIZE):
        new_population[i].individual_id = i
                
    for p in new_population:
        p.fitness = 0
    
    return new_population

def crossover(params_a, params_b):
    """
    Perform crossover between two sets of parameters.
    This can be a simple average or a more complex operation.
    """
    """crossover_point = random.randint(0, NUM_PARAMS - 1)
    child_params = np.concatenate((params_a[:crossover_point], params_b[crossover_point:]))"""
    # Compute harmonic mean while handling potential zeros in parameters
    # Harmonic mean = n / (sum of reciprocals)
    epsilon = 1e-10  # Small value to prevent division by zero
    child_params = np.zeros_like(params_a)
    for i in range(len(params_a)):
        # Add epsilon to avoid division by zero
        p1, p2 = abs(params_a[i]) + epsilon, abs(params_b[i]) + epsilon
        harmonic_mean = 2 * p1 * p2 / (p1 + p2)
        # Preserve sign from one of the parents (randomly chosen)
        sign = np.sign(params_a[i]) if random.random() < 0.5 else np.sign(params_b[i])
        # Round to nearest integer and convert to int
        child_params[i] = int(sign * round(harmonic_mean))
    return child_params

def train(threads: int):
    population = [Individual(params=starting_individual, individual_id=0)]
    for k in range(1,6):
        population.append(Individual(batch_id=0, params=starting_individual, individual_id=k).mutate(50))
    for k in range(6, POPULATION_SIZE-6):
        population.append(Individual(batch_id=k//6, params=starting_individual, individual_id=k).mutate(100))
    for k in range(6):
        population.append(Individual(batch_id=4, individual_id=k+POPULATION_SIZE-6))


    for gen in range(GENERATIONS):
        print(f"\nGeneration {gen}")
        evaluate_population(population, threads)
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        best = population[0]
        print(f"Best fitness: {best.fitness:.2f}")
        print(f"Best params (truncated): {best.params[:5]}...")
        with open("log.txt", "a") as log_file:
            log_file.write(f"Generation {gen}\n\n")
            for p in population:
                log_file.write(f"Fitness: {p.fitness:.2f}\nBatch id: {p.batch_id}\nParams: {p.params.tolist()}\n\n")
            log_file.write(f"\n\n")

        population = evolve_population(population)

    return population[0]

# -------------------------------
# MAIN ENTRY
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of threads (1 = single-core)")
    args = parser.parse_args()

    f = open("log.txt", "w")
    f.close()

    print(f"Running with {args.threads} thread(s)")
    best_agent = train(threads=args.threads)
