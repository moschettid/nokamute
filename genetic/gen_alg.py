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
NUM_PARAMS = 29
POPULATION_SIZE = 30
ELITISM = 6
TOURNAMENT_OPPONENTS = 5

GENERATIONS = 100
NUM_GAMES = 5 #2

#starting_individual = np.array([200.0, 10.0, 40.0, 1, 40.0, 25.0, 3.0, 7.0, 6.0, 3.0, 2.0, 8.0, 4.0, 5.0, 2.0, 4.0, 2.0, 2.0, 200.0, 20.0, 8.0, 20.0, 20.0, 20.0, 10.0, 20.0, 20.0, 20.0, 20.0])
starting_individual = np.array([200.0, 0, 0, 1.0, 40.0, 0, 4.0, 7.0, 6.0, 2.0, 2.0, 8.0, 6.0, 5.0, 0.0, 2.0, 0.5, 0.5, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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

def generate_all_jobs(num_teams: int) -> list:
    """
    Generate all possible matchups for a round-robin tournament.
    Each team plays against every other team exactly once.
    """
    jobs = []
    for i in range(num_teams):
        for j in range(i + 1, num_teams):
            jobs.append((i, j))
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
    path = "./nokamute"
    
    #print(f"Starting interaction with {path}...")
    #print(prompt_a)
    player1 = start_process(path, ["--set-eval", prompt_a])#, "--num-threads=1"
    read_all(player1)
    #print(f"Starting interaction with {path}...")
    player2 = start_process(path, ["--set-eval", prompt_b])#, "--num-threads=1"
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
    #print(f"Game result: {info[1]}")
    
    
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
    params_b = ind_b.params

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

    print(f"Match between {ind_a.individual_id} and {ind_b.individual_id} finished")
    return results
    

def play_game(params_a, params_b) -> int:
    """
    Simulate a game between two parameter sets.
    You MUST replace this with your actual engine logic.
    Return: 1 if A wins, -1 if B wins, 0 for draw
    """
    # params_a = np.round(params_a).astype(int)
    # params_b = np.round(params_b).astype(int)

    prompt_a = "queen_liberty_penalty:{},gates_factor:{},queen_spawn_factor:{},unplayed_bug_factor:{},pillbug_defense_bonus:{},ant_game_factor:{},queen_score:{},ant_score:{},beetle_score:{},grasshopper_score:{},spider_score:{},mosquito_score:{},ladybug_score:{},pillbug_score:{},mosquito_incremental_score:{},stacked_bug_factor:{},queen_movable_penalty_factor:{},opponent_queen_liberty_penalty_factor:{},trap_queen_penalty:{},placeable_pillbug_defense_bonus:{},pinnable_beetle_factor:{}, mosquito_ant_factor:{},mobility_factor:{},compactness_factor:{},pocket_factor:{},beetle_attack_factor:{},beetle_on_enemy_queen_factor:{},beetle_on_enemy_pillbug_factor:{},direct_queen_drop_factor:{}".format(*list(params_a))
    #print(prompt_a)
    prompt_b = "queen_liberty_penalty:{},gates_factor:{},queen_spawn_factor:{},unplayed_bug_factor:{},pillbug_defense_bonus:{},ant_game_factor:{},queen_score:{},ant_score:{},beetle_score:{},grasshopper_score:{},spider_score:{},mosquito_score:{},ladybug_score:{},pillbug_score:{},mosquito_incremental_score:{},stacked_bug_factor:{},queen_movable_penalty_factor:{},opponent_queen_liberty_penalty_factor:{},trap_queen_penalty:{},placeable_pillbug_defense_bonus:{},pinnable_beetle_factor:{}, mosquito_ant_factor:{},mobility_factor:{},compactness_factor:{},pocket_factor:{},beetle_attack_factor:{},beetle_on_enemy_queen_factor:{},beetle_on_enemy_pillbug_factor:{},direct_queen_drop_factor:{}".format(*list(params_b))

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
            self.params = np.round(np.random.uniform(0, 500, NUM_PARAMS), 2)
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
        # Return a new individual with the mutated parameters
        if random_param_sample:
            # Randomly sample a subset of  5 parameters to generate random
            index = np.random.choice(NUM_PARAMS, 1, replace=False)
            mutated_params[index] = np.round(np.random.uniform(0, 500, 1), 2)
        return Individual(batch_id=self.batch_id, params=mutated_params, individual_id=self.individual_id)

# -------------------------------
# FITNESS EVALUATION
# -------------------------------

def evaluate_population(population: List[Individual], threads: int):
    # Define jobs
    #jobs = generate_jobs(POPULATION_SIZE, 6)
    jobs = generate_all_jobs(POPULATION_SIZE)

    if threads > 1:
        with Pool(threads) as pool:
            results = pool.starmap(play_match, [(population[a], population[b]) for a, b in jobs])
    else:
        results = [play_match(population[a], population[b]) for a, b in jobs]

    # Update fitness based on results
    for match_result in results:
        for (winner, loser, white, draw) in match_result: #maybe refactoring
            winner_idx = winner.individual_id
            loser_idx = loser.individual_id
            if draw:
                if winner.individual_id == white.individual_id:
                    if winner_idx is not None:
                        population[winner_idx].fitness += 0.45
                    if loser_idx is not None:
                        population[loser_idx].fitness += 0.55
                else:
                    if winner_idx is not None:
                        population[winner_idx].fitness += 0.55
                    if loser_idx is not None:
                        population[loser_idx].fitness += 0.45
            else:
                if winner.individual_id == white.individual_id:
                    if winner_idx is not None:
                        population[winner_idx].fitness += 0.45 * 3
                else:
                    if winner_idx is not None:
                        population[winner_idx].fitness += 0.55 * 3
            #check if winner_idx and loser_idx are in 1,2,3,4,5,6
            # if winner_idx < 6 or loser_idx < 6:
            #     print(f"Winner: {winner_idx}, Loser: {loser_idx}, Draw: {draw}")
        


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

    # Not even necessary(?)
    while len(new_population) < POPULATION_SIZE:
        parent = random.choice(population[:POPULATION_SIZE // 2])
        new_population.append(parent.mutate(0))
    
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
    """# Compute harmonic mean while handling potential zeros in parameters
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
        child_params[i] = sign * round(harmonic_mean)"""
    # get random crossover subset of params
    number_of_crossover_params = random.randint(1, NUM_PARAMS // 2)
    crossover_indices = np.random.choice(NUM_PARAMS, size=number_of_crossover_params, replace=False)
    remaining_indices = np.setdiff1d(np.arange(NUM_PARAMS), crossover_indices)

    child_params = np.zeros_like(params_a)
    child_params[crossover_indices] = params_a[crossover_indices]
    child_params[remaining_indices] = params_b[remaining_indices]
    return child_params

def read_last_generation_from_log():
    """Read the last generation from log.txt file if it exists.
    Returns: (generation_number, population) if successful, None if file is empty or doesn't exist.
    """
    try:
        # Check if file exists
        import os
        if not os.path.exists("log.txt") or os.path.getsize("log.txt") == 0:
            return None
            
        # File exists, try to read it
        with open("log.txt", "r") as log_file:
            content = log_file.read()
            
        # Split content by generations
        generations = content.split("Generation ")
        if len(generations) <= 1:
            print("Error: log.txt exists but does not contain valid generation data.")
            exit(1)
            
        # Get the last generation's data
        last_gen_data = generations[-1]
        try:
            gen_num = int(last_gen_data.split("\n")[0])
        except ValueError:
            print("Error: Could not parse generation number from log.txt")
            exit(1)
        
        # Parse individuals' data
        individuals = []
        individual_blocks = last_gen_data.split("Fitness: ")
        
        # Skip the first element which is the generation header
        for block in individual_blocks[1:]:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue  # Skip invalid blocks
                
            try:
                fitness = float(lines[0].strip())
                batch_id = int(lines[1].split(":")[1].strip())
                params_str = lines[2].split("Params: ")[1].strip()
                
                # Convert string representation of params to numpy array
                import ast
                params = np.array(ast.literal_eval(params_str))
                
                ind = Individual(batch_id=batch_id, params=params)
                ind.fitness = fitness
                individuals.append(ind)
            except Exception as e:
                print(f"Error parsing individual data: {e}")
                exit(1)
            
        if not individuals:
            print("Error: No valid individuals found in log.txt")
            exit(1)
            
        # Assign individual IDs
        for i, ind in enumerate(individuals):
            ind.individual_id = i
            
        return (gen_num, individuals)
    except Exception as e:
        print(f"Error reading log.txt: {e}")
        print("The file format is unexpected. Please check the file or remove it to start a new training.")
        exit(1)

def train(threads: int):
    # Try to read the last generation from log.txt
    last_gen_data = read_last_generation_from_log()
    
    if last_gen_data:
        gen_num, population = last_gen_data
        print(f"Resuming from generation {gen_num}")
        # Sort by fitness and evolve the population
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        population = evolve_population(population)
        # Start from the next generation
        gen_num += 1
    else:
        gen_num = 0
        # Initialize new population
        population = [Individual(params=starting_individual, individual_id=0)]
        for k in range(1,6):
            population.append(Individual(batch_id=0, params=starting_individual, individual_id=k).mutate(50))
        for k in range(6, POPULATION_SIZE-6):
            population.append(Individual(batch_id=k//6, params=starting_individual, individual_id=k).mutate(100))
        for k in range(6):
            population.append(Individual(batch_id=4, individual_id=k+POPULATION_SIZE-6))

    try:
        # Run indefinitely instead of for a fixed number of generations
        while True:
            print(f"\nGeneration {gen_num}")
            evaluate_population(population, threads)
            population.sort(key=lambda ind: ind.fitness, reverse=True)

            best = population[0]
            print(f"Best fitness: {best.fitness:.2f}")
            print(f"Best params (truncated): {best.params[:5]}...")
            
            # Always append to log file
            with open("log.txt", "a") as log_file:
                log_file.write(f"Generation {gen_num}\n\n")
                for p in population:
                    log_file.write(f"Fitness: {p.fitness:.2f}\nBatch id: {p.batch_id}\nParams: {p.params.tolist()}\n\n")
                log_file.write(f"\n\n")
            
            # Evolve population for next generation
            population = evolve_population(population)
            gen_num += 1
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Return the best individual from the last generation
        population.sort(key=lambda ind: ind.fitness, reverse=True)

    return population[0]

# -------------------------------
# MAIN ENTRY
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of threads (1 = single-core)")
    args = parser.parse_args()

    print(f"Running with {args.threads} thread(s)")
    best_agent = train(threads=args.threads)
