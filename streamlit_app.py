import random
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ------------------------------------------
# Helper Functions
# ------------------------------------------
def simulate_schedule_details(order, jobs, num_machines):
    """
    Simulate processing for a given order of jobs.
    Returns the overall makespan and detailed schedule for each job.
    Each job's schedule is a list of dictionaries with machine, start, finish, and processing time.
    """
    machine_available = [0] * num_machines  # next available time for each machine
    job_details = {}
    
    for job_index in order:
        job = jobs[job_index]
        current_time = 0
        op_details = []
        for machine, proc_time in zip(job['route'], job['processing_times']):
            start = max(current_time, machine_available[machine])
            finish = start + proc_time
            op_details.append({'machine': machine, 'start': start, 'finish': finish, 'proc_time': proc_time})
            current_time = finish
            machine_available[machine] = finish
        job_details[job_index] = op_details
    makespan = max([ops[-1]['finish'] for ops in job_details.values()])
    return makespan, job_details

def create_schedule_dataframe(order, job_details, jobs):
    rows = []
    for job in order:
        details = job_details[job]
        total_time = sum(op['proc_time'] for op in details)
        finish_time = details[-1]['finish']
        row = {
            "Job": job,
            "Route": " -> ".join(str(m) for m in jobs[job]['route']),
            "Processing Times": ", ".join(str(t) for t in jobs[job]['processing_times']),
            "Total Processing Time": total_time,
            "Finish Time": finish_time
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

# GA Components
def initialize_population(pop_size, num_jobs):
    population = []
    base = list(range(num_jobs))
    for _ in range(pop_size):
        chrom = base.copy()
        random.shuffle(chrom)
        population.append(chrom)
    return population

def fitness(chromosome, jobs, num_machines):
    return simulate_schedule_details(chromosome, jobs, num_machines)[0]

def tournament_selection(population, jobs, num_machines, tournament_size):
    selected = []
    for _ in range(len(population)):
        competitors = random.sample(population, tournament_size)
        best = min(competitors, key=lambda chrom: fitness(chrom, jobs, num_machines))
        selected.append(best)
    return selected

def order_crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    # Copy a slice from parent1
    child[a:b] = parent1[a:b]
    fill = [item for item in parent2 if item not in child[a:b]]
    i = 0
    for j in range(size):
        if child[j] is None:
            child[j] = fill[i]
            i += 1
    return child

def swap_mutation(chromosome, mutation_rate):
    chrom = chromosome.copy()
    for i in range(len(chrom)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(chrom) - 1)
            chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom

# ------------------------------------------
# Streamlit Application
# ------------------------------------------
st.title("AI-Powered Job Shop Scheduling Simulation")

st.sidebar.header("Simulation Parameters")
num_jobs = st.sidebar.number_input("Number of Jobs", min_value=10, max_value=500, value=100, step=10)
num_machines = st.sidebar.number_input("Number of Machines", min_value=5, max_value=100, value=20, step=1)
num_runs = st.sidebar.number_input("Number of Runs", min_value=1, max_value=10, value=1, step=1)
run_simulation = st.sidebar.button("Run Simulation")

if run_simulation:
    all_runs_results = []
    for run in range(1, num_runs + 1):
        st.subheader(f"Run #{run}")
        
        # Create a new simulation instance
        jobs = []
        for i in range(num_jobs):
            route_length = random.randint(3, min(6, num_machines))
            route = random.sample(range(num_machines), route_length)
            processing_times = [random.randint(10, 100) for _ in range(route_length)]
            jobs.append({'route': route, 'processing_times': processing_times})
        
        # Baseline Scheduling: FIFO and SPT
        fifo_order = list(range(num_jobs))
        spt_order = sorted(range(num_jobs), key=lambda j: jobs[j]['processing_times'][0])
        
        fifo_makespan, fifo_details = simulate_schedule_details(fifo_order, jobs, num_machines)
        spt_makespan, spt_details = simulate_schedule_details(spt_order, jobs, num_machines)
        
        st.markdown("### FIFO Schedule")
        fifo_df = create_schedule_dataframe(fifo_order, fifo_details, jobs)
        st.dataframe(fifo_df)
        st.write("**FIFO Makespan:**", fifo_makespan)
        
        st.markdown("### SPT Schedule")
        spt_df = create_schedule_dataframe(spt_order, spt_details, jobs)
        st.dataframe(spt_df)
        st.write("**SPT Makespan:**", spt_makespan)
        
        # Genetic Algorithm (GA) Implementation
        pop_size = 50
        generations = 100
        tournament_size = 5
        mutation_rate = 0.1

        population = initialize_population(pop_size, num_jobs)
        best_makespans = []
        best_chromosome = None
        best_fit = float('inf')

        for gen in range(generations):
            for chrom in population:
                fit_val = fitness(chrom, jobs, num_machines)
                if fit_val < best_fit:
                    best_fit = fit_val
                    best_chromosome = chrom
            best_makespans.append(best_fit)

            selected = tournament_selection(population, jobs, num_machines, tournament_size)
            new_population = []
            for i in range(0, pop_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i+1) % pop_size]
                child1 = order_crossover(parent1, parent2)
                child2 = order_crossover(parent2, parent1)
                child1 = swap_mutation(child1, mutation_rate)
                child2 = swap_mutation(child2, mutation_rate)
                new_population.extend([child1, child2])
            population = new_population[:pop_size]
        
        ga_makespan, ga_details = simulate_schedule_details(best_chromosome, jobs, num_machines)
        
        st.markdown("### GA Optimized Schedule")
        ga_df = create_schedule_dataframe(best_chromosome, ga_details, jobs)
        st.dataframe(ga_df)
        st.write("**GA Best Makespan:**", ga_makespan)
        
        # GA Optimization Progress Chart
        st.markdown("#### GA Makespan Improvement Over Generations")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(best_makespans, marker='o')
        ax.set_title(f"Run #{run}: GA Optimization")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Makespan")
        ax.grid(True)
        st.pyplot(fig)
        
        st.markdown("---")
