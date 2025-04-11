import random
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np

# ------------------------------------------
# Helper Functions
# ------------------------------------------

def simulate_schedule_uncertain(order, jobs, num_machines, delta=0.2):
    """
    Simulates one execution of a schedule with uncertain processing times.
    """
    machine_available = [0] * num_machines
    job_details = {}
    
    for job_index in order:
        job = jobs[job_index]
        current_time = 0
        op_details = []
        for machine, proc_time in zip(job['route'], job['processing_times']):
            # Apply uncertainty to processing time
            actual_proc = random.uniform(proc_time * (1 - delta), proc_time * (1 + delta))
            start = max(current_time, machine_available[machine])
            finish = start + actual_proc
            op_details.append({'machine': machine, 'start': start, 'finish': finish, 'proc_time': actual_proc})
            current_time = finish
            machine_available[machine] = finish
        job_details[job_index] = op_details
    makespan = max([ops[-1]['finish'] for ops in job_details.values()])
    return makespan

def evaluate_schedule_monte_carlo(order, jobs, num_machines, K=50, delta=0.2, due_buffer=1.5):
    """
    Evaluate a schedule using K Monte Carlo simulations.
    Returns average makespan, std dev, and on-time completion rate.
    """
    makespans = []
    on_time_counts = []
    due_times = []

    # Assign due time as buffer * sum of nominal processing times
    for job in jobs:
        due = sum(job['processing_times']) * due_buffer
        due_times.append(due)

    for _ in range(K):
        machine_available = [0] * num_machines
        job_completion = [0] * len(jobs)

        for job_index in order:
            job = jobs[job_index]
            current_time = 0
            for machine, proc_time in zip(job['route'], job['processing_times']):
                actual_proc = random.uniform(proc_time * (1 - delta), proc_time * (1 + delta))
                start = max(current_time, machine_available[machine])
                finish = start + actual_proc
                current_time = finish
                machine_available[machine] = finish
            job_completion[job_index] = current_time

        makespans.append(max(job_completion))
        on_time = sum(1 for i in range(len(jobs)) if job_completion[i] <= due_times[i])
        on_time_counts.append(on_time / len(jobs))

    return (
        round(np.mean(makespans), 2),
        round(np.std(makespans), 2),
        round(np.mean(on_time_counts), 2)
    )

# ------------------------------------------
# Streamlit App: Step 1
# ------------------------------------------
st.title("AI-Powered Job Shop Scheduling")

# Sidebar Parameters
st.sidebar.header("Simulation Parameters")
num_jobs = st.sidebar.slider("Number of Jobs", 10, 200, 50, step=10)
num_machines = st.sidebar.slider("Number of Machines", 5, 30, 10, step=1)
generations = st.sidebar.slider("GA Generations", 10, 300, 100, step=10)
uncertainty_mode = st.sidebar.checkbox("Enable Monte Carlo Simulation", value=True)
K = st.sidebar.slider("Monte Carlo Simulations (K)", 10, 200, 50, step=10)
run_button = st.sidebar.button("Run Optimization")

if run_button:
    st.subheader("🔁 Simulation Started")

    # Generate random jobs
    jobs = []
    for i in range(num_jobs):
        route_len = random.randint(3, min(6, num_machines))
        route = random.sample(range(num_machines), route_len)
        times = [random.randint(10, 100) for _ in range(route_len)]
        jobs.append({'route': route, 'processing_times': times})

    # FIFO Schedule
    fifo_order = list(range(num_jobs))
    fifo_avg, fifo_std, fifo_ontime = evaluate_schedule_monte_carlo(fifo_order, jobs, num_machines, K=K) if uncertainty_mode \
        else (simulate_schedule_uncertain(fifo_order, jobs, num_machines), 0, 0)
    st.markdown(f"### 📦 FIFO")
    st.write(f"**Avg Makespan:** {fifo_avg}, **Std:** {fifo_std}, **On-time Rate:** {fifo_ontime}")

    # SPT Schedule
    spt_order = sorted(range(num_jobs), key=lambda j: jobs[j]['processing_times'][0])
    spt_avg, spt_std, spt_ontime = evaluate_schedule_monte_carlo(spt_order, jobs, num_machines, K=K) if uncertainty_mode \
        else (simulate_schedule_uncertain(spt_order, jobs, num_machines), 0, 0)
    st.markdown(f"### 📦 SPT")
    st.write(f"**Avg Makespan:** {spt_avg}, **Std:** {spt_std}, **On-time Rate:** {spt_ontime}")

    # GA Schedule
    def initialize_population(n, size):
        base = list(range(n))
        return [random.sample(base, n) for _ in range(size)]

    def crossover(p1, p2):
        a, b = sorted(random.sample(range(len(p1)), 2))
        child = [None]*len(p1)
        child[a:b] = p1[a:b]
        fill = [x for x in p2 if x not in child[a:b]]
        j = 0
        for i in range(len(p1)):
            if child[i] is None:
                child[i] = fill[j]
                j += 1
        return child

    def mutate(chrom, rate=0.1):
        c = chrom.copy()
        for i in range(len(c)):
            if random.random() < rate:
                j = random.randint(0, len(c)-1)
                c[i], c[j] = c[j], c[i]
        return c

    pop_size = 30
    population = initialize_population(num_jobs, pop_size)
    best_order = None
    best_score = float('inf')
    best_progress = []

    for gen in range(generations):
        scores = []
        for chrom in population:
            score = evaluate_schedule_monte_carlo(chrom, jobs, num_machines, K=10)[0] if uncertainty_mode else simulate_schedule_uncertain(chrom, jobs, num_machines)
            scores.append(score)
            if score < best_score:
                best_score = score
                best_order = chrom
        best_progress.append(best_score)

        selected = random.choices(population, k=pop_size)
        children = []
        for i in range(0, pop_size, 2):
            c1 = mutate(crossover(selected[i], selected[i+1]))
            c2 = mutate(crossover(selected[i+1], selected[i]))
            children.extend([c1, c2])
        population = children

    ga_avg, ga_std, ga_ontime = evaluate_schedule_monte_carlo(best_order, jobs, num_machines, K=K) if uncertainty_mode \
        else (simulate_schedule_uncertain(best_order, jobs, num_machines), 0, 0)

    st.markdown(f"### 🤖 Genetic Algorithm")
    st.write(f"**Avg Makespan:** {ga_avg}, **Std:** {ga_std}, **On-time Rate:** {ga_ontime}")

    # GA Progress Chart
    st.markdown("#### 📈 GA Optimization Progress")
    fig, ax = plt.subplots()
    ax.plot(best_progress)
    ax.set_title("Best Makespan over Generations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Makespan")
    st.pyplot(fig)