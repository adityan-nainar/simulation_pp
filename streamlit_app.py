import random
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
import time

# ------------------------------------------
# Helper Functions (with Uncertainty)
# ------------------------------------------
def simulate_schedule_uncertain(order, jobs, num_machines, delta=0.2):
    machine_available = [0] * num_machines
    for job_index in order:
        job = jobs[job_index]
        current_time = 0
        for machine, proc_time in zip(job['route'], job['processing_times']):
            actual_proc = random.uniform(proc_time * (1 - delta), proc_time * (1 + delta))
            start = max(current_time, machine_available[machine])
            finish = start + actual_proc
            current_time = finish
            machine_available[machine] = finish
    return max(machine_available)

def evaluate_schedule_monte_carlo(order, jobs, num_machines, K=50, delta=0.2, due_buffer=1.5):
    makespans = []
    on_time_counts = []
    due_times = [sum(job['processing_times']) * due_buffer for job in jobs]
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
    return round(np.mean(makespans), 2), round(np.std(makespans), 2), round(np.mean(on_time_counts), 2)

# ------------------------------------------
# Streamlit UI
# ------------------------------------------
st.title("🧠 AI-Powered Job Shop Scheduling with Monte Carlo Robustness")

st.sidebar.header("Experiment Settings")
num_jobs = st.sidebar.slider("Number of Jobs", 10, 100, 30, step=10)
num_machines = st.sidebar.slider("Number of Machines", 5, 20, 10)
generations = st.sidebar.slider("GA Generations", 10, 200, 50, step=10)
num_runs = st.sidebar.slider("Number of Runs", 1, 20, 5, step=1)
K = st.sidebar.slider("Monte Carlo Simulations (K)", 10, 100, 30, step=10)
run = st.sidebar.button("Run Multi-Run Comparison")

if run:
    st.subheader("🔁 Running Multi-Run Experiments")
    ga_scores, spt_scores, fifo_scores = [], [], []
    ga_heatmap_data = []  # for animation

    for r in range(num_runs):
        jobs = []
        for i in range(num_jobs):
            length = random.randint(3, min(6, num_machines))
            route = random.sample(range(num_machines), length)
            times = [random.randint(10, 100) for _ in range(length)]
            jobs.append({'route': route, 'processing_times': times})

        fifo_order = list(range(num_jobs))
        spt_order = sorted(range(num_jobs), key=lambda j: jobs[j]['processing_times'][0])

        fifo_eval = evaluate_schedule_monte_carlo(fifo_order, jobs, num_machines, K)
        spt_eval = evaluate_schedule_monte_carlo(spt_order, jobs, num_machines, K)

        def init_pop(n, size):
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

        def mutate(c, rate=0.1):
            c = c.copy()
            for i in range(len(c)):
                if random.random() < rate:
                    j = random.randint(0, len(c)-1)
                    c[i], c[j] = c[j], c[i]
            return c

        pop_size = 30
        population = init_pop(num_jobs, pop_size)
        best = float('inf')
        best_order = None
        ga_progress = []

        for g in range(generations):
            fitnesses = []
            for chrom in population:
                score = evaluate_schedule_monte_carlo(chrom, jobs, num_machines, K=10)[0]
                fitnesses.append(score)
                if score < best:
                    best = score
                    best_order = chrom
            ga_progress.append(best)
            selected = random.choices(population, k=pop_size)
            children = []
            for i in range(0, pop_size, 2):
                c1 = mutate(crossover(selected[i], selected[i+1]))
                c2 = mutate(crossover(selected[i+1], selected[i]))
                children.extend([c1, c2])
            population = children

        ga_eval = evaluate_schedule_monte_carlo(best_order, jobs, num_machines, K)
        ga_scores.append(ga_eval)
        spt_scores.append(spt_eval)
        fifo_scores.append(fifo_eval)
        ga_heatmap_data.append(ga_progress)

    # Aggregated Results
    def avg_result(scores):
        arr = np.array(scores)
        return np.round(arr.mean(axis=0), 2)

    labels = ['GA', 'SPT', 'FIFO']
    result_df = pd.DataFrame({
        'Algorithm': labels,
        'Avg Makespan': [avg_result(ga_scores)[0], avg_result(spt_scores)[0], avg_result(fifo_scores)[0]],
        'Stability (std)': [avg_result(ga_scores)[1], avg_result(spt_scores)[1], avg_result(fifo_scores)[1]],
        'On-Time Rate': [avg_result(ga_scores)[2], avg_result(spt_scores)[2], avg_result(fifo_scores)[2]]
    })

    st.subheader("📊 Aggregated Performance over Runs")
    st.dataframe(result_df.set_index("Algorithm"))

    # Bar Chart Comparison
    st.markdown("### 📉 Makespan Comparison")
    fig1, ax1 = plt.subplots()
    ax1.bar(labels, result_df['Avg Makespan'], color=['#28a745', '#17a2b8', '#ffc107'])
    ax1.set_ylabel("Avg Makespan")
    ax1.set_title("Average Makespan across Algorithms")
    st.pyplot(fig1)

    # Streamlit-compatible GA Convergence Animation
    st.markdown("### 🎞️ GA Convergence Animation")
    placeholder = st.empty()
    heatmap_data = np.array(ga_heatmap_data)

    for frame in range(generations):
        fig, ax = plt.subplots()
        ax.plot(heatmap_data[:, frame], marker='o', color='purple')
        ax.set_title(f"GA Progress - Generation {frame+1}")
        ax.set_ylabel("Best Makespan")
        ax.set_xlabel("Run #")
        ax.set_ylim(np.min(heatmap_data)-10, np.max(heatmap_data)+10)
        placeholder.pyplot(fig)
        plt.close(fig)
        time.sleep(0.2)

    st.success("✅ Multi-run experiment complete!")