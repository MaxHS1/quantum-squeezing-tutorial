#!/usr/bin/env python
# coding: utf-8

# In[66]:


from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import RZZGate
import numpy as np
import pandas as pd
from qiskit import transpile
import matplotlib.pyplot as plt


# In[68]:


def apply_dipole_interaction(circuit, qubits, chi_t):
    n = len(qubits)
    for i in range(n):
        for j in range(i + 1, n):
            qi, qj = qubits[i], qubits[j]

            # ZZ term
            circuit.append(RZZGate(2 * chi_t), [qi, qj])

            # XX term 
            theta_x = -chi_t
            circuit.h(qi); circuit.h(qj)
            circuit.cx(qi, qj)
            circuit.rz(theta_x, qj)
            circuit.cx(qi, qj)
            circuit.h(qi); circuit.h(qj)

            # YY term
            theta_y = -chi_t
            circuit.rx(np.pi/2, qi); circuit.rx(np.pi/2, qj)
            circuit.cx(qi, qj)
            circuit.rz(theta_y, qj)
            circuit.cx(qi, qj)
            circuit.rx(-np.pi/2, qi); circuit.rx(-np.pi/2, qj)
            
    return circuit


# In[70]:


def calculate_jz_and_delta_jz(counts, n_qubits=3):
    expectation = 0
    expectation_sq = 0
    total_shots = sum(counts.values())
    
    for bitstring, count in counts.items():
        n_zeros = bitstring.count('0')  
        n_ones = bitstring.count('1')   
        jz_value = (n_zeros - n_ones) / 2
        prob = count / total_shots
        expectation += prob * jz_value
        expectation_sq += prob * (jz_value ** 2)
    
    delta_jz = (expectation_sq - expectation ** 2) ** 0.5
    return expectation, delta_jz


# In[76]:


# Parameters
N_values = range(1, 9)                 
chi_t_values = np.linspace(0, 1.0, 21)    
shots = 1000
backend = AerSimulator()


# In[ ]:


# Collect σ_min for each N 
rows = []

for n_qubits in N_values:
    sigmas_closed = []   # d = 1 
    sigmas_d0     = []   # d = 0 (SQL)

    for chi_t in chi_t_values:
        # d=1
        qc_closed = QuantumCircuit(n_qubits)
        qc_closed.h(range(n_qubits))
        apply_dipole_interaction(qc_closed, list(range(n_qubits)), chi_t)  
        qc_closed.measure_all()
        counts_closed = backend.run(transpile(qc_closed, backend), shots=shots).result().get_counts()
        _, delta_jz_closed = calculate_jz_and_delta_jz(counts_closed, n_qubits=n_qubits)
        sigmas_closed.append(float(delta_jz_closed)) 

        # d = 0 (SQL)
        qc_sql = QuantumCircuit(n_qubits)
        qc_sql.h(range(n_qubits))
        qc_sql.measure_all()
        counts_sql = backend.run(transpile(qc_sql, backend), shots=shots).result().get_counts()
        _, delta_jz_sql = calculate_jz_and_delta_jz(counts_sql, n_qubits=n_qubits)
        sigmas_d0.append(float(delta_jz_sql))       

    sigma_min_closed = float(np.min(sigmas_closed))
    sigma_min_d0     = float(np.min(sigmas_d0))

    sigma_sql_line = 1.0 / np.sqrt(n_qubits)

    rows.append({
        "N": n_qubits,
        "sigma_min_d0": sigma_min_d0,                   # blue squares
        "sigma_min_closed": sigma_min_closed,           # red circles
        "sigma_sql_baseline": sigma_sql_line            # dashed line
    })


# In[ ]:


df_sigma_vs_N = pd.DataFrame(rows).sort_values("N").reset_index(drop=True)
display(df_sigma_vs_N)


# In[ ]:


plt.figure()
plt.plot(df_sigma_vs_N["N"], df_sigma_vs_N["sigma_min_d0"],
         marker="s", linestyle="None", label="d = 0")
plt.plot(df_sigma_vs_N["N"], df_sigma_vs_N["sigma_min_closed"],
         marker="o", linestyle="None", label="d/(2π) = 1 MHz")
plt.plot(df_sigma_vs_N["N"], df_sigma_vs_N["sigma_sql_baseline"],
         linestyle="--", label="SQL")

plt.xlabel("N")
plt.ylabel(r"$\sigma_{\min}$")
plt.title(r"$\sigma_{\min}$ vs $N$ — d=0, d/(2\pi)=1\,\mathrm{MHz}$, and SQL")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

