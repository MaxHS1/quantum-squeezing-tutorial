"""
Spin Squeezing Analysis

Description:
This script performs a numerical simulation of spin squeezing dynamics in a system
of N interacting spins (typically N=2 or N=3). It calculates the time evolution
of a quantum state, initially a Coherent Spin State (CSS) oriented along the +x axis,
under a one-axis twisting-like Hamiltonian:
    H_dimless = sum_{i<j} [Sz_i Sz_j - 1/2 (Sx_i Sx_j + Sy_i Sy_j)]
The sign of this Hamiltonian can be controlled.

Key functionalities include:
- Calculation of time-evolved expectation values for collective spin operators (Jx, Jy, Jz).
- Determination of spin uncertainties (Delta_Jy, Delta_Jz).
- Finding the optimal rotation angle (theta_min) around Jx that minimizes Delta_Jy.
- Calculation of normalized noise parameters (sigma_y, sigma_z) to quantify squeezing.
- Comparison of simulation results with experimental data loaded from a CSV/TXT file.
- Generation of plots for visual analysis:
    - Expectation values vs. time.
    - Uncertainties vs. time.
    - Optimal angle theta_min vs. time.
    - Normalized noise sigma vs. time.
    - Detailed angle sweep plots (Delta_J vs. theta) at specific time points.
- Saving comprehensive simulation results to .mat and .csv files.
- Performing physics-based validation checks on the simulation outputs.

The calculations are performed using NumPy and SciPy for linear algebra and
evolution, without relying on external quantum computing libraries like QuTiP.

Required packages for this script:
- numpy
- matplotlib
- scipy
- scikit-learn

If you haven't installed them, you can typically install them using pip
(preferably in a virtual environment):

pip install numpy matplotlib scipy scikit-learn

For setting up a virtual environment and installing from requirements.txt,
please see the provided instructions or a standard Python tutorial.

Usage of this script:
    (Modify N_spins_to_simulate and current_hamiltonian_sign in main() as needed)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import scipy.io
import time
import os
import warnings
import csv
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", category=RuntimeWarning)


class SpinSqueezingSimulator:  # Renamed class for clarity
    output_dir_times_ns_max = None

    def __init__(
        self,
        N_spins,
        hamiltonian_sign_factor=-1.0,
        output_dir="theoretical_comparison_results",
    ):
        self.N = N_spins
        self.dim = 2**self.N
        self.hamiltonian_sign_factor = hamiltonian_sign_factor
        self.output_dir = output_dir
        self._create_output_directory()
        print(
            f"Initializing N={self.N} spin system (Hilbert space dimension: {self.dim})"
        )
        print(f"Hamiltonian Sign Factor: {self.hamiltonian_sign_factor}")
        print(f"Output directory: {self.output_dir}/")

        # Pauli matrices (multiplied by 0.5 to represent S=1/2 spin operators)
        self.sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
        self.sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
        self.si = np.eye(2, dtype=complex)  # Identity matrix for a single spin

        self._build_system()

    def _create_output_directory(self):
        """Creates the output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"✓ Created output directory: {self.output_dir}/")
        else:
            print(f"✓ Using existing output directory: {self.output_dir}/")

    def _get_output_path(self, filename):
        """Constructs the full path for a file in the output directory."""
        return os.path.join(self.output_dir, filename)

    def _build_system(self):
        """Builds the quantum system components: initial state, spin operators, and Hamiltonian."""
        print("Building quantum system components...")
        # Initial state: Coherent Spin State (CSS) along +x for all spins
        css_single = (1 / np.sqrt(2)) * np.array(
            [1, 1], dtype=complex
        )  # |+x> state for a single spin
        self.psi0 = css_single
        for _ in range(1, self.N):
            self.psi0 = np.kron(self.psi0, css_single)
        self.psi0 = self.psi0 / np.linalg.norm(self.psi0)  # Normalize

        # Build individual spin operators (Sx_i, Sy_i, Sz_i) for each spin
        self.Sx_ops, self.Sy_ops, self.Sz_ops = [], [], []
        for i in range(self.N):
            op_list_x, op_list_y, op_list_z = (
                [self.si] * self.N,
                [self.si] * self.N,
                [self.si] * self.N,
            )
            op_list_x[i], op_list_y[i], op_list_z[i] = self.sx, self.sy, self.sz

            # Construct full system operator using Kronecker products
            Sx_i, Sy_i, Sz_i = op_list_x[0], op_list_y[0], op_list_z[0]
            for j in range(1, self.N):
                Sx_i = np.kron(Sx_i, op_list_x[j])
                Sy_i = np.kron(Sy_i, op_list_y[j])
                Sz_i = np.kron(Sz_i, op_list_z[j])
            self.Sx_ops.append(Sx_i)
            self.Sy_ops.append(Sy_i)
            self.Sz_ops.append(Sz_i)

        # Total spin operators (collective operators)
        self.Jx = sum(self.Sx_ops)
        self.Jy = sum(self.Sy_ops)
        self.Jz = sum(self.Sz_ops)

        # Build the dimensionless Hamiltonian (OAT-like interaction)
        # H_dimless = sum_{i<j} [Sz_i Sz_j - 1/2 (Sx_i Sx_j + Sy_i Sy_j)]
        self.H_dimless = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(self.N):
            for j in range(i + 1, self.N):  # Sum over unique pairs only
                term = self.Sz_ops[i] @ self.Sz_ops[j] - 0.5 * (
                    self.Sx_ops[i] @ self.Sx_ops[j] + self.Sy_ops[i] @ self.Sy_ops[j]
                )
                self.H_dimless += term

        # Verify initial state expectation values
        jx0, jy0, jz0 = self.expectation_values(self.psi0)
        print(f"✓ Initial state: ⟨Jx⟩={jx0:.4f}, ⟨Jy⟩={jy0:.4f}, ⟨Jz⟩={jz0:.4f}")
        print(f"✓ Expected for CSS along +x: ⟨Jx⟩={self.N/2:.4f}, ⟨Jy⟩≈0, ⟨Jz⟩≈0")
        print("✓ System components built successfully.")

    def evolve(self, time_ns, d_MHz):
        """Evolves the initial state psi0 under the Hamiltonian H_dimless."""
        if d_MHz == 0:  # No interaction, no evolution from H_dimless
            return self.psi0
        # Phase factor for evolution: chi * t = (2*pi*d) * t
        # d_MHz is the interaction strength 'd' from Yifan's paper (Fig 2a)
        # time_ns is converted to seconds (1e-9), d_MHz to Hz (1e6)
        # phase = (2 * pi * d_MHz * 1e6) * (time_ns * 1e-9) = 2 * pi * d_MHz * time_ns * 1e-3
        phase = 2 * np.pi * d_MHz * time_ns * 1e-3
        # Unitary evolution operator U = exp(-i * H_eff * t)
        # H_eff = hamiltonian_sign_factor * H_dimless (scaled by phase)
        U = expm(-1j * phase * (self.hamiltonian_sign_factor * self.H_dimless))
        return U @ self.psi0

    def expectation_values(self, state):
        """Calculates expectation values <Jx>, <Jy>, <Jz> for a given state."""
        if state.ndim == 1:  # Ensure state is a column vector for matrix multiplication
            state_col = state[:, np.newaxis]
            state_bra = state_col.conj().T
        else:  # Assuming state is already a column vector (e.g., from Qobj.full())
            state_col = state
            state_bra = state.conj().T

        jx = np.real(state_bra @ self.Jx @ state_col)[0, 0]
        jy = np.real(state_bra @ self.Jy @ state_col)[0, 0]
        jz = np.real(state_bra @ self.Jz @ state_col)[0, 0]
        return jx, jy, jz

    def uncertainty(self, operator, state):
        """Calculates uncertainty (standard deviation) Delta J_k for a given operator and state."""
        if state.ndim == 1:
            state_col = state[:, np.newaxis]
            state_bra = state_col.conj().T
        else:
            state_col = state
            state_bra = state.conj().T

        mean_J = np.real(state_bra @ operator @ state_col)[0, 0]
        mean_J_sq = np.real(state_bra @ (operator @ operator) @ state_col)[0, 0]
        variance = mean_J_sq - mean_J**2
        # Ensure variance is non-negative due to potential floating point inaccuracies
        return np.sqrt(np.maximum(0, variance))

    def noise(self, operator, state, spin_length_val=None):
        """Calculates normalized uncertainty (sigma_k = Delta J_k / |J|)."""
        if spin_length_val is None:
            jx, jy, jz = self.expectation_values(state)
            spin_length_val = np.sqrt(jx**2 + jy**2 + jz**2)

        if spin_length_val < 1e-12:  # Avoid division by zero or very small numbers
            return np.inf
        return self.uncertainty(operator, state) / spin_length_val

    def rotate_x(self, state, theta_rad):
        """Rotates a state around the Jx axis by theta_rad."""
        U_rot = expm(-1j * theta_rad * self.Jx)
        return U_rot @ state

    def find_theta_min(self, time_ns, d_MHz):
        """
        Finds the optimal rotation angle theta_min that minimizes DeltaJy.
        Also returns the arrays of scanned thetas and corresponding DeltaJy, DeltaJz values.
        """
        evolved_state = self.evolve(time_ns, d_MHz)

        # --- MODIFICATION FOR N=3 THETA SCAN RANGE ---
        # To correctly calculate Delta_J(tau) and achieve the "w" shape for N=3,
        # the theta scan range must be 0 to 180 degrees (pi radians).
        # This matches c_testestestt.py and ensures the true minimum is found,
        # as theta_min can exceed 90 degrees for N=3 at certain times.
        if self.N == 3:
            theta_array_rad = np.linspace(0, np.pi, 181)  # Scan 0 to 180 degrees
        else:
            # For N=2 (or other N values), scan theta from 0 to 180 degrees (pi).
            theta_array_rad = np.linspace(0, np.pi, 181)
        # --- END MODIFICATION ---

        delta_jy_values = np.zeros(len(theta_array_rad))
        delta_jz_values = np.zeros(len(theta_array_rad))

        for i, theta_rad_current in enumerate(theta_array_rad):
            rotated_state = self.rotate_x(evolved_state, theta_rad_current)
            delta_jy_values[i] = self.uncertainty(self.Jy, rotated_state)
            delta_jz_values[i] = self.uncertainty(self.Jz, rotated_state)

        min_idx = np.argmin(delta_jy_values)

        # For N=2, if DeltaJy(theta) landscape is very flat, default to theta_min near pi/4 (45 deg).
        # This addresses potential "barcode" issues or arbitrary minimum for symmetric states.
        if self.N == 2:
            # Check if the peak-to-peak range of delta_jy_values is very small (landscape is flat)
            if np.ptp(delta_jy_values) < 1e-5:  # Threshold for flatness
                target_theta_rad_n2 = np.pi / 4  # Target 45 degrees for N=2
                # Find the index in theta_array_rad closest to the target
                min_idx = np.argmin(np.abs(theta_array_rad - target_theta_rad_n2))
                # Optional debug print:
                # print(f"DEBUG (N=2, t={time_ns:.2f}ns): DeltaJy landscape flat. Forcing theta_min near 45 deg (actual: {np.degrees(theta_array_rad[min_idx]):.1f}°). Range: {np.ptp(delta_jy_values):.2e}")

        return (
            theta_array_rad[min_idx],
            theta_array_rad,
            delta_jy_values,
            delta_jz_values,
        )

    def comprehensive_analysis(self, times_ns, d_MHz, exp_angle_data=None):
        """Performs a comprehensive simulation analysis over a range of evolution times."""
        n_times = len(times_ns)
        results = {
            "times_ns": np.array(times_ns),
            "N_spins": self.N,
            "d_MHz": d_MHz,
            "hamiltonian_sign_factor": self.hamiltonian_sign_factor,
            "jx_sim": np.zeros(n_times),
            "jy_sim": np.zeros(n_times),
            "jz_sim": np.zeros(n_times),
            "theta_min_deg_sim": np.zeros(n_times),
            "delta_jy_at_theta_min_sim": np.zeros(n_times),
            "delta_jz_at_theta_min_for_jy_sim": np.zeros(
                n_times
            ),  # DeltaJz at the angle that minimizes DeltaJy
            "sigma_y_at_theta_min_sim": np.zeros(n_times),
            "sigma_z_at_theta_min_for_jy_sim": np.zeros(n_times),
            "spin_length_sim": np.zeros(n_times),
            "delta_jy_at_theta_exp_sim": np.full(
                n_times, np.nan
            ),  # Sim DeltaJy at experimental theta_min
            "delta_jz_at_theta_exp_sim": np.full(
                n_times, np.nan
            ),  # Sim DeltaJz at experimental theta_min
        }

        interp_theta_exp = None
        # Create an interpolator for experimental theta_min if data is provided
        if (
            exp_angle_data
            and f"time_ns_angle" in exp_angle_data
            and "optimal_angle_deg_exp" in exp_angle_data
        ):
            exp_times = exp_angle_data[f"time_ns_angle"]
            exp_angles = exp_angle_data["optimal_angle_deg_exp"]
            # Ensure data is sorted by time for interpolation
            sort_idx = np.argsort(exp_times)
            exp_times_sorted = exp_times[sort_idx]
            exp_angles_sorted = exp_angles[sort_idx]
            if len(exp_times_sorted) > 1:  # Need at least 2 points for interpolation
                interp_theta_exp = interp1d(
                    exp_times_sorted,
                    exp_angles_sorted,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                print(
                    f"✓ Experimental theta_min interpolator created for N={self.N} diagnostics."
                )

        print(
            f"Starting comprehensive analysis for N={self.N}, d={d_MHz} MHz, H_sign={self.hamiltonian_sign_factor}..."
        )
        analysis_start_time = time.time()
        for i, t_ns_current in enumerate(times_ns):
            if i % (n_times // 10 or 1) == 0:  # Print progress roughly 10 times
                elapsed = time.time() - analysis_start_time
                eta_seconds = (elapsed / (i + 1e-9)) * (n_times - i) if i > 0 else 0
                print(f"  Progress: {i}/{n_times} (ETA: {eta_seconds:.1f}s)")

            evolved_state_current = self.evolve(t_ns_current, d_MHz)
            jx, jy, jz = self.expectation_values(evolved_state_current)
            results["jx_sim"][i], results["jy_sim"][i], results["jz_sim"][i] = (
                jx,
                jy,
                jz,
            )
            current_spin_length_sim = np.sqrt(jx**2 + jy**2 + jz**2)
            results["spin_length_sim"][i] = current_spin_length_sim

            # Find simulated theta_min and corresponding uncertainties
            theta_min_rad_current, _, all_delta_jy_at_t, all_delta_jz_at_t = (
                self.find_theta_min(t_ns_current, d_MHz)
            )
            results["theta_min_deg_sim"][i] = np.degrees(theta_min_rad_current)
            results["delta_jy_at_theta_min_sim"][i] = np.min(
                all_delta_jy_at_t
            )  # Smallest DeltaJy found
            min_idx_for_jy = np.argmin(all_delta_jy_at_t)
            results["delta_jz_at_theta_min_for_jy_sim"][i] = all_delta_jz_at_t[
                min_idx_for_jy
            ]

            # Calculate normalized noise (sigma) at the simulated theta_min
            state_rotated_by_sim_min_angle = self.rotate_x(
                evolved_state_current, theta_min_rad_current
            )
            jx_rot_sim, jy_rot_sim, jz_rot_sim = self.expectation_values(
                state_rotated_by_sim_min_angle
            )
            spin_length_rotated_state_sim = np.sqrt(
                jx_rot_sim**2 + jy_rot_sim**2 + jz_rot_sim**2
            )  # Spin length of the rotated state
            results["sigma_y_at_theta_min_sim"][i] = self.noise(
                self.Jy,
                state_rotated_by_sim_min_angle,
                spin_length_val=spin_length_rotated_state_sim,
            )
            results["sigma_z_at_theta_min_for_jy_sim"][i] = self.noise(
                self.Jz,
                state_rotated_by_sim_min_angle,
                spin_length_val=spin_length_rotated_state_sim,
            )

            # If experimental theta_min data is available, calculate sim uncertainties at those angles
            if interp_theta_exp is not None:
                theta_exp_deg_interp = interp_theta_exp(t_ns_current)
                if not np.isnan(theta_exp_deg_interp):
                    theta_exp_rad_interp = np.radians(theta_exp_deg_interp)
                    state_rotated_by_exp_theta = self.rotate_x(
                        evolved_state_current, theta_exp_rad_interp
                    )
                    results["delta_jy_at_theta_exp_sim"][i] = self.uncertainty(
                        self.Jy, state_rotated_by_exp_theta
                    )
                    results["delta_jz_at_theta_exp_sim"][i] = self.uncertainty(
                        self.Jz, state_rotated_by_exp_theta
                    )

        total_analysis_time = time.time() - analysis_start_time
        print(f"✓ Comprehensive analysis completed in {total_analysis_time:.1f}s")
        return results

    def load_yifan_data(self, filepath, data_category, N_target):
        """Loads experimental data from Yifan's CSV/TXT file for a specific category and N_target."""
        relevant_file_id = f"{data_category}_S{N_target}.csv"  # e.g., "angle_S3.csv"
        print(
            f"Attempting to load experimental data for '{relevant_file_id}' from: {filepath}"
        )

        exp_data = {"time_s": [], f"time_ns_{data_category}": []}
        if data_category == "angle":
            exp_data["optimal_angle_deg_exp"] = []
        elif data_category == "expectation":
            exp_data.update({"jx_exp": [], "jy_exp": [], "jz_exp": []})
        elif data_category == "uncertainty":
            exp_data.update(
                {"theta_min_deg_exp": [], "delta_jy_exp": [], "delta_jz_exp": []}
            )
        else:
            print(f"Error: Unknown data_category '{data_category}'")
            return None

        try:
            with open(filepath, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)  # Read header row
                try:
                    file_col_idx = header.index("File")
                    time_s_col_idx = header.index("time_s")
                except ValueError:
                    print(
                        f"Error: CSV header in {filepath} missing 'File' or 'time_s'."
                    )
                    return None

                for row_num, row in enumerate(reader):
                    if not row or len(row) <= 1:
                        continue  # Skip empty or very short rows
                    try:
                        if row[file_col_idx] == relevant_file_id:
                            current_time_s = float(row[time_s_col_idx])
                            exp_data["time_s"].append(current_time_s)
                            exp_data[f"time_ns_{data_category}"].append(
                                current_time_s * 1e9
                            )  # Convert s to ns

                            if data_category == "angle":
                                if len(row) > 2:
                                    exp_data["optimal_angle_deg_exp"].append(
                                        float(row[2])
                                    )
                                else:
                                    exp_data["optimal_angle_deg_exp"].append(
                                        np.nan
                                    )  # Handle missing data
                            elif data_category == "expectation":
                                if len(row) > 4:  # Jx, Jy, Jz
                                    exp_data["jx_exp"].append(float(row[2]))
                                    exp_data["jy_exp"].append(float(row[3]))
                                    exp_data["jz_exp"].append(float(row[4]))
                                else:  # Handle missing data
                                    exp_data["jx_exp"].append(np.nan)
                                    exp_data["jy_exp"].append(np.nan)
                                    exp_data["jz_exp"].append(np.nan)
                            elif data_category == "uncertainty":
                                if len(row) > 4:  # theta_min, delta_jy, delta_jz
                                    exp_data["theta_min_deg_exp"].append(float(row[2]))
                                    exp_data["delta_jy_exp"].append(float(row[3]))
                                    exp_data["delta_jz_exp"].append(float(row[4]))
                                else:  # Handle missing data
                                    exp_data["theta_min_deg_exp"].append(np.nan)
                                    exp_data["delta_jy_exp"].append(np.nan)
                                    exp_data["delta_jz_exp"].append(np.nan)
                    except (
                        ValueError
                    ) as e:  # Handle rows with non-numeric data where numbers are expected
                        print(
                            f"Skipping malformed data row {row_num+2} for {relevant_file_id}: {row} - {e}"
                        )
                        # Append NaN to keep array lengths consistent if time was already added
                        if data_category == "angle":
                            exp_data["optimal_angle_deg_exp"].append(np.nan)
                        elif data_category == "expectation":
                            exp_data["jx_exp"].append(np.nan)
                            exp_data["jy_exp"].append(np.nan)
                            exp_data["jz_exp"].append(np.nan)
                        elif data_category == "uncertainty":
                            exp_data["theta_min_deg_exp"].append(np.nan)
                            exp_data["delta_jy_exp"].append(np.nan)
                            exp_data["delta_jz_exp"].append(np.nan)
                        # If time was added but data was bad, pop the time to maintain consistency
                        if exp_data["time_s"] and len(exp_data["time_s"]) > len(
                            exp_data.get(list(exp_data.keys())[-1], [])
                        ):
                            exp_data["time_s"].pop()
                            exp_data[f"time_ns_{data_category}"].pop()

            if not exp_data[
                f"time_ns_{data_category}"
            ]:  # Check if any data was actually loaded for the relevant_file_id
                print(f"No data for File='{relevant_file_id}' in {filepath}")
                return None

            # Convert lists to numpy arrays
            for key in exp_data:
                exp_data[key] = np.array(exp_data[key])

            # Filter data to match simulation time range if output_dir_times_ns_max is set
            if SpinSqueezingSimulator.output_dir_times_ns_max is not None:
                max_time_ns_sim = SpinSqueezingSimulator.output_dir_times_ns_max
                time_key_ns = f"time_ns_{data_category}"
                valid_mask = exp_data[time_key_ns] <= max_time_ns_sim
                for key_to_filter in exp_data.keys():
                    exp_data[key_to_filter] = exp_data[key_to_filter][valid_mask]

            print(
                f"✓ Loaded {len(exp_data[f'time_ns_{data_category}'])} exp data points for {relevant_file_id} from {filepath}"
            )
            return exp_data
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
        except Exception as e:
            print(f"Error loading exp data for {relevant_file_id} from {filepath}: {e}")
            return None

    def plot_main_results(self, results, exp_data_bundle=None, save_filename_prefix=""):
        """Plots the main simulation results and compares with experimental data if provided."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.suptitle(
            f'Spin Squeezing: N={self.N}, d={results["d_MHz"]} MHz, H_sign={results["hamiltonian_sign_factor"]}',
            fontsize=16,
        )
        times_sim = results["times_ns"]

        # Plot 1: Expectation Values <Jk>(tau)
        ax1 = axes[0, 0]
        ax1.plot(times_sim, results["jx_sim"], "r-", label="⟨Jx⟩_sim", lw=2)
        ax1.plot(times_sim, results["jy_sim"], "g-", label="⟨Jy⟩_sim", lw=2)
        ax1.plot(times_sim, results["jz_sim"], "b-", label="⟨Jz⟩_sim", lw=2)
        if exp_data_bundle and "expectation" in exp_data_bundle:
            exp_expect = exp_data_bundle["expectation"]
            if f"time_ns_expectation" in exp_expect and "jx_exp" in exp_expect:
                ax1.plot(
                    exp_expect[f"time_ns_expectation"],
                    exp_expect["jx_exp"],
                    "ro",
                    ms=3,
                    alpha=0.6,
                    label="⟨Jx⟩_exp",
                )
            if f"time_ns_expectation" in exp_expect and "jy_exp" in exp_expect:
                ax1.plot(
                    exp_expect[f"time_ns_expectation"],
                    exp_expect["jy_exp"],
                    "go",
                    ms=3,
                    alpha=0.6,
                    label="⟨Jy⟩_exp",
                )
            if f"time_ns_expectation" in exp_expect and "jz_exp" in exp_expect:
                ax1.plot(
                    exp_expect[f"time_ns_expectation"],
                    exp_expect["jz_exp"],
                    "bo",
                    ms=3,
                    alpha=0.6,
                    label="⟨Jz⟩_exp",
                )
        ax1.axhline(0, color="k", linestyle=":", alpha=0.5)
        ax1.set_xlabel("Evolution time τ (ns)")
        ax1.set_ylabel("Expectation value")
        ax1.set_title("Expectation Values ⟨Jk⟩(τ)")
        ax1.legend()
        ax1.grid(True, alpha=0.4)
        if (
            self.N == 3 and results.get("hamiltonian_sign_factor", 1) < 0
        ):  # Specific N=3 paper reference
            ax1.axhline(
                -0.5,
                color="grey",
                linestyle="--",
                label="Target Jx Min (Paper Fig2a)",
                alpha=0.7,
            )
            ax1.legend()  # Update legend if line added

        # Plot 2: Uncertainties DeltaJ(tau)
        ax2 = axes[0, 1]
        ax2.plot(
            times_sim,
            results["delta_jy_at_theta_min_sim"],
            "b-",
            label="ΔJy_sim (at θ_min_sim)",
            lw=2,
        )
        ax2.plot(
            times_sim,
            results["delta_jz_at_theta_min_for_jy_sim"],
            "r-",
            label="ΔJz_sim (at θ_min_sim for Jy)",
            lw=2,
        )
        # Plot simulated uncertainties at experimental theta_min if available
        if "delta_jy_at_theta_exp_sim" in results:
            ax2.plot(
                times_sim,
                results["delta_jy_at_theta_exp_sim"],
                "c--",
                label="ΔJy_sim (at θ_min_exp)",
                lw=1.5,
                alpha=0.8,
            )
        if "delta_jz_at_theta_exp_sim" in results:
            ax2.plot(
                times_sim,
                results["delta_jz_at_theta_exp_sim"],
                "m--",
                label="ΔJz_sim (at θ_min_exp for Jy)",
                lw=1.5,
                alpha=0.8,
            )
        if exp_data_bundle and "uncertainty" in exp_data_bundle:
            exp_uncert = exp_data_bundle["uncertainty"]
            if f"time_ns_uncertainty" in exp_uncert and "delta_jy_exp" in exp_uncert:
                ax2.plot(
                    exp_uncert[f"time_ns_uncertainty"],
                    exp_uncert["delta_jy_exp"],
                    "bo",
                    ms=3,
                    alpha=0.6,
                    label="ΔJy_exp",
                )
            if f"time_ns_uncertainty" in exp_uncert and "delta_jz_exp" in exp_uncert:
                ax2.plot(
                    exp_uncert[f"time_ns_uncertainty"],
                    exp_uncert["delta_jz_exp"],
                    "ro",
                    ms=3,
                    alpha=0.6,
                    label="ΔJz_exp",
                )
        sql_delta_j = np.sqrt(self.N / 4.0)  # Standard Quantum Limit for Delta J
        ax2.axhline(
            sql_delta_j,
            color="k",
            linestyle="--",
            label=f"SQL (ΔJ ≈ {sql_delta_j:.3f})",
            alpha=0.7,
        )
        ax2.set_xlabel("Evolution time τ (ns)")
        ax2.set_ylabel("Uncertainty ΔJ")
        ax2.set_title("Uncertainties ΔJ(τ)")
        ax2.legend(fontsize="small")
        ax2.grid(True, alpha=0.4)

        # Plot 3: Optimal Angle theta_min(tau)
        ax3 = axes[1, 0]
        ax3.plot(times_sim, results["theta_min_deg_sim"], "k-", lw=2, label="θ_min_sim")
        if exp_data_bundle and "angle" in exp_data_bundle:
            exp_angle = exp_data_bundle["angle"]
            if f"time_ns_angle" in exp_angle and "optimal_angle_deg_exp" in exp_angle:
                ax3.plot(
                    exp_angle[f"time_ns_angle"],
                    exp_angle["optimal_angle_deg_exp"],
                    "o",
                    color="orange",
                    ms=4,
                    alpha=0.7,
                    label=f"θ_min_exp (N={self.N})",
                )
        ax3.set_xlabel("Evolution time τ (ns)")
        ax3.set_ylabel("Optimal Angle θ_min (degrees)")
        ax3.set_title("Evolution of Optimal Angle θ_min(τ)")
        ax3.grid(True, alpha=0.4)
        if self.N == 3:  # N=3 specific reference lines
            ax3.axhline(
                90,
                color="r",
                linestyle=":",
                alpha=0.7,
                label="Target Plateau ~90° (N=3)",  # Maintained original label, can be adjusted
            )
        elif self.N == 2:  # N=2 specific reference lines
            ax3.axhline(
                45, color="g", linestyle=":", alpha=0.7, label="Expected ~45° (N=2)"
            )
        ax3.legend()
        # --- MODIFICATION FOR Y-LIMIT CONSISTENT WITH 0-180 DEGREE SCAN ---
        ax3.set_ylim(0, 180)  # Ensure full range is visible for all N
        # --- END MODIFICATION ---

        # Plot 4: Normalized Noise sigma(tau)
        ax4 = axes[1, 1]
        sigma_y_sim_plot = np.copy(results["sigma_y_at_theta_min_sim"])
        sigma_z_sim_plot = np.copy(results["sigma_z_at_theta_min_for_jy_sim"])
        num_spikes_y_sim = np.sum(np.isinf(sigma_y_sim_plot))
        num_spikes_z_sim = np.sum(np.isinf(sigma_z_sim_plot))
        plot_cap = 5.0  # Cap infinities for plotting
        sigma_y_sim_plot[np.isinf(sigma_y_sim_plot)] = plot_cap
        sigma_z_sim_plot[np.isinf(sigma_z_sim_plot)] = plot_cap
        ax4.plot(
            times_sim, sigma_y_sim_plot, "b-", label="σ_y_sim (at θ_min_sim)", lw=2
        )
        ax4.plot(
            times_sim,
            sigma_z_sim_plot,
            "r-",
            label="σ_z_sim (at θ_min_sim for Jy)",
            lw=2,
        )
        # Calculate and plot experimental normalized uncertainties if data is available
        if (
            exp_data_bundle
            and "uncertainty" in exp_data_bundle
            and "expectation" in exp_data_bundle
        ):
            exp_uncert = exp_data_bundle["uncertainty"]
            exp_expect = exp_data_bundle["expectation"]
            time_ns_uncert_exp = exp_uncert.get(f"time_ns_uncertainty", np.array([]))
            delta_jy_exp = exp_uncert.get("delta_jy_exp", np.array([]))
            delta_jz_exp = exp_uncert.get("delta_jz_exp", np.array([]))
            time_ns_expect_exp = exp_expect.get(f"time_ns_expectation", np.array([]))
            jx_exp = exp_expect.get("jx_exp", np.array([]))
            jy_exp = exp_expect.get("jy_exp", np.array([]))
            jz_exp = exp_expect.get("jz_exp", np.array([]))

            # Ensure all necessary data arrays are present and have compatible lengths
            if (
                len(time_ns_uncert_exp) > 0
                and len(time_ns_expect_exp) > 1
                and len(delta_jy_exp) == len(time_ns_uncert_exp)
                and len(delta_jz_exp) == len(time_ns_uncert_exp)
                and len(jx_exp) == len(time_ns_expect_exp)
                and len(jy_exp) == len(time_ns_expect_exp)
                and len(jz_exp) == len(time_ns_expect_exp)
            ):

                # Interpolate experimental expectation values to the time points of experimental uncertainties
                interp_jx = interp1d(
                    time_ns_expect_exp,
                    jx_exp,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                interp_jy = interp1d(
                    time_ns_expect_exp,
                    jy_exp,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                interp_jz = interp1d(
                    time_ns_expect_exp,
                    jz_exp,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )

                jx_exp_at_uncert_times = interp_jx(time_ns_uncert_exp)
                jy_exp_at_uncert_times = interp_jy(time_ns_uncert_exp)
                jz_exp_at_uncert_times = interp_jz(time_ns_uncert_exp)

                J_exp_at_uncert_times = np.sqrt(
                    jx_exp_at_uncert_times**2
                    + jy_exp_at_uncert_times**2
                    + jz_exp_at_uncert_times**2
                )

                sigma_y_exp = np.full_like(delta_jy_exp, np.nan, dtype=float)
                sigma_z_exp = np.full_like(delta_jz_exp, np.nan, dtype=float)

                valid_J_mask = (J_exp_at_uncert_times >= 1e-12) & ~np.isnan(
                    J_exp_at_uncert_times
                )

                sigma_y_exp[valid_J_mask] = (
                    delta_jy_exp[valid_J_mask] / J_exp_at_uncert_times[valid_J_mask]
                )
                sigma_y_exp[~valid_J_mask] = np.inf

                sigma_z_exp[valid_J_mask] = (
                    delta_jz_exp[valid_J_mask] / J_exp_at_uncert_times[valid_J_mask]
                )
                sigma_z_exp[~valid_J_mask] = np.inf

                # Cap infinities for plotting
                sigma_y_exp_plot = np.copy(sigma_y_exp)
                sigma_z_exp_plot = np.copy(sigma_z_exp)
                sigma_y_exp_plot[np.isinf(sigma_y_exp_plot)] = plot_cap
                sigma_z_exp_plot[np.isinf(sigma_z_exp_plot)] = plot_cap

                ax4.plot(
                    time_ns_uncert_exp,
                    sigma_y_exp_plot,
                    "co",
                    ms=3,
                    alpha=0.6,
                    label="σ_y_exp (calc)",
                )  # Changed color for distinction
                ax4.plot(
                    time_ns_uncert_exp,
                    sigma_z_exp_plot,
                    "mo",
                    ms=3,
                    alpha=0.6,
                    label="σ_z_exp (calc)",
                )  # Changed color
                ax4.legend(fontsize="small")  # Update legend
        sql_sigma = 1.0 / np.sqrt(self.N)  # Standard Quantum Limit for sigma
        ax4.axhline(
            sql_sigma,
            color="k",
            linestyle="--",
            label=f"SQL (σ ≈ {sql_sigma:.3f})",
            alpha=0.7,
        )
        ax4.set_xlabel("Evolution time τ (ns)")
        ax4.set_ylabel("Normalized Uncertainty σ")
        ax4.set_title(
            f"Normalized Noise σ(τ) (Sim Spikes Y:{num_spikes_y_sim}, Z:{num_spikes_z_sim})"
        )
        ax4.legend(fontsize="small")
        ax4.grid(True, alpha=0.4)
        # Dynamically set y-limit for sigma plot based on finite values
        valid_sigma_values = np.concatenate(
            (
                sigma_y_sim_plot[np.isfinite(sigma_y_sim_plot)],
                sigma_z_sim_plot[np.isfinite(sigma_z_sim_plot)],
                [sql_sigma * 2.5],
            )
        )
        if (
            "sigma_y_exp" in locals() and len(sigma_y_exp[np.isfinite(sigma_y_exp)]) > 0
        ):  # check if sigma_y_exp was calculated
            valid_sigma_values = np.concatenate(
                (
                    valid_sigma_values,
                    sigma_y_exp[np.isfinite(sigma_y_exp)],
                    sigma_z_exp[np.isfinite(sigma_z_exp)],
                )
            )
        if len(valid_sigma_values[np.isfinite(valid_sigma_values)]) > 0:
            max_finite_val = np.max(valid_sigma_values[np.isfinite(valid_sigma_values)])
            ax4.set_ylim(
                0,
                min(
                    plot_cap * 1.1,
                    max_finite_val if not np.isnan(max_finite_val) else plot_cap * 1.1,
                ),
            )
        else:
            ax4.set_ylim(0, plot_cap * 1.1)

        plt.tight_layout(
            rect=[0, 0, 1, 0.96]
        )  # Adjust layout to make space for suptitle
        if save_filename_prefix:
            output_path = self._get_output_path(
                f"{save_filename_prefix}_main_plots_exp_comp.png"
            )
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f"✓ Main results plot saved: {output_path}")
        return fig

    def detailed_angle_sweep_plots(self, time_points_ns, d_MHz, save_plots=True):
        """Generates and saves detailed plots of DeltaJ vs. theta for specific time points."""
        print(
            f"\n=== DETAILED ANGLE SWEEP PLOTS & DATA (H_sign={self.hamiltonian_sign_factor}) ==="
        )
        all_sweep_data_for_mat = {}  # For saving data to .mat file
        num_time_points = len(time_points_ns)
        ncols = 3
        nrows = (num_time_points + ncols - 1) // ncols  # Calculate rows needed
        fig_combined, axes_combined = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
        )
        axes_combined = axes_combined.flatten()  # Flatten for easy iteration
        h_sign_str = "posH" if self.hamiltonian_sign_factor > 0 else "negH"

        for idx, t_ns_current in enumerate(time_points_ns):
            print(f"\nAnalyzing angle sweep for time point: τ = {t_ns_current} ns")
            (
                theta_min_y_rad_at_t,
                thetas_scanned_rad,
                delta_jy_values_at_t,
                delta_jz_values_at_t,
            ) = self.find_theta_min(t_ns_current, d_MHz)

            theta_array_deg = np.degrees(thetas_scanned_rad)
            theta_min_deg = np.degrees(theta_min_y_rad_at_t)

            # Store data for .mat file
            sweep_data_mat = {
                "time_ns": t_ns_current,
                "theta_degrees": theta_array_deg,
                "delta_jy": delta_jy_values_at_t,
                "delta_jz": delta_jz_values_at_t,
                "theta_min_degrees": theta_min_deg,
                "delta_jy_min_value": np.min(delta_jy_values_at_t),
            }
            all_sweep_data_for_mat[
                f"t_{t_ns_current}ns_N{self.N}_d{d_MHz}MHz_{h_sign_str}"
            ] = sweep_data_mat

            if idx < len(axes_combined):  # Plot if there's an axis available
                ax = axes_combined[idx]
                ax.plot(
                    theta_array_deg, delta_jy_values_at_t, "b-", label="ΔJy(θ)", lw=1.5
                )
                ax.plot(
                    theta_array_deg, delta_jz_values_at_t, "r-", label="ΔJz(θ)", lw=1.5
                )
                ax.axvline(
                    theta_min_deg,
                    color="k",
                    linestyle="--",
                    label=f"θ_min={theta_min_deg:.1f}°",
                    lw=1.5,
                )
                min_jy_val = np.min(delta_jy_values_at_t)
                ax.plot(
                    theta_min_deg,
                    min_jy_val,
                    "bo",
                    ms=5,
                    label=f"Min ΔJy={min_jy_val:.3f}",
                )
                ax.set_xlabel("Rotation angle θ (degrees)")
                ax.set_ylabel("Uncertainty ΔJ")
                ax.set_title(f"τ = {t_ns_current} ns")
                ax.legend(fontsize="small")
                ax.grid(True, alpha=0.3)
                # --- MODIFICATION FOR X-LIMIT CONSISTENT WITH 0-180 DEGREE SCAN ---
                ax.set_xlim(0, 180)  # Ensure full scan range is plotted for all N
                # --- END MODIFICATION ---

            if save_plots:
                # Save raw data for each sweep to CSV
                csv_filename = f"angle_sweep_t{t_ns_current}ns_N{self.N}_d{d_MHz}MHz_{h_sign_str}_raw_data.csv"
                csv_path = self._get_output_path(csv_filename)
                scan_range_deg = 180  # Scan range is now consistently 0-180
                header_info = (
                    f"# Angle sweep data for N={self.N}, d={d_MHz}MHz, t={t_ns_current}ns, H_sign={self.hamiltonian_sign_factor}\n"
                    f"# theta_min (for Jy) = {theta_min_deg:.3f} degrees (found in [0,{scan_range_deg}] deg range)\n"
                    f"theta_degrees,delta_jy,delta_jz"
                )
                np.savetxt(
                    csv_path,
                    np.column_stack(
                        [theta_array_deg, delta_jy_values_at_t, delta_jz_values_at_t]
                    ),
                    delimiter=",",
                    header=header_info,
                    comments="",
                )
                print(f"  ✓ Raw CSV data saved: {csv_path}")

        # Remove any unused subplots
        for i in range(num_time_points, nrows * ncols):
            if i < len(axes_combined):
                fig_combined.delaxes(axes_combined[i])

        fig_combined.suptitle(
            f"Angle Sweeps ΔJ(θ) (Scan range 0-180°) for N={self.N}, d={d_MHz} MHz, H_sign={self.hamiltonian_sign_factor}",  # Updated title
            fontsize=16,
        )
        fig_combined.tight_layout(rect=[0, 0, 1, 0.95])

        if save_plots:
            combined_plot_filename = (
                f"angle_sweeps_combined_N{self.N}_d{d_MHz}MHz_{h_sign_str}.png"
            )
            combined_plot_path = self._get_output_path(combined_plot_filename)
            fig_combined.savefig(combined_plot_path, dpi=200, bbox_inches="tight")
            print(f"\n✓ Combined angle sweep plot saved: {combined_plot_path}")

            all_sweeps_mat_filename = (
                f"all_angle_sweeps_data_N{self.N}_d{d_MHz}MHz_{h_sign_str}.mat"
            )
            all_sweeps_mat_path = self._get_output_path(all_sweeps_mat_filename)
            scipy.io.savemat(all_sweeps_mat_path, all_sweep_data_for_mat)
            print(f"✓ All sweep data saved to MAT file: {all_sweeps_mat_path}")

        return fig_combined, all_sweep_data_for_mat

    def validate_physics(self, results, exp_data_bundle=None):
        """Performs physics validation checks based on simulation results and experimental data."""
        print(
            f"\n=== PHYSICS VALIDATION (N={self.N}, d={results['d_MHz']} MHz, H_sign={results['hamiltonian_sign_factor']}) ==="
        )

        # 1. Simulated <Jx> behavior
        min_jx_sim = np.min(results["jx_sim"])
        jx_goes_negative_sim = (
            min_jx_sim < -0.1
        )  # Threshold for "significantly negative"
        print(f"1. Simulated ⟨Jx⟩ behavior:\n   Min ⟨Jx⟩_sim: {min_jx_sim:.6f}")
        if self.N == 3:  # N=3 specific Jx check from paper
            expected_jx_behavior_n3 = (
                jx_goes_negative_sim
                if results["hamiltonian_sign_factor"] < 0
                else not jx_goes_negative_sim
            )
            status_jx_n3 = (
                "✓ SIM MATCHES EXPECTATION for N=3"
                if expected_jx_behavior_n3
                else "✗ SIM PROBLEM for N=3"
            )
            print(
                f"   Simulated ⟨Jx⟩ goes significantly negative (<-0.1): {jx_goes_negative_sim}"
            )
            print(
                f"   Status for N=3 Sim (H_sign={results['hamiltonian_sign_factor']}): {status_jx_n3} (Expected for Fig 2a with neg H: Jx goes negative)"
            )
        elif self.N == 2:  # N=2 behavior can differ
            print(
                f"   For N=2, ⟨Jx⟩_sim behavior typically differs from N=3 paper pattern."
            )

        # 2. theta_min behavior
        theta_min_sim_deg = results["theta_min_deg_sim"]
        times_sim_ns = results["times_ns"]
        print(f"\n2. θ_min behavior:")
        if self.N == 3:  # N=3 specific checks based on Yifan Paper Fig 2c pattern
            initial_sim_theta_n3 = (
                theta_min_sim_deg[0] if len(theta_min_sim_deg) > 0 else np.nan
            )
            starts_fig2c_n3 = (
                40 <= initial_sim_theta_n3 <= 55
            )  # Original check for paper consistency
            # If H_sign is positive, initial theta_min might be different (e.g. > 90 as per c_testestestt.py for H_sign=1)
            # The "w" shape in DeltaJy for N=3, H_sign=1 (positive) arises from theta_min varying across 0-180.
            # The paper's Fig 2c is for negative H_sign (implicitly, as Jx goes negative).
            # For positive H_sign, the theta_min evolution might differ from Fig 2c.
            print(f"   N=3 Simulated Initial θ_min (0 ns): {initial_sim_theta_n3:.1f}°")
            if (
                results["hamiltonian_sign_factor"] < 0
            ):  # Checks specific to paper's Fig 2c (negative H)
                print(
                    f"     (Target for Paper Fig 2c [neg H_sign] ~45-50°: {starts_fig2c_n3})"
                )
                plateau_mask_n3 = (times_sim_ns >= 330) & (times_sim_ns <= 660)
                plateaus_near_90_fig2c_n3 = False
                if (
                    np.any(plateau_mask_n3)
                    and len(theta_min_sim_deg[plateau_mask_n3]) > 0
                ):
                    plateau_segment_n3 = theta_min_sim_deg[plateau_mask_n3]
                    plateaus_near_90_fig2c_n3 = np.all(
                        (plateau_segment_n3 >= 85) & (plateau_segment_n3 <= 95)
                    )
                print(
                    f"     N=3 Sim Plateaus near 90° (330-660ns, target 85-95° for Fig 2c [neg H_sign]): {plateaus_near_90_fig2c_n3}"
                )
                final_drop_mask_n3 = times_sim_ns > 660
                drops_sharply_fig2c_n3 = False
                if (
                    np.any(final_drop_mask_n3)
                    and len(theta_min_sim_deg[final_drop_mask_n3]) > 0
                ):
                    final_val_n3 = theta_min_sim_deg[final_drop_mask_n3][-1]
                    drops_sharply_fig2c_n3 = final_val_n3 < 60
                print(
                    f"     N=3 Sim Drops sharply at end (after 660ns, target <60° for Fig 2c [neg H_sign]): {drops_sharply_fig2c_n3}"
                )
                overall_fig2c_match_n3 = (
                    starts_fig2c_n3
                    and plateaus_near_90_fig2c_n3
                    and drops_sharply_fig2c_n3
                )
                print(
                    f"   Status for N=3 Sim θ_min vs Paper Fig 2c pattern (neg H_sign): {'✓ MATCHES Fig 2c pattern' if overall_fig2c_match_n3 else '✗ PROBLEM (does not match Fig 2c pattern)'}"
                )
            else:  # Positive H_sign
                print(
                    f"     (For positive H_sign, θ_min behavior may differ from Paper Fig 2c [neg H_sign]. Check plots.)"
                )

        elif self.N == 2:  # N=2 specific checks
            mean_theta_n2 = (
                np.mean(theta_min_sim_deg[1:])
                if len(theta_min_sim_deg) > 1
                else theta_min_sim_deg[0] if len(theta_min_sim_deg) > 0 else np.nan
            )
            # Check if theta_min is consistently around 45 degrees, or if the flatness condition was triggered
            is_around_45_n2 = (
                40 <= mean_theta_n2 <= 50
                or np.all(
                    np.isclose(
                        theta_min_sim_deg[times_sim_ns > 0.1 * times_sim_ns[-1]],
                        45,
                        atol=10,
                    )
                )
                if len(times_sim_ns) > 0
                else False
            )
            print(
                f"   N=2 Simulated Mean θ_min (excluding t=0 if possible): {mean_theta_n2:.1f}° (Often expected near 45°: {is_around_45_n2})"
            )

        # RMSE for theta_min (Sim vs Exp) - Generic for any N
        if exp_data_bundle and "angle" in exp_data_bundle:
            exp_angle = exp_data_bundle["angle"]
            exp_times_ns_angle = exp_angle.get(f"time_ns_angle", np.array([]))
            exp_theta_deg = exp_angle.get("optimal_angle_deg_exp", np.array([]))
            if len(exp_times_ns_angle) > 1 and len(exp_theta_deg) == len(
                exp_times_ns_angle
            ):
                sort_indices_sim = np.argsort(times_sim_ns)
                times_sim_ns_sorted = times_sim_ns[sort_indices_sim]
                theta_min_sim_deg_sorted = theta_min_sim_deg[sort_indices_sim]
                interp_func = interp1d(
                    times_sim_ns_sorted,
                    theta_min_sim_deg_sorted,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                sim_theta_interp = interp_func(exp_times_ns_angle)
                valid_comparison_mask = ~np.isnan(sim_theta_interp) & ~np.isnan(
                    exp_theta_deg
                )
                if np.sum(valid_comparison_mask) > 1:
                    rmse = np.sqrt(
                        mean_squared_error(
                            exp_theta_deg[valid_comparison_mask],
                            sim_theta_interp[valid_comparison_mask],
                        )
                    )
                    print(
                        f"   RMSE between Sim and Exp θ_min (N={self.N}): {rmse:.2f}°"
                    )
                else:
                    print(
                        f"   Not enough overlapping valid data points for θ_min RMSE (N={self.N})."
                    )
            else:
                print(
                    f"   Exp N={self.N} angle data for θ_min not sufficient for RMSE."
                )
        else:
            print(f"   Exp N={self.N} angle data for θ_min not loaded for RMSE.")

        # 3. Spike feature in simulated normalized noise (sigma_sim)
        num_spikes = np.sum(np.isinf(results["sigma_y_at_theta_min_sim"])) + np.sum(
            np.isinf(results["sigma_z_at_theta_min_for_jy_sim"])
        )
        print(
            f"\n3. Spike feature in simulated normalized noise (σ_sim):\n   Number of np.inf spikes: {num_spikes}"
        )
        if (
            self.N == 2
        ):  # For N=2, expect spikes if Jx_sim goes to zero (spin length in denominator of sigma)
            jx_near_zero_mask = np.isclose(
                results["jx_sim"], 0, atol=1e-3
            )  # Check where Jx is close to zero
            spikes_at_jx_zero = False
            if np.any(
                jx_near_zero_mask
            ):  # Only check for spikes if Jx actually gets near zero
                spikes_at_jx_zero = np.any(
                    np.isinf(results["sigma_y_at_theta_min_sim"][jx_near_zero_mask])
                ) or np.any(
                    np.isinf(
                        results["sigma_z_at_theta_min_for_jy_sim"][jx_near_zero_mask]
                    )
                )
            print(
                f"   For N=2, spikes expected when Jx_sim ≈ 0. Spikes found at Jx_sim ≈ 0: {spikes_at_jx_zero}"
            )
            status_spikes_n2 = "? NO SPIKES or NOT AT Jx=0"
            if num_spikes > 0 and spikes_at_jx_zero:
                status_spikes_n2 = "✓ SPIKES PRESENT AS EXPECTED"
            elif num_spikes > 0:
                status_spikes_n2 = "✓ SPIKES PRESENT (but Jx not always near zero)"
            print(f"   Status (N=2): {status_spikes_n2}")
        else:  # For N=3 or other N
            print(
                f"   Status: {'✓ INF SPIKES PRESENT' if num_spikes > 0 else f'? NO INF SPIKES (check if expected for N={self.N})'}"
            )

        # 4. Squeezing (sigma_y_sim vs SQL)
        sql_sigma = 1.0 / np.sqrt(self.N)
        finite_sigma_y_sim = results["sigma_y_at_theta_min_sim"][
            np.isfinite(results["sigma_y_at_theta_min_sim"])
        ]
        min_finite_sigma_y_sim = (
            np.min(finite_sigma_y_sim) if len(finite_sigma_y_sim) > 0 else np.nan
        )
        squeezing_achieved_sim = (not np.isnan(min_finite_sigma_y_sim)) and (
            min_finite_sigma_y_sim < sql_sigma
        )
        print(
            f"\n4. Squeezing (σ_y_sim vs SQL):\n   Min finite σ_y_sim: {min_finite_sigma_y_sim:.4f} (SQL: {sql_sigma:.4f})\n   Status: {'✓ SQUEEZING ACHIEVED (SIM)' if squeezing_achieved_sim else '✗ NO SQUEEZING (SIM)'}"
        )

        # 5. RMSE for Expectation Values (Sim vs Exp) - Generic for any N
        if exp_data_bundle and "expectation" in exp_data_bundle:
            print(f"\n5. RMSE for Expectation Values (N={self.N} Sim vs Exp):")
            exp_expect = exp_data_bundle["expectation"]
            exp_times_ns_expect = exp_expect.get(f"time_ns_expectation", np.array([]))
            for comp in ["jx", "jy", "jz"]:
                sim_data = results[f"{comp}_sim"]
                exp_data_comp = exp_expect.get(f"{comp}_exp", np.array([]))
                if len(exp_times_ns_expect) > 1 and len(exp_data_comp) == len(
                    exp_times_ns_expect
                ):
                    sort_indices_sim = np.argsort(times_sim_ns)
                    times_sim_ns_sorted = times_sim_ns[sort_indices_sim]
                    sim_data_sorted = sim_data[sort_indices_sim]
                    interp_func = interp1d(
                        times_sim_ns_sorted,
                        sim_data_sorted,
                        kind="linear",
                        bounds_error=False,
                        fill_value=np.nan,
                    )
                    sim_interp = interp_func(exp_times_ns_expect)
                    valid_comparison_mask = ~np.isnan(sim_interp) & ~np.isnan(
                        exp_data_comp
                    )
                    if np.sum(valid_comparison_mask) > 1:
                        rmse = np.sqrt(
                            mean_squared_error(
                                exp_data_comp[valid_comparison_mask],
                                sim_interp[valid_comparison_mask],
                            )
                        )
                        print(f"   RMSE for ⟨{comp.upper()}⟩: {rmse:.4f}")
                    else:
                        print(
                            f"   Not enough overlapping valid data points for ⟨{comp.upper()}⟩ RMSE."
                        )
                else:
                    print(
                        f"   Exp N={self.N} data for ⟨{comp.upper()}⟩ not sufficient for RMSE."
                    )
        else:
            print(f"\n5. Exp N={self.N} expectation data not loaded for RMSE.")

        # 6. RMSE for Uncertainties (DeltaJ) (Sim vs Exp) - Generic for any N
        if exp_data_bundle and "uncertainty" in exp_data_bundle:
            print(f"\n6. RMSE for Uncertainties (N={self.N} Sim vs Exp):")
            exp_uncert = exp_data_bundle["uncertainty"]
            exp_times_ns_uncert = exp_uncert.get(f"time_ns_uncertainty", np.array([]))
            # DeltaJy
            sim_djy = results["delta_jy_at_theta_min_sim"]
            exp_djy = exp_uncert.get("delta_jy_exp", np.array([]))
            if len(exp_times_ns_uncert) > 1 and len(exp_djy) == len(
                exp_times_ns_uncert
            ):
                sort_indices_sim = np.argsort(times_sim_ns)
                times_sim_ns_sorted = times_sim_ns[sort_indices_sim]
                sim_djy_sorted = sim_djy[sort_indices_sim]
                interp_func = interp1d(
                    times_sim_ns_sorted,
                    sim_djy_sorted,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                sim_interp_djy = interp_func(exp_times_ns_uncert)
                valid_mask_djy = ~np.isnan(sim_interp_djy) & ~np.isnan(exp_djy)
                if np.sum(valid_mask_djy) > 1:
                    rmse_djy = np.sqrt(
                        mean_squared_error(
                            exp_djy[valid_mask_djy], sim_interp_djy[valid_mask_djy]
                        )
                    )
                    print(f"   RMSE for ΔJy: {rmse_djy:.4f}")
                else:
                    print(f"   Not enough overlapping valid data points for ΔJy RMSE.")
            else:
                print(f"   Exp N={self.N} data for ΔJy not sufficient for RMSE.")
            # DeltaJz
            sim_djz = results["delta_jz_at_theta_min_for_jy_sim"]
            exp_djz = exp_uncert.get("delta_jz_exp", np.array([]))
            if len(exp_times_ns_uncert) > 1 and len(exp_djz) == len(
                exp_times_ns_uncert
            ):
                sort_indices_sim = np.argsort(times_sim_ns)
                times_sim_ns_sorted = times_sim_ns[sort_indices_sim]
                sim_djz_sorted = sim_djz[sort_indices_sim]
                interp_func = interp1d(
                    times_sim_ns_sorted,
                    sim_djz_sorted,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                sim_interp_djz = interp_func(exp_times_ns_uncert)
                valid_mask_djz = ~np.isnan(sim_interp_djz) & ~np.isnan(exp_djz)
                if np.sum(valid_mask_djz) > 1:
                    rmse_djz = np.sqrt(
                        mean_squared_error(
                            exp_djz[valid_mask_djz], sim_interp_djz[valid_mask_djz]
                        )
                    )
                    print(f"   RMSE for ΔJz: {rmse_djz:.4f}")
                else:
                    print(f"   Not enough overlapping valid data points for ΔJz RMSE.")
            else:
                print(f"   Exp N={self.N} data for ΔJz not sufficient for RMSE.")
        else:
            print(f"\n6. Exp N={self.N} uncertainty data not loaded for RMSE.")

        # 7. RMSE for Normalized Uncertainties (sigma) (Sim vs Exp) - Generic for any N
        if (
            exp_data_bundle
            and "uncertainty" in exp_data_bundle
            and "expectation" in exp_data_bundle
        ):
            print(f"\n7. RMSE for Normalized Uncertainties (N={self.N} Sim vs Exp):")
            exp_uncert = exp_data_bundle["uncertainty"]
            exp_expect = exp_data_bundle["expectation"]
            time_ns_uncert_exp = exp_uncert.get(f"time_ns_uncertainty", np.array([]))
            delta_jy_exp = exp_uncert.get("delta_jy_exp", np.array([]))
            delta_jz_exp = exp_uncert.get("delta_jz_exp", np.array([]))
            time_ns_expect_exp = exp_expect.get(f"time_ns_expectation", np.array([]))
            jx_exp_raw = exp_expect.get("jx_exp", np.array([]))
            jy_exp_raw = exp_expect.get("jy_exp", np.array([]))
            jz_exp_raw = exp_expect.get("jz_exp", np.array([]))

            if not (
                len(time_ns_uncert_exp) > 0
                and len(time_ns_expect_exp) > 1
                and len(delta_jy_exp) == len(time_ns_uncert_exp)
                and len(delta_jz_exp) == len(time_ns_uncert_exp)
                and len(jx_exp_raw) == len(time_ns_expect_exp)
                and len(jy_exp_raw) == len(time_ns_expect_exp)
                and len(jz_exp_raw) == len(time_ns_expect_exp)
            ):
                print(
                    f"   Experimental data for N={self.N} normalized uncertainties not sufficient for RMSE."
                )
            else:
                interp_jx = interp1d(
                    time_ns_expect_exp,
                    jx_exp_raw,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                interp_jy = interp1d(
                    time_ns_expect_exp,
                    jy_exp_raw,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                interp_jz = interp1d(
                    time_ns_expect_exp,
                    jz_exp_raw,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                jx_exp_interp = interp_jx(time_ns_uncert_exp)
                jy_exp_interp = interp_jy(time_ns_uncert_exp)
                jz_exp_interp = interp_jz(time_ns_uncert_exp)
                J_exp_interp = np.sqrt(
                    jx_exp_interp**2 + jy_exp_interp**2 + jz_exp_interp**2
                )

                sigma_y_exp = np.full_like(delta_jy_exp, np.nan, dtype=float)
                sigma_z_exp = np.full_like(delta_jz_exp, np.nan, dtype=float)
                valid_J_mask = (J_exp_interp >= 1e-12) & ~np.isnan(J_exp_interp)
                sigma_y_exp[valid_J_mask] = (
                    delta_jy_exp[valid_J_mask] / J_exp_interp[valid_J_mask]
                )
                sigma_z_exp[valid_J_mask] = (
                    delta_jz_exp[valid_J_mask] / J_exp_interp[valid_J_mask]
                )

                # RMSE for sigma_y
                sim_sigma_y = results["sigma_y_at_theta_min_sim"]
                sort_indices_sim_sy = np.argsort(times_sim_ns)
                times_sim_ns_sorted_sy = times_sim_ns[sort_indices_sim_sy]
                sim_sigma_y_sorted = sim_sigma_y[sort_indices_sim_sy]
                interp_func_sy = interp1d(
                    times_sim_ns_sorted_sy,
                    sim_sigma_y_sorted,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                sim_interp_sigma_y = interp_func_sy(time_ns_uncert_exp)
                valid_mask_sigma_y = (
                    ~np.isnan(sim_interp_sigma_y)
                    & ~np.isnan(sigma_y_exp)
                    & np.isfinite(sim_interp_sigma_y)
                    & np.isfinite(sigma_y_exp)
                )
                if np.sum(valid_mask_sigma_y) > 1:
                    rmse_sigma_y = np.sqrt(
                        mean_squared_error(
                            sigma_y_exp[valid_mask_sigma_y],
                            sim_interp_sigma_y[valid_mask_sigma_y],
                        )
                    )
                    print(f"   RMSE for σ_y: {rmse_sigma_y:.4f}")
                else:
                    print(f"   Not enough overlapping valid data points for σ_y RMSE.")

                # RMSE for sigma_z
                sim_sigma_z = results["sigma_z_at_theta_min_for_jy_sim"]
                sort_indices_sim_sz = np.argsort(times_sim_ns)
                times_sim_ns_sorted_sz = times_sim_ns[sort_indices_sim_sz]
                sim_sigma_z_sorted = sim_sigma_z[sort_indices_sim_sz]
                interp_func_sz = interp1d(
                    times_sim_ns_sorted_sz,
                    sim_sigma_z_sorted,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                sim_interp_sigma_z = interp_func_sz(time_ns_uncert_exp)
                valid_mask_sigma_z = (
                    ~np.isnan(sim_interp_sigma_z)
                    & ~np.isnan(sigma_z_exp)
                    & np.isfinite(sim_interp_sigma_z)
                    & np.isfinite(sigma_z_exp)
                )
                if np.sum(valid_mask_sigma_z) > 1:
                    rmse_sigma_z = np.sqrt(
                        mean_squared_error(
                            sigma_z_exp[valid_mask_sigma_z],
                            sim_interp_sigma_z[valid_mask_sigma_z],
                        )
                    )
                    print(f"   RMSE for σ_z: {rmse_sigma_z:.4f}")
                else:
                    print(f"   Not enough overlapping valid data points for σ_z RMSE.")
        else:
            print(
                f"\n7. Exp N={self.N} data not loaded for normalized uncertainty RMSE."
            )
        print("=" * 50)


def main():

    # --- Select parameters for the run ---
    N_spins_to_simulate = int(os.environ.get('N_SPINS', 4))  # Read from environment, default to 4
    current_hamiltonian_sign = (
        1.0  # Change to 1.0 (positive H) or -1.0 (negative H) as needed
    )
    # ---

    interaction_strength_d_MHz = 1.0  # Interaction strength 'd' in MHz

    # Adjust time parameters if N=2 requires different ranges than N=3
    if N_spins_to_simulate == 2:
        simulation_times_ns = np.linspace(
            0, 700, 151
        )  # Example, adjust if needed for N=2
        detailed_sweep_times_ns = [
            0,
            100,
            200,
            333,
            400,
            500,
            600,
            670,
        ]  # Example for N=2
        print(f"INFO: Using N=2 specific time parameters.")
    else:  # Default to N=3 times (or generic times)
        simulation_times_ns = np.linspace(0, 700, 151)
        detailed_sweep_times_ns = [0, 50, 91.9, 150, 250, 330, 400, 600, 670]

    SpinSqueezingSimulator.output_dir_times_ns_max = simulation_times_ns[
        -1
    ]  # Used to filter exp data

    # Attempt to locate experimental data file
    exp_data_filepath_csv = "yifan_data.csv"
    exp_data_filepath_txt = "yifan_data.txt"  # Alternative format if CSV not found
    exp_data_filepath = None
    if os.path.exists(exp_data_filepath_csv):
        exp_data_filepath = exp_data_filepath_csv
        print(f"Using experimental data file: '{exp_data_filepath_csv}'")
    elif os.path.exists(exp_data_filepath_txt):
        exp_data_filepath = (
            exp_data_filepath_txt  # Fallback to .txt if .csv is not present
        )
        print(
            f"Found '{exp_data_filepath_txt}' for experimental data (treating as CSV)."
        )
    else:
        print(
            f"Warning: Neither '{exp_data_filepath_csv}' nor '{exp_data_filepath_txt}' found.\nExperimental data will not be loaded or plotted."
        )

    print(
        f"\nSimulation Parameters:\n  N_spins: {N_spins_to_simulate}, d: {interaction_strength_d_MHz} MHz, H_sign: {current_hamiltonian_sign}"
    )
    squeezer_system = SpinSqueezingSimulator(
        N_spins_to_simulate,
        hamiltonian_sign_factor=current_hamiltonian_sign,
        output_dir="theoretical_comparison_results",
    )  # Updated output dir

    # Load experimental data bundle (will be empty if file not found or no data for N_target)
    exp_data_bundle = {}
    if exp_data_filepath:
        print(
            f"\n{'-'*20}Loading Experimental Data for N={N_spins_to_simulate}{'-'*20}"
        )
        for category in ["angle", "expectation", "uncertainty"]:
            loaded_data = squeezer_system.load_yifan_data(
                exp_data_filepath, data_category=category, N_target=N_spins_to_simulate
            )
            if loaded_data:
                exp_data_bundle[category] = loaded_data
        if not exp_data_bundle:
            print(
                f"No experimental data successfully loaded for N={N_spins_to_simulate}."
            )

    # Pass experimental angle data to comprehensive_analysis for diagnostics
    exp_angles_for_diag = exp_data_bundle.get("angle")

    print(f"\n{'-'*20}Running Comprehensive Analysis{'-'*20}")
    simulation_results = squeezer_system.comprehensive_analysis(
        simulation_times_ns,
        interaction_strength_d_MHz,
        exp_angle_data=exp_angles_for_diag,
    )

    print(f"\n{'-'*20}Plotting Main Results{'-'*20}")
    h_sign_prefix = "posH" if current_hamiltonian_sign > 0 else "negH"
    main_plot_filename = (
        f"N{N_spins_to_simulate}_d{interaction_strength_d_MHz}MHz_{h_sign_prefix}"
    )
    squeezer_system.plot_main_results(
        simulation_results,
        exp_data_bundle=exp_data_bundle,
        save_filename_prefix=main_plot_filename,
    )

    print(f"\n{'-'*20}Running Detailed Angle Sweep Analysis{'-'*20}")
    squeezer_system.detailed_angle_sweep_plots(
        detailed_sweep_times_ns, interaction_strength_d_MHz, save_plots=True
    )

    print(f"\n{'-'*20}Validating Physics{'-'*20}")
    squeezer_system.validate_physics(
        simulation_results, exp_data_bundle=exp_data_bundle
    )

    print(f"\n{'-'*20}Saving Comprehensive Results{'-'*20}")
    results_mat_filename = f"comprehensive_sim_N{N_spins_to_simulate}_d{interaction_strength_d_MHz}MHz_{h_sign_prefix}.mat"
    # Ensure all elements in the dict are suitable for savemat
    mat_save_dict = {
        k: v
        for k, v in simulation_results.items()
        if not isinstance(v, list)
        or all(isinstance(i, (int, float, complex, str, np.ndarray)) for i in v)
    }
    scipy.io.savemat(
        squeezer_system._get_output_path(results_mat_filename), mat_save_dict
    )
    print(
        f"✓ Comprehensive simulation data saved to: {squeezer_system._get_output_path(results_mat_filename)}"
    )

    # Save key results to CSV
    csv_summary_filename = f"summary_results_N{N_spins_to_simulate}_d{interaction_strength_d_MHz}MHz_{h_sign_prefix}.csv"
    # Define columns to save based on what's in results
    csv_columns_data = [
        simulation_results["times_ns"],
        simulation_results["jx_sim"],
        simulation_results["jy_sim"],
        simulation_results["jz_sim"],
        simulation_results["theta_min_deg_sim"],
        simulation_results["delta_jy_at_theta_min_sim"],
        simulation_results["delta_jz_at_theta_min_for_jy_sim"],
        simulation_results["sigma_y_at_theta_min_sim"],
        simulation_results["sigma_z_at_theta_min_for_jy_sim"],
        simulation_results["spin_length_sim"],
    ]
    csv_header_list = [
        "time_ns",
        "jx_sim",
        "jy_sim",
        "jz_sim",
        "theta_min_deg_sim",
        "delta_jy_at_theta_min_sim",
        "delta_jz_at_theta_min_for_jy_sim",
        "sigma_y_at_theta_min_sim",
        "sigma_z_at_theta_min_for_jy_sim",
        "spin_length_sim",
    ]
    # Add diagnostic columns if they exist (from using exp_angle_data)
    if "delta_jy_at_theta_exp_sim" in simulation_results:
        csv_columns_data.append(simulation_results["delta_jy_at_theta_exp_sim"])
        csv_header_list.append("delta_jy_at_theta_exp_sim")
    if "delta_jz_at_theta_exp_sim" in simulation_results:
        csv_columns_data.append(simulation_results["delta_jz_at_theta_exp_sim"])
        csv_header_list.append("delta_jz_at_theta_exp_sim")

    data_for_csv = np.column_stack(csv_columns_data)
    csv_header = ",".join(csv_header_list)
    np.savetxt(
        squeezer_system._get_output_path(csv_summary_filename),
        data_for_csv,
        delimiter=",",
        header=csv_header,
        comments="",
    )
    print(
        f"✓ Key results summary saved to CSV: {squeezer_system._get_output_path(csv_summary_filename)}"
    )

    print(f"\n{'-'*20}\nAnalysis Finished. Showing plots...\n{'-'*20}")
    plt.show()


if __name__ == "__main__":
    main()
