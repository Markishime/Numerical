import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd
from typing import Tuple, List

# Set page configuration
st.set_page_config(page_title="Secant Method Solver", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 20px;}
    .stTextInput>div>input {border-radius: 5px; padding: 8px;}
    .stNumberInput>div>input {border-radius: 5px; padding: 8px;}
    .stSuccess {background-color: #e6ffe6; padding: 10px; border-radius: 5px;}
    .stInfo {background-color: #e6f7ff; padding: 10px; border-radius: 5px;}
    .stError {background-color: #ffe6e6; padding: 10px; border-radius: 5px;}
    h1 {color: #2c3e50;}
    h2 {color: #34495e;}
    .stMultiSelect {margin-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

# Function to evaluate the user-defined equation
def evaluate_function(expr: str, x: float) -> float:
    """Evaluates the mathematical expression at a given x value."""
    x_sym = sp.Symbol('x')
    try:
        func = sp.lambdify(x_sym, sp.sympify(expr), 'numpy')
        result = float(func(x))
        if np.isnan(result) or np.isinf(result):
            raise ValueError("Function returned NaN or infinity")
        return result
    except Exception as e:
        st.error(f"Error in expression: {e}")
        return None

# Secant Method Implementation
def secant_method(expr: str, x0: float, x1: float, tol: float = 0.5) -> Tuple[float, List[float], List[float], str, List[dict]]:
    """
    Implements the Secant Method with step tracking.
    
    Parameters:
    - expr: Mathematical expression as a string
    - x0, x1: Initial guesses
    - tol: Tolerance for convergence (fixed at 0.5)
    
    Returns:
    - root: Approximated root
    - x_values: List of x values
    - errors: List of absolute errors
    - message: Status message
    - steps: List of dictionaries with iteration details
    """
    x_values = [x0, x1]
    errors = []
    steps = [
        {"Iteration": 0, "x_n": x0, "f(x_n)": evaluate_function(expr, x0), "Error": None, "Formula": "Initial guess x0"},
        {"Iteration": 1, "x_n": x1, "f(x_n)": evaluate_function(expr, x1), "Error": None, "Formula": "Initial guess x1"}
    ]
    
    f_x0 = evaluate_function(expr, x0)
    f_x1 = evaluate_function(expr, x1)
    
    if f_x0 is None or f_x1 is None:
        return None, [], [], "Invalid function evaluation at initial guesses", []

    iter_count = 0
    max_iter_safety = 1000  # Safety limit to prevent infinite loops
    while iter_count < max_iter_safety:
        denominator = f_x1 - f_x0
        if abs(denominator) < 1e-12:
            return None, x_values, errors, "Near-zero denominator encountered", steps
        
        x2 = x1 - f_x1 * (x1 - x0) / denominator
        f_x2 = evaluate_function(expr, x2)
        
        if f_x2 is None:
            return None, x_values, errors, "Function evaluation failed during iteration", steps
        
        x_values.append(x2)
        error = abs(x2 - x1)
        errors.append(error)
        
        # Record step details
        formula = f"x_{iter_count+2} = x_{iter_count+1} - f(x_{iter_count+1}) * (x_{iter_count+1} - x_{iter_count}) / (f(x_{iter_count+1}) - f(x_{iter_count}))"
        steps.append({
            "Iteration": iter_count + 2,
            "x_n": x2,
            "f(x_n)": f_x2,
            "Error": error,
            "Formula": formula
        })
        
        if error < tol:
            return x2, x_values, errors, f"Converged after {iter_count + 1} iterations (absolute error < {tol})", steps
        
        x0, f_x0 = x1, f_x1
        x1, f_x1 = x2, f_x2
        iter_count += 1
    
    return x1, x_values, errors, "Did not converge within safety limit (best approximation)", steps

# Function to create visualization for a step
def plot_step(expr: str, x_prev: float, x_curr: float, x_next: float, iteration: int, x_range: np.ndarray, 
              step_details: dict, plot_width: float, plot_height: float) -> plt.Figure:
    """Creates a plot for a single Secant Method step."""
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    
    # Plot the function
    y_range = [evaluate_function(expr, x) for x in x_range]
    ax.plot(x_range, y_range, label=f"f(x) = {expr}", color='#3498db')
    
    # Plot the secant line
    f_prev = evaluate_function(expr, x_prev)
    f_curr = evaluate_function(expr, x_curr)
    if f_prev is not None and f_curr is not None:
        slope = (f_curr - f_prev) / (x_curr - x_prev) if x_curr != x_prev else 0
        secant_y = f_prev + slope * (x_range - x_prev)
        ax.plot(x_range, secant_y, linestyle='--', color='#e67e22', label='Secant Line')
    
    # Plot points with labels
    ax.scatter([x_prev, x_curr, x_next], 
               [f_prev, f_curr, evaluate_function(expr, x_next)], 
               color='red', zorder=5, label='Points')
    ax.annotate(f'x_{iteration-2}', (x_prev, f_prev), xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.annotate(f'x_{iteration-1}', (x_curr, f_curr), xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.annotate(f'x_{iteration}', (x_next, evaluate_function(expr, x_next)), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add step details as text box
    error_str = f"{step_details['Error']:.2e}" if step_details['Error'] is not None else "N/A"
    textstr = '\n'.join([
        f"x_{iteration} = {step_details['x_n']:.6f}",
        f"f(x_{iteration}) = {step_details['f(x_n)']:.6f}",
        f"Error = {error_str}",
        f"Formula: {step_details['Formula']}"
    ])
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Iteration {iteration}")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    return fig

# Streamlit App
def main():
    st.title("Secant Method Solver")
    st.markdown("A tool to find roots using the Secant Method with visualizations.")

    # Sidebar for Inputs
    with st.sidebar:
        st.header("Controls")
        expr = st.text_input("Function f(x)", value="x**2 - 4", help="e.g., 'x**2 - 4', 'cos(x) - x'")
        col1, col2 = st.columns(2)
        with col1:
            x0 = st.number_input("x0 (First Guess)", value=1.0, step=0.1)
        with col2:
            x1 = st.number_input("x1 (Second Guess)", value=2.0, step=0.1)
        
        # Plot size controls
        st.subheader("Visualization Settings")
        plot_width = st.slider("Plot Width (inches)", min_value=4.0, max_value=10.0, value=6.0, step=0.5)
        plot_height = st.slider("Plot Height (inches)", min_value=3.0, max_value=8.0, value=4.0, step=0.5)
        
        solve_button = st.button("Solve", use_container_width=True)

    # Main Content
    if solve_button:
        with st.spinner("Calculating..."):
            root, x_values, errors, message, steps = secant_method(expr, x0, x1)
        
        st.header("Results")
        col_result, col_export = st.columns([3, 1])
        with col_result:
            if root is not None:
                st.success(f"Root: **{root:.10f}**")
                st.info(message)
            else:
                st.error(f"Error: {message}")
        
        with col_export:
            if root is not None:
                # Construct DataFrame for iteration history
                iter_count = len(errors)
                df = pd.DataFrame({
                    "Iteration": list(range(iter_count)),
                    "x": x_values[2:2+iter_count],
                    "Error": errors
                })
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Export Results (CSV)",
                    data=csv,
                    file_name="secant_method_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        if root is not None:
            # Iteration Details
            with st.expander("Iteration History", expanded=True):
                st.dataframe(df.style.format({"x": "{:.10f}", "Error": "{:.2e}"}))

            # Solution Steps with Visualizations and Export
            with st.expander("Step-by-Step Solution", expanded=True):
                st.markdown("### Solution Steps")
                st.markdown("Detailed process of the Secant Method with calculations and visualizations:")
                
                # Create steps DataFrame
                steps_df = pd.DataFrame(steps)
                steps_df = steps_df[["Iteration", "x_n", "f(x_n)", "Error", "Formula"]]
                
                # Export steps table
                steps_csv = steps_df.to_csv(index=False)
                st.download_button(
                    label="Export Steps (CSV)",
                    data=steps_csv,
                    file_name="secant_method_steps.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Display steps table
                st.dataframe(
                    steps_df.style.format({
                        "x_n": "{:.10f}",
                        "f(x_n)": "{:.10f}",
                        "Error": "{:.2e}",
                        "Formula": "{}"
                    }, na_rep="N/A").set_properties(**{"Formula": {"text-align": "left"}})
                )

                # Interactive step navigation
                st.markdown("### Step Visualizations")
                st.markdown("Select which iterations to visualize:")
                iter_options = [f"Iteration {step['Iteration']}" for step in steps[2:]]  # Skip initial guesses
                default_iters = iter_options
                selected_iters = st.multiselect(
                    "Select Iterations",
                    options=iter_options,
                    default=default_iters,
                    help="Choose which iteration steps to display visualizations for."
                )
                
                # Generate visualizations
                if selected_iters:
                    x_range = np.linspace(min(x_values) - 1, max(x_values) + 1, 400)
                    for step in steps[2:]:
                        iter_num = step["Iteration"]
                        iter_label = f"Iteration {iter_num}"
                        if iter_label in selected_iters:
                            x_n = step["x_n"]
                            x_n_minus_1 = steps[iter_num - 1]["x_n"]
                            x_n_minus_2 = steps[iter_num - 2]["x_n"]
                            
                            st.markdown(f"**{iter_label}**")
                            fig = plot_step(expr, x_n_minus_2, x_n_minus_1, x_n, iter_num, x_range, 
                                          step, plot_width, plot_height)
                            st.pyplot(fig)
                            plt.close(fig)
                else:
                    st.info("No iterations selected. Select iterations above to view visualizations.")

            # Plots
            col_plot1, col_plot2 = st.columns(2)
            with col_plot1:
                st.subheader("Convergence Plot")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(range(len(errors)), errors, marker='o', label='Absolute Error', color='#2ecc71')
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Error (log scale)")
                ax.set_yscale('log')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                st.pyplot(fig)

            with col_plot2:
                st.subheader("Function Plot")
                x_range = np.linspace(min(x_values) - 1, max(x_values) + 1, 400)
                y_range = [evaluate_function(expr, x) for x in x_range]
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(x_range, y_range, label=f"f(x) = {expr}", color='#3498db')
                ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
                ax.axvline(root, color='r', linestyle='--', label=f"Root ≈ {root:.4f}")
                ax.set_xlabel("x")
                ax.set_ylabel("f(x)")
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                st.pyplot(fig)

            # Explanation
            st.header("Explanation")
            st.markdown(f"""
            The Secant Method approximates roots using secant lines. Results:

            - **Function**: `f(x) = {expr}`
            - **Initial Guesses**: `x0 = {x0}`, `x1 = {x1}`
            - **Iterations**: Converged in **{len(errors)}** steps.
            - **Convergence**: Stopped when absolute error (|x_{{n+1}} - x_n|) was less than tolerance (0.5).
            - **Root**: **{root:.10f}**, where `f(x) ≈ 0`.
            - **Insights**: Success depends on guesses near the root and function continuity.
            - **Advantages**: No derivatives needed, but may oscillate if function flattens.
            """)

    # Examples and Help
    with st.expander("Examples & Help"):
        st.markdown("""
        ### Test Cases
        - **f(x) = x² - 4**: Roots at ±2. Try `x0 = 1`, `x1 = 3`.
        - **f(x) = cos(x) - x**: Root ≈ 0.739. Try `x0 = 0`, `x1 = 1`.
        - **f(x) = x³ - x - 2**: Root ≈ 1.521. Try `x0 = 1`, `x1 = 2`.

        ### Tips
        - Use Python syntax: `x**2` for x², `cos(x)` for cosine, etc.
        - Ensure initial guesses are near the root for faster convergence.
        - Iterations stop automatically when absolute error is less than 0.5.
        - Adjust plot sizes for better visualization.
        - Select specific iterations to focus on steps.
        """)

if __name__ == "__main__":
    main()