import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd
from typing import Tuple, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# Set page configuration
st.set_page_config(
    page_title="Secant Method Solver",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .step-box {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Gemini 1.5 Flash LLM
def initialize_gemini_llm() -> ChatGoogleGenerativeAI:
    """Initialize Gemini 1.5 Flash model using Google API key."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY environment variable not set. Please set it to use Gemini 1.5 Flash.")
            return None
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.7
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize Gemini 1.5 Flash: {e}")
        return None

# Validate expression using Gemini 1.5 Flash
def validate_expression(expr: str, llm: ChatGoogleGenerativeAI) -> str:
    """Validate mathematical expression using Gemini 1.5 Flash."""
    if not llm:
        return f"Invalid: {expr} (Reason: Gemini LLM not initialized)"
    prompt = PromptTemplate(
        input_variables=["expression"],
        template="""
        You are a mathematical expert solver. Validate the following mathematical expression for use in a Python-based secant method solver.
        The expression should be a valid Python expression using 'x' as the variable, supporting 'e' for the exponential constant,
        and common functions like 'sin', 'cos', 'exp', etc. If the expression is invalid, suggest a corrected version.
        If valid, return the expression unchanged.

        Expression: {expression}

        Output format:
        - Valid: [expression]
        - Invalid: [corrected_expression] (Reason: [reason])
        """
    )
    try:
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(expression=expr)
        return response
    except Exception as e:
        st.error(f"Gemini API error during expression validation: {e}")
        return f"Invalid: {expr} (Reason: Gemini API error)"

# Function to evaluate the user-defined equation
def evaluate_function(expr: str, x: float) -> float:
    """Evaluates the mathematical expression at a given x value, supporting 'e' as the natural logarithm base."""
    x_sym = sp.Symbol('x')
    try:
        # Define the mathematical constant e
        e = sp.E  # This is sympy's representation of e
        namespace = {'e': e, 'E': e}  # Allow both 'e' and 'E' as the constant
        func = sp.lambdify(x_sym, sp.sympify(expr, locals=namespace), 'numpy')
        result = float(func(x))
        if np.isnan(result) or np.isinf(result):
            raise ValueError("Function returned NaN or infinity")
        return result
    except Exception as e:
        st.error(f"Error in expression: {e}")
        return None

# Secant Method Implementation with detailed steps
def secant_method(expr: str, x0: float, x1: float, tol: float = 0.5, max_iter: int = 100) -> Tuple[float, List[float], List[float], str, List[dict]]:
    """
    Implements the Secant Method with detailed step tracking.
    Stops when absolute error is less than 0.5 or maximum iterations reached.
    Handles near-zero denominators by perturbing guesses.
    """
    x_values = [x0, x1]
    errors = []
    steps = []
    
    f_x0 = evaluate_function(expr, x0)
    f_x1 = evaluate_function(expr, x1)
    
    if f_x0 is None or f_x1 is None:
        return None, [], [], "Invalid function evaluation at initial guesses", []

    iter_count = 0
    while iter_count < max_iter:
        # Calculate denominator
        denominator = f_x1 - f_x0
        if abs(denominator) < 1e-12:
            # Perturb x1 slightly to avoid near-zero denominator
            perturbation = 1e-6 * (1 + abs(x1))
            x1 += perturbation
            f_x1 = evaluate_function(expr, x1)
            if f_x1 is None:
                return None, x_values, errors, f"Function evaluation failed after perturbation at x1 = {x1}", steps
            denominator = f_x1 - f_x0
            if abs(denominator) < 1e-12:
                return None, x_values, errors, "Near-zero denominator persists after perturbation", steps
            st.warning(f"Near-zero denominator detected at iteration {iter_count + 1}. Perturbed x1 by {perturbation:.2e}.")

        # Calculate next approximation
        x2 = x1 - f_x1 * (x1 - x0) / denominator
        f_x2 = evaluate_function(expr, x2)
        
        if f_x2 is None:
            return None, x_values, errors, "Function evaluation failed during iteration", steps
        
        # Calculate error
        error = abs(x2 - x1)
        errors.append(error)
        x_values.append(x2)
        
        # Record step details
        step = {
            "Iteration": iter_count + 1,
            "x_n": x2,
            "f(x_n)": f_x2,
            "Error": error,
            "Formula": f"x_{iter_count+2} = x_{iter_count+1} - f(x_{iter_count+1}) * (x_{iter_count+1} - x_{iter_count}) / (f(x_{iter_count+1}) - f(x_{iter_count}))",
            "Explanation": f"Step {iter_count + 1}: Using points (x_{iter_count}, f(x_{iter_count})) and (x_{iter_count+1}, f(x_{iter_count+1})), we calculate x_{iter_count+2} = {x2:.6f}"
        }
        steps.append(step)
        
        if error < tol:
            return x2, x_values, errors, f"Converged after {iter_count + 1} iterations (absolute error < {tol})", steps
        
        x0, f_x0 = x1, f_x1
        x1, f_x1 = x2, f_x2
        iter_count += 1
    
    return x1, x_values, errors, f"Did not converge within {max_iter} iterations (best approximation)", steps

# Function to create visualization for a step
def plot_step(expr: str, x_prev: float, x_curr: float, x_next: float, iteration: int, x_range: np.ndarray, 
              step_details: dict, plot_width: float, plot_height: float) -> plt.Figure:
    """Creates a detailed plot for a single Secant Method step."""
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    
    # Plot function
    y_range = [evaluate_function(expr, x) for x in x_range]
    ax.plot(x_range, y_range, label=f"f(x) = {expr}", color='#1f77b4', linewidth=2)
    
    # Plot secant line
    f_prev = evaluate_function(expr, x_prev)
    f_curr = evaluate_function(expr, x_curr)
    if f_prev is not None and f_curr is not None:
        slope = (f_curr - f_prev) / (x_curr - x_prev) if x_curr != x_prev else 0
        secant_y = f_prev + slope * (x_range - x_prev)
        ax.plot(x_range, secant_y, linestyle='--', color='#ff7f0e', label='Secant Line', linewidth=1.5)
    
    # Plot points
    points = [(x_prev, f_prev), (x_curr, f_curr), (x_next, evaluate_function(expr, x_next))]
    ax.scatter([p[0] for p in points], [p[1] for p in points], 
               color='#d62728', zorder=5, label='Points', s=100)
    
    # Add annotations
    for i, (x, y) in enumerate(points):
        ax.annotate(f'x_{iteration-2+i}', (x, y), 
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add step details
    error_str = f"{step_details['Error']:.2e}" if step_details['Error'] is not None else "N/A"
    textstr = '\n'.join([
        f"Iteration {iteration}:",
        f"x_{iteration} = {step_details['x_n']:.6f}",
        f"f(x_{iteration}) = {step_details['f(x_n)']:.6f}",
        f"Error = {error_str}"
    ])
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', bbox=props)
    
    # Customize plot
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.set_title(f"Secant Method - Iteration {iteration}", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=10)
    
    return fig

# Streamlit App
def main():
    st.title("📊 Secant Method Solver")
    st.markdown("""
    This application solves for roots of functions using the Secant Method, with detailed step-by-step visualization
    and AI-powered validation. The method stops when the absolute error is less than 0.5.
    
    **Mathematical Constants:**
    - `e` or `E`: The natural logarithm base (≈ 2.71828)
    """)
    
    # Initialize LLM
    llm = initialize_gemini_llm()
    
    # Sidebar for Inputs
    with st.sidebar:
        st.header("Input Parameters")
        expr = st.text_input("Function f(x)", value="x**2 - 4", 
                            help="Enter a function using Python syntax (e.g., x**2 - 4, cos(x) - x, e**x - 2)")
        
        col1, col2 = st.columns(2)
        with col1:
            x0 = st.number_input("x₀ (First Guess)", value=1.0, step=0.1)
        with col2:
            x1 = st.number_input("x₁ (Second Guess)", value=2.0, step=0.1)
        
        st.subheader("Advanced Settings")
        tol = st.number_input("Tolerance", value=0.5, step=0.1, 
                             help="Convergence criterion (absolute error). Default is 0.5")
        max_iter = st.number_input("Maximum Iterations", value=100, 
                                  help="Maximum number of iterations allowed")
        
        solve_button = st.button("Solve", type="primary")
    
    # Main Content
    if solve_button:
        if not expr.strip():
            st.error("Please enter a valid function expression.")
        else:
            # Validate expression
            with st.spinner("Validating expression..."):
                validated_expr = validate_expression(expr, llm)
                if validated_expr.startswith("Invalid"):
                    st.warning(validated_expr)
                    expr = validated_expr.split(":")[1].split("(Reason")[0].strip()
                else:
                    expr = validated_expr.split(":")[1].strip()
            
            # Run Secant Method
            with st.spinner("Calculating..."):
                root, x_values, errors, message, steps = secant_method(expr, x0, x1, tol, max_iter)
            
            # Display Results
            if root is not None:
                st.success(f"Root found: {root:.10f}")
                st.info(message)
                
                # Display iteration history
                st.subheader("Iteration History")
                df = pd.DataFrame(steps)
                st.dataframe(df.style.format({
                    "x_n": "{:.10f}",
                    "f(x_n)": "{:.10f}",
                    "Error": "{:.2e}"
                }))
                
                # Visualizations
                st.subheader("Visualizations")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Convergence Plot")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.semilogy(range(len(errors)), errors, 'o-', label='Absolute Error')
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Error (log scale)")
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend()
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("### Function Plot")
                    x_range = np.linspace(min(x_values) - 1, max(x_values) + 1, 400)
                    y_range = [evaluate_function(expr, x) for x in x_range]
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(x_range, y_range, label=f"f(x) = {expr}")
                    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
                    ax.axvline(root, color='r', linestyle='--', label=f"Root ≈ {root:.4f}")
                    ax.set_xlabel("x")
                    ax.set_ylabel("f(x)")
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend()
                    st.pyplot(fig)
                
                # Step-by-step visualization
                st.subheader("Step-by-Step Visualization")
                for step in steps:
                    with st.expander(f"Iteration {step['Iteration']}"):
                        x_n = step["x_n"]
                        x_n_minus_1 = steps[step["Iteration"]-1]["x_n"] if step["Iteration"] > 1 else x0
                        x_n_minus_2 = steps[step["Iteration"]-2]["x_n"] if step["Iteration"] > 2 else x0
                        
                        fig = plot_step(expr, x_n_minus_2, x_n_minus_1, x_n, 
                                      step["Iteration"], x_range, step, 8, 4)
                        st.pyplot(fig)
                        st.markdown(f"**Explanation:** {step['Explanation']}")
            else:
                st.error(message)
    
    # Examples and Help
    with st.expander("Examples & Help"):
        st.markdown("""
        ### Example Functions
        - **f(x) = x² - 4**: Roots at ±2
        - **f(x) = cos(x) - x**: Root ≈ 0.739
        - **f(x) = x³ - x - 2**: Root ≈ 1.521
        - **f(x) = e^x - 2**: Root ≈ 0.693
        
        ### Tips
        - Use Python syntax for mathematical expressions
        - Choose initial guesses close to the expected root
        - The method may not converge if the function is not well-behaved
        - Adjust tolerance and maximum iterations as needed
        - If a near-zero denominator is encountered, the method will attempt to perturb the guess
        """)

if __name__ == "__main__":
    main()