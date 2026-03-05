import matplotlib.pyplot as plt
import pandas as pd
import re
import sys
import os

def parse_benchmark_output(filepath):
    """
    Parses the text output of the benchmark tool into structured DataFrames.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = {}
    current_section = None
    section_lines = []

    # Regex patterns for different table headers
    patterns = {
        'PROPOSITION_5': r'PROPOSITION 5: Box Polytope',
        'LEMMA_2': r'LEMMA 2: Farthest Good Vertex',
        'PROPOSITION_1': r'PROPOSITION 1: Exit Face Alignment',
        'THEOREM_1_A': r'\(a\) Phase II hops <= adjacency bound',
        'THEOREM_1_B': r'\(b\) L sweep: hops vs separation',
        'THEOREM_1_C': r'\(c\) General polytopes -- Phase II hop count',
        'COROLLARY_1': r'COROLLARY 1: Well-Conditioned Polytopes',
        'ASSUMPTION_1': r'ASSUMPTION 1: Good Vertex Existence',
        'KKT_VERIFICATION': r'KKT / SOLUTION QUALITY VERIFICATION',
        'PERFORMANCE_BOX': r'>>> Box QP\s*$',  # Strict end of line to avoid matching timing sections
        'PERFORMANCE_POLY': r'>>> Polytope QP \(m=2n\)',
        'PERFORMANCE_MPC': r'>>> MPC QP\s*$',
        'PERFORMANCE_DMPC': r'>>> Diagonal MPC QP',
        'TIMING_BOX_DIAG': r'>>> Box QP \(diagonal Q\)',
        'TIMING_BOX_DENSE': r'>>> Box QP \(dense Q\)',
        'TIMING_POLY': r'>>> General Polytope QP',
        'TIMING_MPC_DENSE': r'>>> MPC QP \(dense Q\)',
        'TIMING_MPC_DIAG': r'>>> MPC QP \(diagonal Q\)',
        'HOTSTART': r'HOTSTART BENCHMARK',
        'CONDITIONING': r'CONDITIONING STRESS TEST'
    }
    
    # Helper to parse a block of lines into a DataFrame
    def parse_block(lines, columns):
        parsed_data = []
        for line in lines:
            line = line.strip()
            # Skip separator lines, empty lines, and header lines (which start with 'n ' or 'scale_g')
            if not line or line.startswith('---') or line.startswith('===') or line.startswith('n ') or line.startswith('scale_g') or 'overall:' in line:
                continue
            
            # Handle cases where columns are merged or spacing is irregular
            parts = line.split()
            
            # Clean up percentage signs or 'x' suffixes
            clean_parts = []
            for p in parts:
                p = p.replace('%', '').replace('x', '')
                clean_parts.append(p)
            
            # Additional cleanup for 'Result' column which might have pass/fail
            if len(clean_parts) >= len(columns):
                 parsed_data.append(clean_parts[:len(columns)])
            
        return pd.DataFrame(parsed_data, columns=columns)

    # State machine to grab blocks
    capture = False
    headers = []
    
    for line in lines:
        # Check if new section starts
        found_new = False
        for key, pat in patterns.items():
            if re.search(pat, line):
                # Process previous section if any
                if current_section and section_lines:
                    data[current_section] = parse_block(section_lines, headers)
                
                current_section = key
                section_lines = []
                capture = True
                found_new = True
                
                # Set headers based on section
                if key == 'PROPOSITION_5': headers = ['n', 'Q type', 'Verts', 'v*@bound', 'ObjDiff', 'Ph2hops', 'Time(ms)', 'FastPath', 'CoordOK', 'Solved']
                elif key == 'PERFORMANCE_BOX': headers = ['n', 'Type', 'Geo(ms)', 'OSQP(ms)', 'qpOAS(ms)', 'Geo/OSQP', 'Geo/qpO', 'GeoHops', 'ObjDiff', 'Result']
                elif key == 'PERFORMANCE_POLY': headers = ['n', 'Type', 'Geo(ms)', 'OSQP(ms)', 'qpOAS(ms)', 'Geo/OSQP', 'Geo/qpO', 'GeoHops', 'ObjDiff', 'Result']
                elif key == 'TIMING_BOX_DIAG': headers = ['n', 'Type', 'Total(us)', 'Setup(us)', 'Ph1(us)', 'Ph2(us)', 'Ph3(us)', 'Setup', 'Ph1', 'Ph2', 'Ph3', 'Hops', 'us/hop']
                elif key == 'TIMING_POLY': headers = ['n', 'Type', 'Total(us)', 'Setup(us)', 'Ph1(us)', 'Ph2(us)', 'Ph3(us)', 'Setup', 'Ph1', 'Ph2', 'Ph3', 'Hops', 'us/hop']
                # ... add other headers as needed for plotting
                break
        
        if found_new: continue

        if capture:
            if line.strip().startswith('==='): # End of section marker sometimes
               capture = False
               if current_section and section_lines:
                    data[current_section] = parse_block(section_lines, headers)
                    current_section = None
                    section_lines = []
            else:
               section_lines.append(line)
               
    # Catch last one
    if current_section and section_lines:
        data[current_section] = parse_block(section_lines, headers)

    return data

def plot_performance(data, output_dir):
    """Generates comparison plots for solver performance."""
    
    # --- Box QP Performance ---
    if 'PERFORMANCE_BOX' in data:
        df = data['PERFORMANCE_BOX']
        # Convert numeric columns only
        numeric_cols = ['n', 'Geo(ms)', 'OSQP(ms)', 'qpOAS(ms)']
        for col in numeric_cols:
             if col in df.columns:
                  df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=numeric_cols)
        df = df.sort_values(by='n')

        if not df.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(df['n'], df['Geo(ms)'], 'b-o', label='GeoQP')
            plt.plot(df['n'], df['OSQP(ms)'], 'r-s', label='OSQP')
            plt.plot(df['n'], df['qpOAS(ms)'], 'g-^', label='qpOASES')
            plt.yscale('log')
            plt.xlabel('Dimension n')
            plt.ylabel('Time (ms)')
            plt.title('Box QP Runtime Comparison (Log Scale)')
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'box_qp_performance.png'))
            plt.close()
            print(f"Generated box_qp_performance.png with {len(df)} data points.")
        else:
            print("PERFORMANCE_BOX data is empty after processing.")
    else:
        print("PERFORMANCE_BOX section not found.")

    # --- Polytope QP Performance ---
    if 'PERFORMANCE_POLY' in data:
        df = data['PERFORMANCE_POLY']
        
        numeric_cols = ['n', 'Geo(ms)', 'OSQP(ms)', 'qpOAS(ms)']
        for col in numeric_cols:
             if col in df.columns:
                  df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=numeric_cols)
        df = df.sort_values(by='n')

        if not df.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(df['n'], df['Geo(ms)'], 'b-o', label='GeoQP')
            plt.plot(df['n'], df['OSQP(ms)'], 'r-s', label='OSQP')
            plt.plot(df['n'], df['qpOAS(ms)'], 'g-^', label='qpOASES')
            plt.yscale('log')
            plt.xlabel('Dimension n (m=2n)')
            plt.ylabel('Time (ms)')
            plt.title('General Polytope QP Runtime Comparison (Log Scale)')
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'polytope_qp_performance.png'))
            plt.close()
            print(f"Generated polytope_qp_performance.png with {len(df)} data points.")
        else:
            print("PERFORMANCE_POLY data is empty after processing.")
    else:
        print("PERFORMANCE_POLY section not found.")

def plot_timing_breakdown(data, output_dir):
    """Generates stacked bar chart for timing breakdown."""
    
    # --- Box QP (Diag) Timing ---
    if 'TIMING_BOX_DIAG' in data:
        df = data['TIMING_BOX_DIAG']
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        n = df['n']
        setup = df['Setup(us)']
        ph1 = df['Ph1(us)']
        ph2 = df['Ph2(us)']
        ph3 = df['Ph3(us)']

        plt.figure(figsize=(10, 6))
        p1 = plt.bar(n, setup, width=5, label='Setup')
        p2 = plt.bar(n, ph1, bottom=setup, width=5, label='Phase I')
        p3 = plt.bar(n, ph2, bottom=setup+ph1, width=5, label='Phase II')
        p4 = plt.bar(n, ph3, bottom=setup+ph1+ph2, width=5, label='Phase III')

        plt.xlabel('Dimension n')
        plt.ylabel('Time (us)')
        plt.title('GeoQP Time Breakdown (Box QP, Diag Q)')
        plt.legend()
        plt.grid(True, axis='y', alpha=0.5)
        plt.savefig(os.path.join(output_dir, 'timing_breakdown_box.png'))
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_benchmarks.py <benchmark_output.txt>")
        sys.exit(1)
        
    log_file = sys.argv[1]
    output_dir = os.path.dirname(log_file)
    if not output_dir: output_dir = '.'
    
    print(f"Parsing {log_file}...")
    data = parse_benchmark_output(log_file)
    
    print("Generating plots...")
    plot_performance(data, output_dir)
    plot_timing_breakdown(data, output_dir)
    print(f"Plots saved to {output_dir}")
