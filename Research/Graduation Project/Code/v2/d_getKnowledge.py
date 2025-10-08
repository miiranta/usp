import os
import pandas as pd
import d_getKnowledge_aux as aux

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.abspath(os.path.join(SCRIPT_FOLDER, 'appended_csvs'))
OUTPUT_FOLDER = os.path.abspath(os.path.join(SCRIPT_FOLDER, 'results_knowledge'))

if not os.path.exists(INPUT_FOLDER):
    print("Input folder does not exist.")
    exit(1)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

FILTER_DEFAULT_REPLACE = 30  # Replace filter "0 sigma" with this value

# Load appended_benchmarks.csv
appended_benchmarks_path = os.path.join(INPUT_FOLDER, 'appended_benchmarks.csv')
appended_benchmarks_df = pd.read_csv(appended_benchmarks_path)

# Split rows in types at "-" to a list
appended_benchmarks_df['types'] = appended_benchmarks_df['types'].apply(lambda x: x.split('-') if pd.notnull(x) else [])

# Parse filter column to an int "0 sigma" to 0
def parse_filter(value):
    if pd.isnull(value):
        return None
    try:
        return int(value.split()[0])
    except (ValueError, IndexError):
        return None
appended_benchmarks_df['filter'] = appended_benchmarks_df['filter'].apply(parse_filter)

# Change filter 0
appended_benchmarks_df['filter'] = appended_benchmarks_df['filter'].replace(0, FILTER_DEFAULT_REPLACE)

# File structure
# model,types,filter,count,min,max,mean,std,bin_count,shannon_entropy,desequilibrium,complexity,BENCH-OPEN_LLM_AVERAGE,BENCH-LMARENA_SCORE,BENCH-MMLU_5,BENCH-MMLU_PRO_5

def main():
    
    # (1) COMPLEXITY vs FILTER
    # 1.1 Para cada valor de filter, calcular a complexidade média e máxima (desvio padrao?)
    # 1.2 Regressao livre entre filter e complexidade, Regressao livre na complexidade maxima
    if False:
        aux.plot_filter_vs_complexity(appended_benchmarks_df, OUTPUT_FOLDER)
    
    # (2) COMPLEXITY vs TYPES
    # 2.1 Para cada tipo, calcular a complexidade média e máxima
    if False:
        aux.plot_complexity_vs_types(appended_benchmarks_df, OUTPUT_FOLDER)
    
    # (3) COMPLEXITY vs NUMBER OF PARAMS
    # 3.1 Para cada número de parâmetros, calcular a complexidade média, com histograma
    # 3.2 Regressao livre entre número de parâmetros e complexidade
    if False:
        aux.plot_complexity_vs_num_params(appended_benchmarks_df, OUTPUT_FOLDER)
    
    # (4) COMPLEXITY vs NUMBER OF BINS
    # 4.1 Para cada número de bins, calcular a complexidade média, com histograma
    # 4.2 Regressao livre entre número de bins e complexidade
    if False:
        aux.plot_complexity_vs_num_bins(appended_benchmarks_df, OUTPUT_FOLDER)
    
    # ===========
    
    # (5) FOR EACH BENCHMARK, FILTER and TYPE - COMPLEXITY vs BENCHMARK
    # 5.1 Para cada benchmark, regressao livre e linear entre complexidade e benchmark
    # 5.2 Plot das correlações lineares
    # 5.3 Plot das correlações livres
    # 5.4 Plot dos R2 das regressões lineares
    # 5.5 Plot dos R2 das regressões livres
    if False:
        aux.analyze_benchmarks_vs_complexity(appended_benchmarks_df, OUTPUT_FOLDER)
    
    # ===========
    
    # (6) FOR EACH BENCHMARK, FILTER and TYPE - PARAM COUNT vs BENCHMARK
    # 6.1 Para cada benchmark, regressao livre e linear entre número de parâmetros e benchmark
    # 6.2 Plot das correlações lineares
    # 6.3 Plot das correlações livres
    # 6.4 Plot dos R2 das regressões lineares
    # 6.5 Plot dos R2 das regressões livres
    if False:
        aux.analyze_param_count_vs_benchmarks(appended_benchmarks_df, OUTPUT_FOLDER)
    
    # ===========
    
    # (7) MAGICAL VARIABLE EXPLORE
    if False:
        aux.equation_exploration(appended_benchmarks_df, OUTPUT_FOLDER)
        
    if True:
        aux.analyze_equation_exploration_results(OUTPUT_FOLDER)
    
    # (8) FOR EACH BENCHMARK - MAGICAL VARIABLE vs BENCHMARK
    if False:
        aux.analyze_magical_var_vs_benchmarks(appended_benchmarks_df, OUTPUT_FOLDER)
    
if __name__ == "__main__":
    main()