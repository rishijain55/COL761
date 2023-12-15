# Files Description

## Task-1 Files
1. **Folders:** 
   - task1/ - Contains source code for subgraph mining methods.
   - task2/ - Contains source code for clustering.

2. **Subgraph Mining:**
   - Folders: FSG, gSpan, Gaston - Source codes for subgraph mining methods.
   - Input_conversion_<Method Name>.cpp - Converts the dataset 167.txt_graph into the corresponding format.
   - 167.txt_graph - Yiest Dataset.
   - plot.py - Python script to create the curve between runtime vs support for all methods.
   - plot_gen.sh - Shell script to extract the runtime from the outputs of each method to plot the curve.
   - run.sh - PBS script to submit the job on HPC to run all methods and save the output.

3. **Task-2 Files:**
   - Clustering.py - Python script for k-means clustering and to generate the elbow plot.
   - elbow_plot.sh - Shell script to generate the elbow plot.
   - generateDataset_d_dim_hpc_compiled - Dataset generator.
   - CS1200335_generated_dataset_*D.dat - Datasets generated from generateDataset_d_dim_hpc_compiled using HPC.

## Instructions for Code Execution

### Task-1
1. Upload the gcc module on HPC using the command: `module load compiler/gcc/9.1.0`.
2. Run `run.sh` to execute all three methods: `sh run.sh`.
3. To run on another dataset, add it to the directory and update `input_file` in `run.sh`.
4. After completion, a `plot_point.txt` file will be generated.
5. To plot the runtime vs support curve, run `plot.py` locally: `python plot.py`.
6. To plot on HPC, set up the python environment and install the `Matplotlib` module.

### Task-2
1. Generate the elbow plot locally using `elbow_plot.sh` and `Clustering.py`.
2. Ensure python libraries: scikit-learn, Matplotlib, Numpy are installed.
3. To generate the elbow plot on HPC, set up the python environment and install the required modules.
4. Run `elbow_plot.sh` with dataset and dimension: `sh elbow_plot.sh <dataset> <dimension> q3_<dimension>_<RollNo>.png`.