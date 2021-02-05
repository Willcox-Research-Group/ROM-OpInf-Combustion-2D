#!/bin/bash
# History of commands used for several figures in the paper
# "Data-driven reduced-order models via regularised Operator Inference for a single-injector combustion process"
# by Shane A. McQuarrie, Cheng Huang, and Karen E. Willcox.
# RECORDING PURPOSES ONLY; DO NOT RUN THIS FILE DIRECTLY.
exit 1

## INSTALLATION / SETUP
# Download GEMS .tar data files with globus to /storage1/combustion_gems_2d/rawdata (see wiki for instructions).
# Set config.BASE_FOLDER = "/storage1/combustion_gems_2d"
# Install Python dependencies
python -m pip3 install --user -r requirements.txt

## STEP 1 ---------------------------------------------------------------------
# Unpack the raw GEMS .tar data files (once).
python step1_unpack.py /storage1/combustion_gems_2d/rawdata

# Remove the raw GEMS .tar data files if desired (data now saved as gems.h5).
# rm /storage1/combustion_gems_2d/rawdata/Data_*to*.tar

## STEP 2 ---------------------------------------------------------------------
# Preprocess the first k snapshots for ROM learning.
python step2_preprocess.py 10000  50                            # k=10000, r_max=050
python step2_preprocess.py 20000  75                            # k=20000, r_max=075
python step2_preprocess.py 30000 100                            # k=30000, r_max=100

# Get all singular values for Figure 2.
python step2b_basis.py 10000 -1
python step2b_basis.py 20000 -1
python step2b_basis.py 30000 -1

# Remove the scaled data if desired (data has already been projected).
# rm /storage1/combustion_gems_2d/k*/data_scaled.h5

## STEP 3 ---------------------------------------------------------------------
# Train the ROMs with dimension r and regularization hyperparameters λ1, λ2.
python step3_train.py --single 10000 22  91 32251               # k=10000, r=22, λ1=091, λ2=32251
python step3_train.py --single 20000 43 316 18199               # k=20000, r=43, λ1=316, λ2=18199
python step3_train.py --single 30000 66 105 27906               # k=30000, r=66, λ1=105, λ2=27906

## STEP 4 ---------------------------------------------------------------------
# Generate plots like Figure 3.
python step4_plot.py --point-traces 10000 22  91 32251          # k=10000, r=22, λ1=091, λ2=32251
python step4_plot.py --point-traces 20000 43 316 18199          # k=20000, r=43, λ1=316, λ2=18199
python step4_plot.py --point-traces 30000 66 105 27906          # k=30000, r=66, λ1=105, λ2=27906

# Generate plots like Figure 4 (left).
python step4_plot.py --relative-errors 10000 22  91 32251       # k=10000, r=22, λ1=091, λ2=32251
python step4_plot.py --relative-errors 20000 43 316 18199       # k=20000, r=43, λ1=316, λ2=18199
python step4_plot.py --relative-errors 30000 66 105 27906       # k=30000, r=66, λ1=105, λ2=27906

# Generate plots like Figure 5.
python step4_plot.py --spatial-statistics 10000 22  91 32251    # k=10000, r=22, λ1=091, λ2=32251
python step4_plot.py --spatial-statistics 20000 43 316 18199    # k=20000, r=43, λ1=316, λ2=18199
python step4_plot.py --spatial-statistics 30000 66 105 27906    # k=30000, r=66, λ1=105, λ2=27906

# Reproduce several of the actual plots in the paper.
# NOTE: this script requires pod_deim.h5, which is not included on globus.
python plots.py

## STEP 5 ---------------------------------------------------------------------
# Export data to tecplot-readable format for creating figures 1, 4b, 7, and 8.
python step5_export.py fom rom --timeindex 5000 10000 15000 20000 25000 --variables T CH4 --trainsize 20000 --modes 43 --regularization 316 18199
python poddeim.py --export --timeindex 5000 10000 15000 20000 25000 --variables T CH4


# Notes =======================================================================

'''
       |               ROM dimension r needed to exceed cumulative_energy(r) and corresponding data matrix column dimension d(r)
   k   |     .985      |      .990     |      .995      |      .9975      |       .999      |      .9999      |   .99999        |     .999999 
-------|---------------|---------------|----------------|-----------------|-----------------|-----------------|------------- ---|----------------
10,000 | r=22, d=  277 | r=27, d=  407 | r= 36, d=  704 | r= 46, d= 1,129 | r= 61, d= 1,954 | r=108, d= 5,996 | r=172 d= 15,052 | r=268 d= 36,316
20,000 | r=43, d=  991 | r=53, d=1,486 | r= 72, d=2,701 | r= 92, d= 4,372 | r=121, d= 7,504 | r=214, d=23,221 | r=342 d= 58,997 | r=524 d=138,076
30,000 | r=66, d=2,279 | r=82, d=3,487 | r=110, d=6,217 | r=141, d=10,154 | r=186, d=17,579 | r=326, d=53,629 | r=521 d=136,504 | r=804 d=324,416
'''
