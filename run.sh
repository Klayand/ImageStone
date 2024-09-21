CUDA_VISIBLE_DEVICES=0 python benchmark_scripts/chronomagic_bench_150.py --pipeline=VIVI &
CUDA_VISIBLE_DEVICES=1 python benchmark_scripts/chronomagic_bench_150.py --pipeline=IVIV &
CUDA_VISIBLE_DEVICES=2 python benchmark_scripts/chronomagic_bench_150.py --pipeline=VVII &
CUDA_VISIBLE_DEVICES=3 python benchmark_scripts/chronomagic_bench_150.py --pipeline=IIVV &
CUDA_VISIBLE_DEVICES=4 python benchmark_scripts/chronomagic_bench_150.py --pipeline=IVVI &
CUDA_VISIBLE_DEVICES=5 python benchmark_scripts/chronomagic_bench_150.py --pipeline=VIIV 
