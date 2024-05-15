MAX_JOBS=${1:-10}
CONFIG=${2:example_link_ecoli}
(
  trap 'kill 0' SIGINT
  CUR_JOBS=0
  for ((i=1; i<=$MAX_JOBS; i++)); do
    ((CUR_JOBS >= MAX_JOBS)) && wait -n

    # Run your Python script here
    python main_optuna.py --cfg ./configs/pyg/${CONFIG}.yaml &
    sleep 1  # Adjust sleep duration if needed

    ((++CUR_JOBS))
  done

  wait
)