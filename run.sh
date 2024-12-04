# python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --user-model gpt-4o --user-model-provider openai --user-strategy llm --max-concurrency 10

# python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --user-model gpt-4o --user-model-provider openai --user-strategy llm --max-concurrency 10 --task-ids 2

# python auto_error_identification.py --env <airline/retail> --results-path <the path to your results file here> --max-concurrency 16 --output-path test-auto-error-identification -n 10

python eval_only.py --env airline --results-path ./historical_trajectories/gpt-4o-airline.json
# Loaded 200 results

# python eval_only.py --env retail --results-path ./historical_trajectories/gpt-4o-retail.json
# Loaded 460 results

# python eval_only.py --env airline --results-path ./historical_trajectories/sonnet-35-new-airline.json
# Loaded 400 results

# python eval_only.py --env retail --results-path ./historical_trajectories/sonnet-35-new-retail.json
# Loaded 920 results