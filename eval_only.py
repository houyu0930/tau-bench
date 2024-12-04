import json
import argparse
from math import comb
from pydantic import BaseModel
from tau_bench.envs import get_env
from tau_bench.envs.airline.tasks_test import TASKS as AIRLINE_TASKS
from tau_bench.envs.retail.tasks_test import TASKS_TEST as RETAIL_TASKS
from tau_bench.types import Task, Action, EnvRunResult, RESPOND_ACTION_NAME
from typing import List, Dict, Any
from tqdm import tqdm


class OriginalResult(BaseModel):
    task_id: int
    user_instruction: str
    traj: List[Dict[str, Any]]
    ground_truth_actions: List[Action]
    ground_truth_outputs: List[str]


def message_to_action(
    message: Dict[str, Any],
) -> Action:
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})


def reproduce_run(task_index, env, messages):
    isolated_env = get_env(
        env,
        user_strategy='llm',
        user_model='gpt-4o',
        task_split='test',
        user_provider='openai',
        task_index=task_index,
    )

    # print(f"Running task {task_index}")
    reward = 0.0
    mid = 0
    while mid < len(messages):
        if mid > 1:
            next_message = messages[mid]
            action = message_to_action(next_message)
            env_response = isolated_env.step(action, mid, messages)
            reward = env_response.reward
            if action.name != RESPOND_ACTION_NAME:
                mid += 1
            else:
                mid += 1
        mid += 1

    return reward
    

def display_metrics(results: List[EnvRunResult]) -> None:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r.trial for r in results]))
    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(
                result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(
                result.reward) else 0
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, choices=[
                        "airline", "retail"], help="The environment that the original trajectories are from (used to fetch the user instructions)")
    parser.add_argument("--results-path", type=str,
                        help="Path to the results file")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    with open(args.results_path, "r") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results")
    env = args.env
    if env == "airline":
        tasks: List[Task] = AIRLINE_TASKS
    elif env == "retail":
        tasks: List[Task] = RETAIL_TASKS
    else:
        raise ValueError(f"Invalid environment: {env}")

    env_run_results = []
    for result in tqdm(results):
        task_id: int = result["task_id"]
        task = tasks[task_id]
        user_instruction = task.instruction
        ground_truth_actions = task.actions
        ground_truth_outputs = task.outputs
        
        # original_result = OriginalResult(task_id=task_id,
        #                                  user_instruction=user_instruction,
        #                                  traj=result["traj"],
        #                                  ground_truth_actions=ground_truth_actions,
        #                                  ground_truth_outputs=ground_truth_outputs)
        
        new_reward = reproduce_run(task_id, env, result["traj"])
        assert new_reward == result["reward"], f"Rewards do not match: {new_reward} vs {result['reward']}"

        env_run_result = EnvRunResult(task_id=task_id,
                                    #   reward=result["reward"],
                                      reward=new_reward,
                                      info=result["info"],
                                      traj=result["traj"],
                                      trial=result["trial"])
        env_run_results.append(env_run_result)

        # if env_run_result.reward == 1.0 and len(ground_truth_outputs) > 0:
        #     print(f"Task {task_id} succeeded")

    display_metrics(env_run_results)


if __name__ == "__main__":
    main()
