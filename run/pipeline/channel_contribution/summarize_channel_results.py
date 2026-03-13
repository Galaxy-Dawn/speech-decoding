import collections
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import fire


initial_f1_chance_level_dict={
    "S1"   :0.1067,
    "S2"   :0.1063,
    "S3"   :0.1062,
    "S4"   :0.1056,
    "S5"   :0.1059,
    "S6"   :0.1067,
    "S7"   :0.1074,
    "S8"   :0.1066,
    "S9"   :0.1067,
    "S10"  :0.1053,
    "S11"  :0.0916,
    "S12"  :0.0908,
}

final_f1_chance_level_dict={
    "S1"  :0.0906,
    "S2"  :0.0889,
    "S3"  :0.0889,
    "S4"  :0.0907,
    "S5"  :0.0891,
    "S6"  :0.0911,
    "S7"  :0.0896,
    "S8"  :0.0891,
    "S9"  :0.0886,
    "S10" :0.0859,
    "S11" :0.0665,
    "S12" :0.0669,
}

tone_f1_chance_level_dict={
    "S1"  :0.259,
    "S2"  :0.2585,
    "S3"  :0.259,
    "S4"  :0.2591,
    "S5"  :0.2587,
    "S6"  :0.2586,
    "S7"  :0.2597,
    "S8"  :0.2585,
    "S9"  :0.2583,
    "S10" :0.2572,
    "S11" :0.2563,
    "S12" :0.2543,
}

def calculate_improvement(row, subject_id):
    task = row['Task']
    # Calculate improvement percentage using average value (core logic unchanged)
    f1_score = row['Avg F1_Score']
    if task == "speaking_initial" or task == "listening_initial":
        chance_level = initial_f1_chance_level_dict.get(subject_id, initial_f1_chance_level_dict["S1"])
    elif task == "speaking_final" or task == "listening_final":
        chance_level = final_f1_chance_level_dict.get(subject_id, final_f1_chance_level_dict["S1"])
    elif task == "speaking_tone" or task == "listening_tone":
        chance_level = tone_f1_chance_level_dict.get(subject_id, tone_f1_chance_level_dict["S1"])
    else:
        return 0
    if chance_level > 0:
        return round((f1_score - chance_level) / chance_level * 100, 4)
    else:
        return 0


def main(subject_ids: str):
    ckpt_dir = "" #cfg.model_save_dir
    results_base_dir = "" #cfg.data_dir

    if isinstance(subject_ids, str):
        subject_ids = [sid.strip() for sid in subject_ids.replace(',', ' ').split() if sid.strip()]
    elif isinstance(subject_ids, (tuple, list)):
        subject_ids = [sid.strip() for sid in list(subject_ids) if sid.strip()]
    else:
        raise ValueError("subject_ids must be a string, tuple, or list")

    tasks = ["speaking_initial", "speaking_final", "speaking_tone", "listening_initial", "listening_final",
             "listening_tone"]

    repeat_cols = [f"Seed {i} F1_Score" for i in range(42, 52)]
    columns = ["Subject", "Task", "Channel"] + repeat_cols + ["Avg F1_Score", "F1_Improvement_Percentage"]

    # Iterate through each subject ID
    for subject_id in subject_ids:
        print(f"Processing test results for subject {subject_id}...")
        channel_repeat_data = collections.defaultdict(dict)
        channel_results = collections.defaultdict(list)
        target_prefix = subject_id + "_channel_contribution"
        for root, dirs, files in os.walk(ckpt_dir):
            dirs[:] = [d for d in dirs if d.startswith(target_prefix)]  # Filter target directories
            for dir_name in dirs:
                # Skip directories that don't contain the target task
                if not any(task in dir_name for task in tasks):
                    continue

                # Match the task corresponding to the current directory
                current_task = None
                for task in tasks:
                    if task in dir_name:
                        current_task = task
                        break
                if not current_task:
                    continue

                try:
                    if "inference" in dir_name:
                        task_start = dir_name.find(current_task) + len(current_task) + 1
                        channel_repeat_str = dir_name[task_start:]
                        channel = int(channel_repeat_str.split("_")[0])
                        seed_num = int(channel_repeat_str.split("_")[2])
                    else:
                        continue
                    # Read F1 score
                    task_folder_path = os.path.join(root, dir_name)
                    eval_results_file = os.path.join(task_folder_path, "all_results.json")
                    if not os.path.isfile(eval_results_file):
                        print(f"Warning: all_results.json file not found for {dir_name}!")
                        continue

                    with open(eval_results_file, 'r', encoding='utf-8') as f:
                        eval_results = json.load(f)
                        f1_score = round(eval_results.get("eval_f1", 0), 4)

                    # ========== Key modification: Keep the maximum F1 score under the same repeat_num ==========
                    key = (current_task, channel)
                    channel_repeat_data[key][seed_num] = f1_score
                    print(f"Read {dir_name} -> Channel{channel} | Seed{seed_num} | F1={f1_score}")

                except Exception as e:
                    print(f"Error processing directory {dir_name}: {str(e)}")
                    continue

        # Build final data rows
        subject_data = []
        for (task, channel), repeat_scores in channel_repeat_data.items():
            # Initialize scores for 10 repeats (fill 0 if missing)
            repeat_f1 = [repeat_scores.get(i, 0.0) for i in range(10)]
            # Calculate average (only count non-zero values)
            valid_scores = [s for s in repeat_f1 if s > 0]
            avg_f1 = round(np.mean(valid_scores), 4) if valid_scores else 0.0
            # Calculate improvement percentage
            improvement = calculate_improvement(
                {"Task": task, "Avg F1_Score": avg_f1},
                subject_id
            )
            # Assemble a row of data
            row = [subject_id, task, channel] + repeat_f1 + [avg_f1, improvement]
            subject_data.append(row)

        # Generate DataFrame and sort
        if not subject_data:
            print(f"Warning: No valid data found for subject {subject_id}!")
            continue

        df = pd.DataFrame(subject_data, columns=columns)
        # Sort by improvement percentage in descending order (maintain original logic)
        df = df.groupby('Task', group_keys=False).apply(
            lambda group: group.sort_values(by="Channel", ascending=True)
        ).reset_index(drop=True)

        # Save Excel file
        output_detail_file = Path(results_base_dir) / subject_id / f"{subject_id}_channel_contribution_detail.xlsx"
        output_detail_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_detail_file, index=False)
        print(f"Filtering details for subject {subject_id} saved to {output_detail_file}")

        print('Filtering channels...')
        grouped_df = df.groupby("Task")
        for task, group in grouped_df:
            if task == "speaking_tone" or task == "listening_tone":
                filtered_group = group[group["F1_Improvement_Percentage"] >= 100]
                if filtered_group.empty:
                    filtered_group = group[group["F1_Improvement_Percentage"] >= 50]
                    if filtered_group.empty:
                        filtered_group = group[group["F1_Improvement_Percentage"] >= 0]
                for channel in filtered_group["Channel"]:
                    channel_results[task].append(int(channel))
                channel_results[task] = sorted(channel_results[task])
            else:
                filtered_group = group[group["F1_Improvement_Percentage"] >= 100]
                for channel in filtered_group["Channel"]:
                    channel_results[task].append(int(channel))
                channel_results[task] = sorted(channel_results[task])

        speaking_initial_output_results_file = Path(results_base_dir) / subject_id / "speaking_initial_channel_selection_results.txt"
        speaking_final_output_results_file = Path(results_base_dir) / subject_id / "speaking_final_channel_selection_results.txt"
        speaking_tone_output_results_file = Path(results_base_dir) / subject_id / "speaking_tone_channel_selection_results.txt"
        listening_initial_output_results_file = Path(results_base_dir) / subject_id / "listening_initial_channel_selection_results.txt"
        listening_final_output_results_file = Path(results_base_dir) / subject_id / "listening_final_channel_selection_results.txt"
        listening_tone_output_results_file = Path(results_base_dir) / subject_id / "listening_tone_channel_selection_results.txt"

        with open(speaking_initial_output_results_file, 'w', encoding='utf-8') as f:
            f.write(f"speaking_initial: {channel_results['speaking_initial']}\n") if "speaking_initial" in channel_results else None
        with open(speaking_final_output_results_file, 'w', encoding='utf-8') as f:
            f.write(f"speaking_final: {channel_results['speaking_final']}\n") if "speaking_final" in channel_results else None
        with open(speaking_tone_output_results_file, 'w', encoding='utf-8') as f:
            f.write(f"speaking_tone: {channel_results['speaking_tone']}\n") if "speaking_tone" in channel_results else None

        with open(listening_initial_output_results_file, 'w', encoding='utf-8') as f:
            f.write(f"listening_initial: {channel_results['listening_initial']}\n") if "listening_initial" in channel_results else None
        with open(listening_final_output_results_file, 'w', encoding='utf-8') as f:
            f.write(f"listening_final: {channel_results['listening_final']}\n") if "listening_final" in channel_results else None
        with open(listening_tone_output_results_file, 'w', encoding='utf-8') as f:
            f.write(f"listening_tone: {channel_results['listening_tone']}\n") if "listening_tone" in channel_results else None

        print(f"Speaking initial channel selection results for subject {subject_id} saved to {speaking_initial_output_results_file}")
        print(f"Speaking final channel selection results for subject {subject_id} saved to {speaking_final_output_results_file}")
        print(f"Speaking tone channel selection results for subject {subject_id} saved to {speaking_tone_output_results_file}")

        print(f"Listening initial channel selection results for subject {subject_id} saved to {listening_initial_output_results_file}")
        print(f"Listening final channel selection results for subject {subject_id} saved to {listening_final_output_results_file}")
        print(f"Listening tone channel selection results for subject {subject_id} saved to {listening_tone_output_results_file}")


if __name__ == "__main__":
    fire.Fire(main)