# Code mostly from: https://github.com/EleutherAI/lm-evaluation-harness/pull/1339, credit to: https://github.com/ayulockin
import copy
import logging
import re
import os
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Literal, Tuple, Union

from packaging.version import Version

from  .utils  import simple_parse_args_string
import tenacity

logger = logging.getLogger(__name__)

try:
    import wandb

    assert Version(wandb.__version__) >= Version("0.13.6")
    if Version(wandb.__version__) < Version("0.13.6"):
        wandb.require("report-editing:v0")
except Exception as e:
    logger.warning("To use the wandb reporting functionality please install wandb>=0.13.6.\n" "To install the latest version of wandb run `pip install wandb --upgrade`\n" f"{e}")


def remove_none_pattern(input_string):
    # Define the pattern to match ',none' at the end of the string
    pattern = re.compile(r",none$")

    # Use sub() to replace ',none' with an empty string
    result = re.sub(pattern, "", input_string)

    # check if the input_string changed
    removed = result != input_string

    return result, removed


def _handle_non_serializable(o: Any) -> Union[int, str, list]:
    """Handle non-serializable objects by converting them to serializable types.

    Args:
        o (Any): The object to be handled.

    Returns:
        Union[int, str, list]: The converted object. If the object is of type np.int64 or np.int32,
            it will be converted to int. If the object is of type set, it will be converted
            to a list. Otherwise, it will be converted to str.
    """
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def get_wandb_printer() -> Literal["Printer"]:
    """Returns a wandb printer instance for pretty stdout."""
    from wandb.sdk.lib.printer import get_printer
    from wandb.sdk.wandb_settings import Settings

    printer = get_printer(Settings()._jupyter)
    return printer


# class WandbLogger:
class WandbLogger:
    def __init__(self, args):
        self.wandb_args = simple_parse_args_string(args.wandb_args)
        self.args = args
        self.all_args_dict = vars(args)
        self.printer = get_wandb_printer()
        try:
            self.init_run()
        except Exception as e:
            logger.warning(f"Failed to initialize W&B run: {e}")
            os.environ["WANDB_MODE"] = "offline"
            self.init_run()

    def finish(self):
        self.run.finish()

    @tenacity.retry(wait=tenacity.wait_fixed(5), stop=tenacity.stop_after_attempt(5))
    def init_run(self):
        if "name" not in self.wandb_args:
            if "config" in self.all_args_dict and self.all_args_dict["config"] != "":
                self.wandb_args["name"] = self.all_args_dict["config"].split("/")[-1].replace(".yaml", "") + "_" + self.args.log_samples_suffix
            else:
                task_names = self.args.tasks.replace(",", "/")
                self.wandb_args["name"] = f"{self.args.model}_{task_names}_{self.args.log_samples_suffix}"
                if self.args.num_fewshot:
                    self.wandb_args["name"] += f"_{self.args.num_fewshot}shot"
        if "project" not in self.wandb_args:
            self.wandb_args["project"] = "lmms-eval"
        # initialize a W&B run
        self.run = wandb.init(**self.wandb_args)

    def post_init(self, results: Dict[str, Any]) -> None:
        self.results: Dict[str, Any] = copy.deepcopy(results)
        self.task_names: List[str] = list(results.get("results", {}).keys())
        self.group_names: List[str] = list(results.get("groups", {}).keys())

    def _get_config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        self.task_configs = self.results.get("configs", {})
        cli_configs = self.results.get("config", {})
        configs = {
            "task_configs": self.task_configs,
            "cli_configs": cli_configs,
        }

        return configs

    def _sanitize_results_dict(self) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Sanitize the results dictionary."""
        _results = copy.deepcopy(self.results.get("results", dict()))

        # Remove None from the metric string name
        tmp_results = copy.deepcopy(_results)
        for task_name in self.task_names:
            task_result = tmp_results.get(task_name, dict())
            for metric_name, metric_value in task_result.items():
                _metric_name, removed = remove_none_pattern(metric_name)
                if removed:
                    _results[task_name][_metric_name] = metric_value
                    _results[task_name].pop(metric_name)

        # remove string valued keys from the results dict
        wandb_summary = {}
        for task in self.task_names:
            task_result = _results.get(task, dict())
            for metric_name, metric_value in task_result.items():
                if isinstance(metric_value, str):
                    wandb_summary[f"{task}/{metric_name}"] = metric_value

        for summary_metric, summary_value in wandb_summary.items():
            _task, _summary_metric = summary_metric.split("/")
            _results[_task].pop(_summary_metric)

        tmp_results = copy.deepcopy(_results)
        for task_name, task_results in tmp_results.items():
            for metric_name, metric_value in task_results.items():
                _results[f"{task_name}/{metric_name}"] = metric_value
                _results[task_name].pop(metric_name)
        for task in self.task_names:
            _results.pop(task)

        return wandb_summary, _results

    def _log_results_as_table(self) -> None:
        """Generate and log evaluation results as a table to W&B."""
        columns = [
            "Model",
            "Args",
            "Tasks",
            "Version",
            "Filter",
            "num_fewshot",
            "Metric",
            "Value",
            "Stderr",
        ]

        def make_table(columns: List[str], key: str = "results"):
            table = wandb.Table(columns=columns)
            results = copy.deepcopy(self.results)

            model_name = results.get("model_configs").get("model")
            model_args = results.get("model_configs").get("model_args")

            for k, dic in results.get(key).items():
                if k in self.group_names and not key == "groups":
                    continue
                version = results.get("versions").get(k)
                if version == "N/A":
                    version = None
                n = results.get("n-shot").get(k)

                for (mf), v in dic.items():
                    m, _, f = mf.partition(",")
                    if m.endswith("_stderr"):
                        continue
                    if m == "alias":
                        continue

                    if m + "_stderr" + "," + f in dic:
                        se = dic[m + "_stderr" + "," + f]
                        if se != "N/A":
                            se = "%.4f" % se
                        data = [model_name, model_args, k, version, f, n, m, str(v), str(se)]
                        if key == "groups":
                            data = [self.group_names] + data
                        table.add_data(*data)
                    else:
                        data = [model_name, model_args, k, version, f, n, m, str(v), ""]
                        if key == "groups":
                            data = [self.group_names] + data
                        table.add_data(*data)

            return table

        # log the complete eval result to W&B Table
        table = make_table(columns, "results")
        self.run.log({"evaluation/eval_results": table})

        if "groups" in self.results.keys():
            table = make_table(["Groups"] + columns, "groups")
            self.run.log({"evaluation/group_eval_results": table})

    def _log_results_as_artifact(self) -> None:
        """Log results as JSON artifact to W&B."""
        dumped = json.dumps(self.results, indent=2, default=_handle_non_serializable, ensure_ascii=False)
        artifact = wandb.Artifact("results", type="eval_results")
        with artifact.new_file("results.json", mode="w", encoding="utf-8") as f:
            f.write(dumped)
        self.run.log_artifact(artifact)

    def log_eval_result(self) -> None:
        """Log evaluation results to W&B."""
        # Log configs to wandb
        configs = self._get_config()
        self.run.config.update(configs, allow_val_change=True)

        wandb_summary, self.wandb_results = self._sanitize_results_dict()
        # update wandb.run.summary with items that were removed
        self.run.summary.update(wandb_summary)
        # Log the evaluation metrics to wandb
        self.run.log(self.wandb_results)
        # Log the evaluation metrics as W&B Table
        self._log_results_as_table()
        # Log the results dict as json to W&B Artifacts
        self._log_results_as_artifact()

    def _generate_dataset(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> pd.DataFrame:
        """Generate a dataset from evaluation data.

        Args:
            data (List[Dict[str, Any]]): The data to generate a dataset for.
            config (Dict[str, Any]): The configuration of the task.

        Returns:
            pd.DataFrame: A dataframe that is ready to be uploaded to W&B.
        """
        ids = [x["doc_id"] for x in data]
        labels = [x["target"] for x in data]
        instance = [""] * len(ids)
        resps = [""] * len(ids)
        filtered_resps = [""] * len(ids)
        model_outputs = {}

        metrics_list = config["metric_list"]
        metrics = {}
        for metric in metrics_list:
            metric = metric.get("metric")
            if metric in ["word_perplexity", "byte_perplexity", "bits_per_byte"]:
                metrics[f"{metric}_loglikelihood"] = [x[metric][0] for x in data]
                if metric in ["byte_perplexity", "bits_per_byte"]:
                    metrics[f"{metric}_bytes"] = [x[metric][1] for x in data]
                else:
                    metrics[f"{metric}_words"] = [x[metric][1] for x in data]
            else:
                metrics[metric] = [x[metric] for x in data]

        if config["output_type"] == "loglikelihood":
            instance = [x["arguments"][0][0] for x in data]
            labels = [x["arguments"][0][1] for x in data]
            resps = [f'log probability of continuation is {x["resps"][0][0][0]} ' + "\n\n" + "continuation will {} generated with greedy sampling".format("not be" if not x["resps"][0][0][1] else "be") for x in data]
            filtered_resps = [f'log probability of continuation is {x["filtered_resps"][0][0]} ' + "\n\n" + "continuation will {} generated with greedy sampling".format("not be" if not x["filtered_resps"][0][1] else "be") for x in data]
        elif config["output_type"] == "multiple_choice":
            instance = [x["arguments"][0][0] for x in data]
            choices = ["\n".join([f"{idx}. {y[1]}" for idx, y in enumerate(x["arguments"])]) for x in data]
            resps = [np.argmax([n[0][0] for n in x["resps"]]) for x in data]
            filtered_resps = [np.argmax([n[0] for n in x["filtered_resps"]]) for x in data]
        elif config["output_type"] == "generate_until":
            instance = [x["arguments"][0][0] for x in data]
            resps = [x["resps"][0][0] for x in data]
            filtered_resps = [x["filtered_resps"][0] for x in data]

        model_outputs["raw_predictions"] = resps
        model_outputs["filtered_predictions"] = filtered_resps

        df_data = {
            "id": ids,
            "data": instance,
        }
        if config["output_type"] == "multiple_choice":
            df_data["choices"] = choices

        tmp_data = {
            "input_len": [len(x) for x in instance],
            "labels": labels,
            "output_type": config["output_type"],
        }
        df_data.update(tmp_data)
        df_data.update(model_outputs)
        df_data.update(metrics)

        return pd.DataFrame(df_data)

    def _log_samples_as_artifact(self, data: List[Dict[str, Any]], task_name: str) -> None:
        # log the samples as an artifact
        dumped = json.dumps(
            data,
            indent=2,
            default=_handle_non_serializable,
            ensure_ascii=False,
        )
        artifact = wandb.Artifact(f"{task_name}", type="samples_by_task")
        with artifact.new_file(f"{task_name}_eval_samples.json", mode="w", encoding="utf-8") as f:
            f.write(dumped)
        self.run.log_artifact(artifact)
        # artifact.wait()

    def log_eval_samples(self, samples: Dict[str, List[Dict[str, Any]]]) -> None:
        """Log evaluation samples to W&B.

        Args:
            samples (Dict[str, List[Dict[str, Any]]]): Evaluation samples for each task.
        """
        task_names: List[str] = [x for x in self.task_names if x not in self.group_names]

        ungrouped_tasks = []
        tasks_by_groups = {}

        for task_name in task_names:
            group_names = self.task_configs[task_name].get("group", None)
            if group_names:
                if isinstance(group_names, str):
                    group_names = [group_names]

                for group_name in group_names:
                    if not tasks_by_groups.get(group_name):
                        tasks_by_groups[group_name] = [task_name]
                    else:
                        tasks_by_groups[group_name].append(task_name)
            else:
                ungrouped_tasks.append(task_name)

        for task_name in ungrouped_tasks:
            eval_preds = samples[task_name]

            # log the samples as a W&B Table
            df = self._generate_dataset(eval_preds, self.task_configs.get(task_name))
            self.run.log({f"{task_name}_eval_results": df})

            # log the samples as a json file as W&B Artifact
            self._log_samples_as_artifact(eval_preds, task_name)

        for group, grouped_tasks in tasks_by_groups.items():
            grouped_df = pd.DataFrame()
            for task_name in grouped_tasks:
                eval_preds = samples[task_name]
                df = self._generate_dataset(eval_preds, self.task_configs.get(task_name))
                df["group"] = group
                df["task"] = task_name
                grouped_df = pd.concat([grouped_df, df], ignore_index=True)

                # log the samples as a json file as W&B Artifact
                self._log_samples_as_artifact(eval_preds, task_name)

            self.run.log({f"{group}_eval_results": grouped_df})
