from typing import List
import importlib.util
import logging
import importlib.metadata
import importlib
import os
from collections import OrderedDict

logger = logging.getLogger(__name__)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_per_module(model):
    """
    Return an ordered dictionary containing the number of trainable parameters for each submodule in the model
    Format: {module_name: parameter_count}
    """
    module_params = OrderedDict()
    # Iterate through all submodules (including nested structures)
    for name, module in model.named_modules():
        # Skip the top-level module (top-level module name is an empty string)
        if not name:
            continue
        # Calculate the number of parameters in the current module itself (excluding submodules)
        params = 0
        for p in module.parameters(recurse=False):  # Key: recurse=False only checks current module parameters
            if p.requires_grad:
                params += p.numel()

        # Only record modules with parameters
        if params > 0:
            module_params[name] = params

    return module_params


def print_detailed_parameters(model):
    # Get parameters for each module
    params_dict = count_parameters_per_module(model)

    # Calculate total
    total = sum(params_dict.values())

    # Print table
    print("{:<20} {:<15} {:<15}".format('Module', 'Params', 'Percentage'))
    print('-' * 45)
    for name, params in params_dict.items():
        percent = 100 * params / total
        print("{:<20} {:<15,} {:>.2f}%".format(name, params, percent))
    print('-' * 45)
    print("{:<20} {:<15,} {:<15}".format('Total', total, '100%'))


def replace_eval_with_test(metrics):
    """
    Replace keys from 'eval_' to 'test_' in the dictionary.
    :param metrics: Dictionary containing evaluation metrics
    :return: Dictionary with replaced keys
    """
    new_metrics = {}
    for key, value in metrics.items():
        if key.startswith('eval_'):
            new_key = key.replace('eval_', 'test_')
        else:
            new_key = key
        new_metrics[new_key] = value
    return new_metrics


# Define the import_modules function, which accepts two parameters:
# - models_dir: Directory where model files are stored
# - namespace: Module namespace used to build the complete module path
def import_modules(models_dir, namespace, specific_models: List[str] = None):
    # Iterate through all files in the models_dir directory
    for file in os.listdir(models_dir):
        # Get the full path of the current file
        path = os.path.join(models_dir, file)

        # Check if the file is a valid Python file
        if (
                not file.startswith("_")  # File name does not start with "_" (usually indicates internal files)
                and not file.startswith(".")  # File name does not start with "." (hidden files)
                and file.endswith(".py")  # File ends with ".py" (Python source files)
        ):
            # Extract module name from the filename (remove .py suffix)
            if specific_models == None:
                model_name = file[: file.find(".py")] if file.endswith(".py") else file
                # Dynamically import the module, construct the full module path (namespace + model_name)
                importlib.import_module(namespace + "." + model_name)
            else:
                for model_name in specific_models:
                    importlib.import_module(namespace + "." + model_name)