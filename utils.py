import argparse
import json
import os
import pickle
from collections import ChainMap

import yaml


def read_file(filepath, encoding="utf-8"):
    """Read text from a file."""
    try:
        with open(filepath, encoding=encoding) as f:
            return f.read()
    except FileNotFoundError as e:
        raise ValueError(f"File '{filepath}' does not exist.") from e


def read_json_file(filepath):
    """Read json from a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except ValueError as e:
        abspath = os.path.abspath(filepath)
        raise ValueError(f"Failed to read json from '{abspath}") from e


def json_to_string(js_dic, **kwargs):
    """Converts a dictionary to a string."""
    indent = kwargs.pop("indent", 2)
    ensure_ascii = kwargs.pop("ensure_ascii", False)
    return json.dumps(js_dic,
                      indent=indent,
                      ensure_ascii=ensure_ascii,
                      **kwargs)


def write_text_file(content, filepath, encoding='utf-8', append=False):
    """Writes text to a file.
    Args:
        content: The content to write.
        filepath: The path to which the content should be written.
        encoding: The encoding which should be used.
        append: Whether to append to the file or to truncate the file.
    """
    mode = "a" if append else "w"
    with open(filepath, mode, encoding=encoding) as file:
        file.write(content)


def write_json_file(js_obj, filepath, **kwargs):
    """Writes object to a json_file."""
    json_string = json_to_string(js_obj, **kwargs)
    write_text_file(json_string, filepath)


def load_pickle(filepath):
    """Loads an object from pickle."""
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError as e:
        raise ValueError(f"File '{filepath}' does not exist.") from e


def save_pickle(filepath, obj):
    """Saves object to a pickle."""
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def read_yaml(yaml_path):
    """Read yaml from a file."""
    try:
        with open(yaml_path, "r") as f:
            return yaml.load(f, Loader=yaml.loader.Loader)
    except FileNotFoundError as e:
        raise ValueError(f"File '{yaml_path}' does not exist.") from e


def write_yaml(yaml_obj, yaml_path):
    """Write dict to a yaml config."""
    write_text_file(yaml.safe_dump(yaml_obj), yaml_path)


def read_config(config_path):
    return dict2namespace(read_yaml(config_path))


def join_list_of_dict(ls):
    """Joins a list of dictionaries into a single dictionary."""
    return dict(ChainMap(*ls))


def list_dict_to_dict_list(ls):
    return {k: [dic[k] for dic in ls] for k in ls[0]}


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
