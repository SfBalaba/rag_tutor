import os
from string import Template

import yaml
from dotenv import load_dotenv

load_dotenv()


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

    with open(config_path, encoding="utf-8") as f:
        config_content = f.read()

    template = Template(config_content)
    config_content = template.safe_substitute(os.environ)

    return yaml.safe_load(config_content)


config = load_config()
