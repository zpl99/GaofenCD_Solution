import json


def readConfigs(configs):
    with open(configs) as f:
        json_ = json.load(f)
    return json_
