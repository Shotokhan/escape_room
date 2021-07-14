import json


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=Singleton):
    def __init__(self, config_name="config.JSON"):
        self.config_name = config_name
        self.conf = self.read_config()

    def read_config(self):
        with open(self.config_name, 'r') as f:
            conf = json.load(f)
        return conf

    def __getitem__(self, item):
        return self.conf[item]
