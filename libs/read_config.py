import json

class ReadConfig:
    """
    Class represent reading config and all variables from config file.
    """
    def __init__(self, config_path:str) -> None:
        self.config_path = config_path
    
    def read_config(self) -> dict:
        """ Read all variables from file"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        return config