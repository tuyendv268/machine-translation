from src import utils
import configparser

config = configparser.ConfigParser()
config.read('src/conf/config.cfg')

# print(config[""])
path = config["path"]["train_path"]
utils.build_en_dict(path)