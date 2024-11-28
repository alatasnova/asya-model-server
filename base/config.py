import configparser, json
import os.path

CONFIG_PATH = "MODEL_CONFIG.ini"

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Can't find {os.path.join(os.getcwd(), CONFIG_PATH)} config file.")

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

# SERVER
SERVER_DEVICE = config.get("Server", "device")

# TRAIN
TRAIN_DEVICE = config.get("Train", "device")
TRAIN_NUM_EPOCHS = int(config.get("Train", "num_epochs"))

# RECOGNITION
RECOGNITION_LABELS = json.loads(config.get("Recognition", "labels"))
RECOGNITION_SAVE_PATH = config.get("Recognition", "save_path")
RECOGNITION_VOCAB_PATH = config.get("Recognition", "vocab_path")
RECOGNITION_DATASET_PATH = config.get("Recognition", "dataset_path")