cp MultiFL/fed_config_DeepLabv3_carla_fedavg.py MultiFL/fed_config.py
python fed_train.py

cp MultiFL/fed_config_DeepLabv3_carla_centralized.py MultiFL/fed_config.py
python fed_train.py
