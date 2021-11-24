import logging
from datetime import datetime
from pathlib import Path
import yaml

with open(str(Path.cwd() / 'CONFIG.yaml')) as fh:
    config = yaml.safe_load(fh)

image_w, image_h, image_d = 256, 256, 400  # config['image_w'], config['image_h'], config['image_d']

today = datetime.now()
timestamp = datetime.now().strftime('%m-%d-%Y_%H.%M.%S')
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(Path.cwd() / f'logs/{today}.log', 'w'),
                              logging.StreamHandler()])

log = logging.getLogger(__name__)
