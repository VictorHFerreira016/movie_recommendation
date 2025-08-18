# The function Path is used to colect the relative Path of the user.
from pathlib import Path
# BASE_DIR stores the base directory of the project. 
# "__file__" is a special variable that holds the path of the current file.
# ".parent" is used to get the parent directory of the current file. Parent means the directory 
# that contains the current file. Ex.: /home/user/project/scripts/config.py, if ".parent" is applied 
# it will return /home/user/project/scripts.
BASE_DIR = Path(__file__).parent.parent
# DATA_RAW is the directory where the raw data is stored. It colects BASE_DIR, and add with
# "data" and "raw" to create the full path. And it do the same with IMG_DIR.
DATA_RAW = BASE_DIR / "data" / "raw"
IMG_DIR = BASE_DIR / "data" / "images"