from .arrays import *
from .config import *
from .serialization import *
from .setup import *
from .training import *


class LcdParser(Parser):
    config: str = "lcd.config.calvin"
    
def d_args():
    return LcdParser().parse_args("diffusion", add_extras=False)
