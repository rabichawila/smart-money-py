from dataclasses import dataclass
import enum

class TimeFrame(enum.Enum):
    M5   = "5m"
    M15  = "15m"
    M30  = "30m"
    H4   = "4h"
    D1   = "1d"


class BuySell(enum.Enum):
   
   Neutral     = 0
   Buy         = 1
   Sell        = 2
   BuyLimit    = 3
   BuyStop     = 4
   SellLimit   = 5
   SellStop    = 6
   
class Direction(enum.Enum):
    UP      = "UP"
    DOWN    = "DOWN"
    NONE    = "NONE"

@dataclass
class Channel:
    is_channel: bool        =       False

    lower_slop: float       =       0
    upper_slop: float       =       0

    lower_intersect: float  =       0
    upper_intersect: float  =       0

    lower_percent: float    =       0
    upper_percent: float    =       0

    lower_line_val: float   =       0
    upper_line_val: float   =       0
