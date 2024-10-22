from tap import Tap
from typing import Literal


class Args(Tap):
    solver: Literal["ali", "ipopt", "altro"]
    record: bool = False
