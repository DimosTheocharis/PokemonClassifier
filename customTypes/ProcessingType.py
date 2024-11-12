from enum import Enum

class ProcessingType(Enum):
    ClockwiseRotation = "CR"
    CounterClockwiseRotation = "CCR"
    IncreaseBrightness = "IB"
    DecreaseBrightness = "DB"
    CropCenter = "CC"
    CropTopLeftCorner = "CTLC"
    CropTopRightCorner = "CTRC"
    CropBottomRightCorner = "CBRC"
    CropBottomLeftCorner = "CBLC"
    Translate = "T"
