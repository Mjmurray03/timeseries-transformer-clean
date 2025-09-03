import numpy as np
from typing import List, Union
from fastapi import HTTPException

def validate_and_reshape_features(features: Union[List[float], List[List[float]]]) -> List[List[float]]:
    """
    Validates and reshapes features to proper 60x10 format
    """
    # Handle flat list
    if isinstance(features, list) and len(features) > 0:
        if isinstance(features[0], (int, float)):
            # Flat list of 600 elements
            if len(features) == 600:
                # Reshape to 60x10
                reshaped = []
                for i in range(60):
                    row = features[i*10:(i+1)*10]
                    reshaped.append(row)
                return reshaped
            else:
                raise HTTPException(
                    status_code=422,
                    detail=f"Flat feature array must have exactly 600 elements (60 timesteps x 10 features), got {len(features)}"
                )
        elif isinstance(features[0], list):
            # Already 2D array
            if len(features) == 60 and all(len(row) == 10 for row in features):
                return features
            else:
                raise HTTPException(
                    status_code=422,
                    detail=f"2D feature array must be 60x10, got {len(features)}x{len(features[0]) if features else 0}"
                )
    
    raise HTTPException(
        status_code=422,
        detail="Features must be either a flat list of 600 values or a 60x10 2D array"
    )