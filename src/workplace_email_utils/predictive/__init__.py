"""
Predictive analytics module.

Includes response time prediction, volume forecasting, priority prediction,
and escalation prediction.
"""

from .response_time import (
    predict_response_time,
    train_response_time_model,
    ResponseTimePredictor
)
from .volume_forecast import (
    forecast_email_volume,
    train_volume_forecast_model,
    VolumeForecastModel
)
from .priority_prediction import (
    predict_priority_score,
    train_priority_predictor,
    PriorityPredictor
)
from .escalation import (
    predict_escalation_risk,
    train_escalation_model,
    EscalationPredictor
)

__all__ = [
    'predict_response_time',
    'train_response_time_model',
    'ResponseTimePredictor',
    'forecast_email_volume',
    'train_volume_forecast_model',
    'VolumeForecastModel',
    'predict_priority_score',
    'train_priority_predictor',
    'PriorityPredictor',
    'predict_escalation_risk',
    'train_escalation_model',
    'EscalationPredictor',
]

