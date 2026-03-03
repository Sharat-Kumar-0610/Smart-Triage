# temporal_features.py

def compute_temporal_score(time_since_onset, worsening_flag):
    """
    time_since_onset in hours
    worsening_flag = 0 or 1
    """

    time_factor = min(time_since_onset / 24, 1)

    temporal_score = (
        0.6 * time_factor +
        0.4 * worsening_flag
    )

    return temporal_score