def convert_timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert text timestamp in format "M.SS.ss" to total seconds
    where M=minutes, SS=seconds, ss=decimal seconds
    
    Examples:
        "1.41.91" -> 101.91 (1 min 41.91 sec)
        "1.40.12" -> 100.12 (1 min 40.12 sec)
        "0.58.41" -> 58.41 (58.41 sec)
    
    Args:
        timestamp: String timestamp in M.SS.ss format
        
    Returns:
        Float value representing total seconds
    """
    parts = timestamp.split('.')
    
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp format: {timestamp}. Expected format: M.SS.ss")
        
    minutes = float(parts[0])
    seconds = float(parts[1])
    decimal = float(parts[2]) / 100  # Convert decimal part to fraction
    
    total_seconds = minutes * 60 + seconds + decimal
    
    return total_seconds



def convert_date_to_int(date_str: str) -> int:
    """
    Convert date string in YYYY-MM-DD format to integer that preserves ordering
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        
    Returns:
        Integer in format YYYYMMDD
        
    Example:
        '2015-11-18' -> 20151118b
        '2016-03-31' -> 20160331
    """
    # Remove hyphens and convert to integer
    return int(date_str.replace('-', ''))

# Example usage:
dates = ['2015-11-18', '2015-03-25', '2016-03-31', '2015-07-05', '2016-11-06']
date_ints = [convert_date_to_int(d) for d in dates]
print(f"Original dates: {dates}")
print(f"Integer dates: {date_ints}")




def calculate_running_position_stats(df, horse_id_feature, feature_col='clean_position', specific_value = None):
    """
    Calculate running averages of a feature for each horse based on their last 3, 5 and 7 races
    
    Args:
        df: DataFrame containing horse race data with horse_id and feature columns
        feature_col: Name of the column to calculate running averages for (default: 'position')
        
    Returns:
        DataFrame with added columns for running averages
    """
    # Create copy to avoid modifying original
    result_df = df[[horse_id_feature]].copy()  # Use .copy() to avoid SettingWithCopyWarning
    
    # Create new columns using loc to avoid warnings
    new_cols = [f'{feature_col}_mavg_{n}' for n in [3,5,7]]
    for col in new_cols:
        result_df.loc[:, col] = np.nan

    # Get unique horses
    horses = df[horse_id_feature].unique()
    
    # Calculate running averages for each horse
    for horse in horses:
        # Get all races for this horse in chronological order
        horse_mask = df['horse_id'] == horse
        
        # Get feature values, replacing 99 with nan
        if specific_value is not None:
            values = df.loc[horse_mask, feature_col].replace(99, np.nan)
        else:
            values = df.loc[horse_mask, feature_col]
            
        # Calculate running means with different windows
        mavg_3 = values.rolling(window=3, min_periods=3).mean()
        mavg_5 = values.rolling(window=5, min_periods=5).mean()
        mavg_7 = values.rolling(window=7, min_periods=7).mean()
        
        # Store results using loc
        result_df.loc[horse_mask, f'{feature_col}_mavg_3'] = mavg_3
        result_df.loc[horse_mask, f'{feature_col}_mavg_5'] = mavg_5
        result_df.loc[horse_mask, f'{feature_col}_mavg_7'] = mavg_7
        
    return result_df



def evaluate_race_predictions(df, pred_probs, race_ids):
    """
    Evaluate prediction quality for a list of races
    
    Args:
        df: DataFrame with race results
        pred_probs: Dict of {race_id: list of predictions ordered by draw number}
        race_ids: List of race IDs to evaluate
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    metrics = {
        'top1_accuracy': [],  # Did we predict the winner?
        'top3_accuracy': [],  # Did our top 3 contain any actual top 3?
        'top3_overlap': [],   # How many of our top 3 were in actual top 3?
        'winner_pred_rank': [],  # Where did we rank the actual winner?
        'winner_pred_value': [],  # What probability did we give the winner?
    }
    
    for race_id in race_ids:
        # Get race data
        race_df = df[df['race_id'] == race_id].sort_values('draw_number')
        
        # Get predictions for this race
        race_preds = pred_probs[race_id]
        
        # Get actual results
        actual_winner_idx = race_df['is_winner'].values.argmax()
        actual_top3_idx = set(np.where(race_df['is_top3'] == 1)[0])
        
        # Get our predictions
        pred_top1_idx = np.argmax(race_preds)
        pred_top3_idx = set(np.argsort(race_preds)[-3:])
        
        # Calculate metrics
        metrics['top1_accuracy'].append(int(pred_top1_idx == actual_winner_idx))
        metrics['top3_accuracy'].append(int(len(pred_top3_idx & actual_top3_idx) > 0))
        metrics['top3_overlap'].append(len(pred_top3_idx & actual_top3_idx) / 3)
        metrics['winner_pred_rank'].append(len(race_preds) - np.where(np.argsort(race_preds) == actual_winner_idx)[0][0])
        metrics['winner_pred_value'].append(race_preds[actual_winner_idx])
    
    # Calculate final metrics
    return {
        'top1_accuracy': np.mean(metrics['top1_accuracy']),
        'top3_accuracy': np.mean(metrics['top3_accuracy']),
        'top3_overlap': np.mean(metrics['top3_overlap']),
        'avg_winner_rank': np.mean(metrics['winner_pred_rank']),
        'avg_winner_prob': np.mean(metrics['winner_pred_value'])
    }


# The key metrics to evaluate prediction quality are:
# Top 1 Accuracy
# Did we correctly predict the winner?
# Simple but harsh metric as horse racing is highly variable
# Top 3 Accuracy
# Did any of our top 3 predictions finish in the actual top 3?
# More forgiving metric that accounts for racing variability
# Top 3 Overlap
# How many of our predicted top 3 actually finished in top 3?
# Ranges from 0 to 1 (0 = no overlap, 1 = perfect prediction)
# Winner Prediction Rank
# Where did we rank the actual winner in our predictions?
# Helps understand if we were "close" even when wrong
# Winner Prediction Value
# What probability did we assign to the actual winner?
# Helps evaluate calibration of our predictions


# Example usage
# metrics = evaluate_race_predictions(
#     df=race_results_df,
#     pred_probs={
#         'race1': [0.1, 0.2, 0.3, 0.4],  # predictions for each horse by draw
#         'race2': [0.2, 0.3, 0.5]
#     },
#     race_ids=['race1', 'race2']
# )

# print("Prediction Quality Metrics:")
# for metric, value in metrics.items():
#     print(f"{metric}: {value:.3f}")


# This evaluation framework provides a comprehensive view of prediction quality, considering both exact predictions (top1) and "close" predictions (top3), which is important in horse racing where outcomes can be highly variable.