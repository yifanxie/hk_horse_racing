# check package - lightgbm, xgboost, bayesian-optimization, 


# major technical items to review:
# 1. groupby-aggregator handy func
# 2. numerai validator class with different algo
# 3. null hypo. feature selection routine - if not using the numerai class
# 4. weight optimisation routine 


# handy code to develop to prep for interview
# 1. final position to target transformation 
# 2. probability normalization for horses in the same race
# 3. prediction optimisation for metric per race 



# generic plan 
# 1. setup 
# 1.1 set up package and load data
# 1.2 copy & paste core utilit function for machine learning

# 2. initial analysis
# 2.1 dataframe.info 
# 2.2 key columns format, null count
# 2.3 remove unrelevant data - check finishing position, win_odds, missing finishing time, etc
# 2.4 identify data cleaning target

# 3. target & ml trainable dataset - first pass
# 3.1 generate measurable targets - if needed 
# 3.2 generate baseline prediction and evaluation
# 3.3 generate machine trainable dataset - for tree and for linear & NN algo.


# 4. ML first pass
# 4.1 lightgbm - basic numerical and categorical features
# 4.2 linear model with ridge regression 
# 


# 5.1 identify "very import features" - do transformation of those if needed 
# 6. generate additional features


# 7. train with alternative targets
# 8. train with alternative problem framing & data setup 




# to-do:
# 1. create generate data analysis coding blocks including simple visualizations on notebook
# 2. switch to jupyber lab setting, and write the code there 





## potential feature engineering:
# jockey performance
# training performance
# domain specific feature:
# horse group dynamics
# jockey group dynamics
# horse/trainer/jockey/race course dynamics
# horse weather dynamics


## alternative model approach:
# class specfic model
# track specific model
#


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


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from scipy.stats import kendalltau, spearmanr

def evaluate_horse_race_positions(y_true, y_pred_proba, dnf_value=99):
    n_races, n_horses = y_pred_proba.shape
    
    # Handle NaN values and DNF values in ground truth by replacing with max rank + 1
    y_true_processed = y_true.copy()
    for i in range(n_races):
        # Create mask for both NaN and DNF values
        invalid_mask = np.logical_or(
            np.isnan(y_true[i]),
            y_true[i] == dnf_value
        )
        
        # Get max valid rank in this race (excluding DNF values)
        valid_ranks = y_true[i][~invalid_mask]
        if len(valid_ranks) > 0:
            max_rank = np.max(valid_ranks)
            # Replace invalid values with max_rank + 1
            y_true_processed[i][invalid_mask] = max_rank + 1
    
    # Convert probabilities to predicted rankings
    y_pred_ranks = n_horses - np.argsort(y_pred_proba, axis=1)
    
    # Mean Squared Error
    mse = mean_squared_error(y_true_processed, y_pred_ranks)
    
    # Mean Absolute Error
    mae = mean_absolute_error(y_true_processed, y_pred_ranks)
    
    # Spearman's Rank Correlation
    spearman_corr = np.mean([spearmanr(y_true_processed[i], y_pred_ranks[i]).correlation 
                             for i in range(n_races)])
    
    # Normalized Discounted Cumulative Gain (NDCG)
    ndcg = ndcg_score(y_true_processed.reshape(1, -1), y_pred_proba.reshape(1, -1))
    
    # Top 3 Exact Match
    top3_exact_match = np.mean([np.array_equal(np.sort(y_true_processed[i][:3]), np.sort(y_pred_ranks[i][:3])) 
                               for i in range(n_races)])
    
    # Top-K Accuracy (for K=1, 2, 3)
    top_k_accuracy = {}
    for k in [1, 2, 3]:
        top_k_pred = np.argsort(-y_pred_proba, axis=1)[:, :k]
        top_k_true = np.argsort(y_true_processed, axis=1)[:, :k]
        top_k_accuracy[f'Top-{k} Accuracy'] = np.mean([
            len(set(top_k_pred[i]) & set(top_k_true[i])) / k 
            for i in range(n_races)
        ])
    
    return {
        'Mean Squared Error': mse,
        'Mean Absolute Error': mae,
        "Spearman's Rank Correlation": spearman_corr,
        'NDCG': ndcg,
        'Top 3 Exact Match': top3_exact_match,
        **top_k_accuracy
    }




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


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from scipy.stats import kendalltau, spearmanr

def evaluate_horse_race_positions(y_true, y_pred_proba, dnf_value=99):
    """
    Evaluate predictions for a single race's finishing positions.
    
    Args:
        y_true: 1D array of true finishing positions
        y_pred_proba: 1D array of predicted probabilities
        dnf_value: Value used to indicate Did Not Finish
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Handle NaN and DNF values in ground truth
    y_true_processed = y_true.copy()
    invalid_mask = np.logical_or(
        np.isnan(y_true),
        y_true == dnf_value
    )
    
    # Get max valid rank (excluding DNF values)
    valid_ranks = y_true[~invalid_mask]
    if len(valid_ranks) > 0:
        max_rank = np.max(valid_ranks)
        # Replace invalid values with max_rank + 1
        y_true_processed[invalid_mask] = max_rank + 1
        
    # Winner match
    y_true_ranksort = np.argsort(y_true_processed)
    y_pred_ranksort = np.argsort(y_pred_proba, axis=0)[::-1]
    
    winner_match = y_true_ranksort[0] == y_pred_ranksort[0]
    
    # Top 3 Set Match - considers [1,3,2] and [2,3,1] as matching
    top3_set_match = set(y_true_ranksort[:3]) == set(y_pred_ranksort[:3])
    
    # Top 3 Exact Match - only considers exact matches like [1,3,2] and [1,3,2]
    top3_exact_match = np.array_equal(y_true_ranksort[:3], y_pred_ranksort[:3])
    
    return {
        'Winner Match': float(winner_match),
        'Top 3 Set Match': float(top3_set_match), 
        'Top 3 Exact Match': float(top3_exact_match)
    }






def evaluate_prediction_sets(eval_dict):
    """
    Evaluate different prediction sets against ground truth for each race and calculate mean metrics
    
    Args:
        eval_dict: Dictionary containing race data with ground truth and different prediction sets
        
    Returns:
        tuple: (eval_result, mean_results_df)
            - eval_result: Dictionary with detailed evaluation metrics for each race
            - mean_results_df: DataFrame comparing mean metrics across prediction types
    """
    # Initialize results dictionary with race_ids as first level keys
    eval_result = {race_id: {} for race_id in eval_dict}

    # Get prediction types from first race data
    first_race_id = next(iter(eval_dict))
    pred_types = [key for key in eval_dict[first_race_id].keys() if key != 'ground_truth']

    # Initialize dictionaries to store mean results
    mean_results = {pred_type: {} for pred_type in pred_types}

    # Loop through each race
    for race_id in eval_dict:
        race_data = eval_dict[race_id]
        ground_truth = race_data['ground_truth']
        
        # Evaluate each prediction type
        for pred_type in pred_types:
            pred_probs = race_data[pred_type]
            
            # Evaluate predictions for this race
            race_eval = evaluate_horse_race_positions(
                ground_truth,
                pred_probs
            )
            
            # Store results for this race under race_id first, then pred_type
            eval_result[race_id][pred_type] = race_eval

    # Calculate mean results for each prediction type
    for pred_type in pred_types:
        # Initialize dict to store means for each metric
        metric_means = {}
        
        # Get metrics from first race to know what metrics exist
        first_race = next(iter(eval_result.values()))
        metrics = first_race[pred_type].keys()
        
        # For each metric, calculate mean across all races
        for metric in metrics:
            total = 0
            num_races = 0
            for race_id in eval_result:
                total += eval_result[race_id][pred_type][metric]
                num_races += 1
            metric_means[metric] = total / num_races
            
        mean_results[pred_type] = metric_means
    
    # Convert mean results to DataFrame for easy comparison
    mean_results_df = pd.DataFrame(mean_results)
    
    return eval_result, mean_results_df

# # Run evaluation
# eval_result, mean_results_df = evaluate_prediction_sets(eval_dict)

# # Display mean results comparison
# print("\nMean Evaluation Metrics Comparison:")
# print(mean_results_df)


# Run evaluation
eval_result, mean_results_df = evaluate_prediction_sets(eval_dict)

# Display mean results comparison
print("\nMean Evaluation Metrics Comparison:")
print(mean_results_df)



def train_lightgbm_model(train_df, val_df, label_col, cat_features=None, params=None):
    """
    Train a LightGBM model for binary classification using LGBMClassifier
    
    Args:
        train_df: Training dataframe containing features and label
        val_df: Validation dataframe containing features and label  
        label_col: Name of label column (should contain binary values 0/1)
        cat_features: List of categorical feature names
        params: Dict of LightGBM parameters
        
    Returns:
        Trained model and validation predictions
    """
    # Default parameters if none provided
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,  # Column sampling
            'bagging_fraction': 0.8,  # Row sampling 
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': -1,
            'min_child_samples': 20,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'is_unbalance': True  # Handle unbalanced datasets
        }

    # Separate features and labels
    features = [col for col in train_df.columns if col != label_col]
    X_train = train_df[features]
    y_train = train_df[label_col]
    X_val = val_df[features]
    y_val = val_df[label_col]

    # Initialize and train model
    model = lgb.LGBMClassifier(**params)
    
    # Fit model
    model.fit(
        X_train, y_train,
        n_estimators=150,
        categorical_feature=cat_features if cat_features else 'auto',
        verbose=100
    )
    
    # Make validation predictions
    val_preds = model.predict_proba(X_val)[:, 1]  # Get probability of positive class
    val_logloss = log_loss(y_val, val_preds)
    val_acc = accuracy_score(y_val, val_preds > 0.5)  # Convert probs to binary predictions
    print(f'Validation LogLoss: {val_logloss:.4f}')
    print(f'Validation Accuracy: {val_acc:.4f}')
    
    return model, val_preds

# Example usage:
"""
# Example parameters
params = {
    'objective': 'regression',  
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

cat_features = ['track_id', 'horse_id', 'jockey_id']

model, val_preds = train_lightgbm_model(
    train_df=train_data,
    val_df=val_data, 
    label_col='finishing_time',
    cat_features=cat_features,
    params=params
)
"""

