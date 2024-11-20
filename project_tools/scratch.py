# check package - lightgbm, xgboost, bayesian-optimization, 


# key questions to ask:
# 1. ask to use jupyter lab instead of notebook
# 2. ask about setting up project utilities file, and project folder/file structure
# 3. ask about/check what get-started code are provided
# 4. ask about how/when to raise questions mid-way throught the technical assessment
# 5. ask about installing additional packages




# major technical items to review:
# 1. groupby-aggregator handy func - done
# 2. numerai validator class with different algo - no need
# 3. null hypo. feature selection routine - if not using the numerai class - tbd
# 4. weight optimisation routine - no need
# 5. race by race progressive prediction by increasling adding race into origional training set - to do tonight
# 6. NN: race-to-finnish position modelling - to do tonight 
# 7. additional feature engineering - to do tonight
# 7.1 days since last race for each horse and jockey
# 6.2 horse and jockey number of race in the same track in the last 30 days 
# 7.3 horse and jockey win rate in the last 30 days     
# 7.4 horse and jockey top 3 rate in the last 30 days 
# 7.5 horse and jockey top length to winner in the last 30 days
# 7.6 group track condition into smaller amount of categories 
# 8. download and check out new york horse racing data
# 9. generate additional ideas for - features/data processing/problem framing/modelling approach/post processing



# handy code to develop to prep for interview
# 1. final position to target transformation  - done
# 2. probability normalization for horses in the same race - leave for now
# 3. prediction optimisation for metric per race - leave for now 



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




def prepare_pairwise_ranking_data(df, race_id_col='race_id', label_col='clean_position',
                                  keep_race_id_col=False):
    """
    Prepare data for pairwise ranking by generating pairs of samples within each race.
    For each race, creates pairs of horses where one finished ahead of the other.
    
    Args:
        df: DataFrame containing race data
        race_id_col: Column name for race identifier
        label_col: Column name containing finishing position
        
    Returns:
        DataFrame with pairwise samples, containing features for both horses and binary label
        indicating if first horse finished ahead of second horse
    """
    feature_cols = [c for c in df.columns if c not in [race_id_col, label_col]]
    pairs_data = []
    
    # Process each race
    for race_id in df[race_id_col].unique():
        race_df = df[df[race_id_col] == race_id].copy()
        
        # Generate all pairs of horses in this race
        for i in range(len(race_df)):
            for j in range(i+1, len(race_df)):
                horse1 = race_df.iloc[i]
                horse2 = race_df.iloc[j]
                
                # Create feature vectors for both horses
                features1 = horse1[feature_cols].values
                features2 = horse2[feature_cols].values
                
                # Label is 1 if horse1 finished ahead of horse2
                label = 1 if horse1[label_col] < horse2[label_col] else 0
                if not keep_race_id_col:                
                    pairs_data.append(np.concatenate([
                        features1,features2,[label]
                    ]))                    
                    # Add reverse pair with opposite label
                    pairs_data.append(np.concatenate([
                        features2, features1,[1-label]
                    ]))
                else:
                    pairs_data.append(np.concatenate([
                        features1,features2,[label, race_id]
                    ]))                    
                    # Add reverse pair with opposite label
                    pairs_data.append(np.concatenate([
                        features2, features1,[1-label, race_id], 
                    ]))
    
    # Create column names for paired data
    pair_cols = []
    pair_cols1 = [f'{col}_1' for col in feature_cols]
    pair_cols2 = [f'{col}_2' for col in feature_cols]
    pair_cols = pair_cols1 + pair_cols2
    pair_cols.append('label')
    if keep_race_id_col:
        pair_cols.append('race_id')    
    
    return pd.DataFrame(pairs_data, columns=pair_cols)



def train_pairwise_ranker_ridge(train_df, val_df=None, race_id_col='race_id', label_col='clean_position',
                               alpha=1.0):
    """
    Train a Ridge Classifier model for pairwise ranking of horses.
    Converts the ranking problem into binary classification of horse pairs.
    
    Args:
        train_df: Training DataFrame with features and labels
        val_df: Optional validation DataFrame
        race_id_col: Column name for race identifier 
        label_col: Column name containing position/rank labels
        alpha: Regularization strength parameter for Ridge Classifier
        
    Returns:
        Trained model and validation predictions (if val_df provided)
    """
    from sklearn.linear_model import RidgeClassifier
    
    # Prepare pairwise training data
    print("Preparing pairwise training data...")
    train_pairs = prepare_pairwise_ranking_data(
        train_df,
        race_id_col=race_id_col,
        label_col=label_col
    )
    
    # Split features and labels
    X_train = train_pairs.drop(['label'], axis=1)
    y_train = train_pairs['label']
    
    # Initialize and train model
    model = RidgeClassifier(alpha=alpha)
    model.fit(X_train, y_train)
    
    val_preds = None
    if val_df is not None:
        print("Preparing validation data...")
        val_pairs = prepare_pairwise_ranking_data(
            val_df,
            race_id_col=race_id_col, 
            label_col=label_col
        )
        X_val = val_pairs.drop(['label'], axis=1)
        val_preds = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_val)
        
    return model, val_preds





def train_pairwise_ranker(train_df, val_df=None, race_id_col='race_id', label_col='clean_position',
                         cat_features=None, params=None):
    """
    Train a LightGBM model for pairwise ranking of horses.
    Converts the ranking problem into binary classification of horse pairs.
    
    Args:
        train_df: Training DataFrame with features and labels
        val_df: Optional validation DataFrame
        race_id_col: Column name for race identifier 
        label_col: Column name containing position/rank labels
        cat_features: List of categorical feature names
        params: LightGBM parameters dict
        
    Returns:
        Trained model and validation predictions (if val_df provided)
    """
    # Default parameters if none provided
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': -1,
            'min_child_samples': 20,
        }
    
    # Prepare pairwise training data
    print("Preparing pairwise training data...")
    train_pairs = prepare_pairwise_ranking_data(
        train_df, 
        race_id_col=race_id_col,
        label_col=label_col
    )
    
    # Split features and labels
    X_train = train_pairs.drop('label', axis=1)
    y_train = train_pairs['label']
    
    # Handle categorical features if provided
    if cat_features:
        cat_features_pairs = []
        for feat in cat_features:
            cat_features_pairs.extend([f'{feat}_1', f'{feat}_2'])
    else:
        cat_features_pairs = 'auto'
    
    # Prepare validation data if provided
    if val_df is not None:
        print("Preparing pairwise validation data...")
        val_pairs = prepare_pairwise_ranking_data(
            val_df,
            race_id_col=race_id_col,
            label_col=label_col
        )
        X_val = val_pairs.drop('label', axis=1)
        y_val = val_pairs['label']
    
    # Initialize and train model
    print("Training model...")
    model = lgb.LGBMClassifier(**params)
    
    if val_df is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=cat_features_pairs,
            verbose=100
        )
        # Get validation metrics
        val_preds = model.predict_proba(X_val)[:, 1]
        val_logloss = log_loss(y_val, val_preds)
        val_acc = accuracy_score(y_val, val_preds > 0.5)
        print(f'Validation LogLoss: {val_logloss:.4f}')
        print(f'Validation Accuracy: {val_acc:.4f}')
        return model, val_preds
    else:
        model.fit(
            X_train, y_train,
            categorical_feature=cat_features_pairs,
            verbose=100
        )
        return model



def minmax_scale_features(df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
    """
    Scale features in a dataframe using MinMaxScaler, excluding specified columns.
    Modifies the input dataframe in place by replacing values with scaled versions.
    
    Args:
        df: Input pandas DataFrame to scale
        exclude_cols: List of column names to exclude from scaling (e.g. target, categorical cols)
        
    Returns:
        DataFrame with features scaled to 0-1 range
    """
    from sklearn.preprocessing import MinMaxScaler
    
    # Make copy of input df
    result_df = df.copy()
    
    # Determine columns to scale
    if exclude_cols is None:
        exclude_cols = []
    cols_to_scale = [col for col in df.columns if col not in exclude_cols]
    
    # Initialize and fit scaler
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(result_df[cols_to_scale])
    
    # Replace values with scaled version
    result_df[cols_to_scale] = scaled_values
    
    return result_df







def train_lightgbm_ranker(train_df, val_df=None, race_id_col='race_id', label_col='clean_position', 
                         cat_features=None, params=None, n_estimators=150):
    """
    Train a LightGBM ranking model for horse race position prediction.
    
    Args:
        train_df: Training DataFrame with features and labels
        val_df: Optional validation DataFrame with features and labels
        race_id_col: Column name for race identifier
        label_col: Column name containing position/rank labels
        cat_features: List of categorical feature names
        params: LightGBM parameters dict
        n_estimators: Number of boosting rounds
        
    Returns:
        Trained model and validation predictions (if val_df provided)
    """
    # Default ranking parameters if none provided
    if params is None:
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_position': 20,  # Maximum number of positions to consider
            'label_gain': list(range(20)), # Gain for each position 0-19
        }

    # Prepare features and group info
    features = [col for col in train_df.columns if col not in [label_col, race_id_col]]
    X_train = train_df[features]
    y_train = train_df[label_col]
    # Convert positions to gains (lower position = higher gain)
    max_pos = y_train.max()
    y_train = max_pos - y_train + 1
    
    # Get group sizes for training data
    train_groups = train_df.groupby(race_id_col).size().values

    # Create training dataset
    train_dataset = lgb.Dataset(
        X_train, 
        label=y_train,
        group=train_groups,
        categorical_feature=cat_features if cat_features else 'auto'
    )
    
    # Prepare validation data if provided
    if val_df is not None:
        X_val = val_df[features]
        y_val = val_df[label_col]
        y_val = max_pos - y_val + 1
        val_groups = val_df.groupby(race_id_col).size().values
        
        val_dataset = lgb.Dataset(
            X_val,
            label=y_val, 
            group=val_groups,
            reference=train_dataset,
            categorical_feature=cat_features if cat_features else 'auto'
        )
        valid_sets = [val_dataset]
    else:
        valid_sets = None
        X_val = None

    # Train model
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=n_estimators,
        valid_sets=valid_sets,
        verbose_eval=100 if val_df is not None else -1
    )
    
    # Get validation predictions if validation data was provided
    val_preds = model.predict(X_val) if val_df is not None else None
    
    return model, val_preds


# Example usage:
"""
# Prepare data ensuring each race's horses are grouped together
race_data = pd.DataFrame({
    'race_id': [1,1,1,1, 2,2,2,2],
    'horse_id': [101,102,103,104, 201,202,203,204],
    'clean_position': [1,2,3,4, 2,1,4,3],
    'speed_rating': [95,92,88,85, 94,96,86,89],
    'weight': [120,118,122,119, 121,120,118,122],
    'jockey_id': [1,2,3,4, 2,1,4,3]
})

# Example 1: With validation data
train_races = [1]  
val_races = [2]
train_data = race_data[race_data['race_id'].isin(train_races)]
val_data = race_data[race_data['race_id'].isin(val_races)]

params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'max_position': 4,  # Max horses per race
    'label_gain': [3,2,1,0]  # Gains for positions 1-4
}

model1, val_preds = train_lightgbm_ranker(
    train_df=train_data,
    val_df=val_data,
    race_id_col='race_id',
    label_col='clean_position',
    cat_features=['horse_id', 'jockey_id'],
    params=params
)

# Example 2: Without validation data
model2, _ = train_lightgbm_ranker(
    train_df=race_data,
    val_df=None,
    race_id_col='race_id',
    label_col='clean_position',
    cat_features=['horse_id', 'jockey_id'],
    params=params,
    n_estimators=100
)
"""








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





def calculate_running_position_stats(df, mavg_feature, target_col='clean_position', 
                                     calculate_offset = True,
                                     specific_value=None):
    """
    Calculate running averages of a feature for each horse based on their previous 3, 5 and 7 races,
    excluding the current race to avoid data leakage.
    
    Args:
        df: DataFrame containing horse race data with horse_id and feature columns
        horse_id_feature: Column name containing horse IDs
        feature_col: Name of the column to calculate running averages for (default: 'clean_position')
        specific_value: Value to replace with NaN if specified
        
    Returns:
        DataFrame with added columns for running averages of previous races
    """
    # Create copy to avoid modifying original
    result_df = df[[mavg_feature]].copy()
    
    # Initialize new columns
    result_df[f'{mavg_feature}_{target_col}_mavg_3'] = np.nan
    result_df[f'{mavg_feature}_{target_col}_mavg_5'] = np.nan 
    result_df[f'{mavg_feature}_{target_col}_mavg_7'] = np.nan

    # Get unique horses
    horses = df[mavg_feature].unique()
    
    # Calculate running averages for each horse
    for horse in horses:
        # Get all races for this horse in chronological order
        horse_mask = df[mavg_feature] == horse
        horse_data = df[horse_mask].copy()
        
        # Get feature values, replacing specific value with nan if needed
        if specific_value is not None:
            values = horse_data[target_col].replace(specific_value, np.nan)
        else:
            values = horse_data[target_col]
            
        # Calculate means of previous races for each row
        for i in range(len(horse_data)):
            if calculate_offset:
                prev_values = values.iloc[:i]  # Only use races before current row
            else:
                prev_values = values
            # Calculate means if we have enough previous races
            if len(prev_values) >= 3:
                result_df.loc[horse_data.index[i], f'{mavg_feature}_{target_col}_mavg_3'] = prev_values.tail(3).mean()
            if len(prev_values) >= 5:
                result_df.loc[horse_data.index[i], f'{mavg_feature}_{target_col}_mavg_5'] = prev_values.tail(5).mean()
            if len(prev_values) >= 7:
                result_df.loc[horse_data.index[i], f'{mavg_feature}_{target_col}_mavg_7'] = prev_values.tail(7).mean()
                
    return result_df

def remove_nan_values(arr):
    """
    Remove NaN values from a numpy array
    
    Args:
        arr: Input numpy array that may contain NaN values
        
    Returns:
        Numpy array with NaN values removed
        
    Example:
        [1.0, 2.0, 2.0, 1.0, nan, nan] -> [1.0, 2.0, 2.0, 1.0]
    """
    # Convert to float array first to handle NaN values
    arr = np.array(arr, dtype=np.float64)
    return arr[~np.isnan(arr)]




def calculate_weight_delta(df, horse_id_col, weight_col):
    """
    Calculate weight change between consecutive races for each horse.
    
    Args:
        df: DataFrame containing horse race data
        horse_id_col: Column name containing horse IDs
        weight_col: Column name containing horse weights
        
    Returns:
        Numpy array containing weight deltas between consecutive races.
        For first race of each horse, delta will be NaN.
    """
    # Create array to store deltas, initialize with NaN
    weight_deltas = np.full(len(df), np.nan)
    
    # Get unique horses
    horses = df[horse_id_col].unique()
    
    # Calculate weight delta for each horse
    for horse in horses:
        # Get all races for this horse in chronological order
        horse_mask = df[horse_id_col] == horse
        horse_data = df[horse_mask].copy()
        
        if len(horse_data) > 1:  # Only calculate if horse has multiple races
            # Get weight values
            weights = horse_data[weight_col].values
            
            # Calculate deltas between consecutive races
            deltas = weights[1:] - weights[:-1]
            
            # Store deltas in result array, skipping first race
            horse_indices = horse_data.index[1:]  # Indices for all races except first
            weight_deltas[horse_indices] = deltas
            
    return weight_deltas





def convert_race_dates_and_plot_metrics(df, date_col='race_date'):
    """
    Convert race dates from string format (YYYYMMDD) to datetime and create time series plots
    for winner match, top3 set, and top3 exact means. Only plots dates with values.
    
    Args:
        df: DataFrame containing race data with date and metric columns
        date_col: Name of column containing date strings
    """
    # Convert dates to datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d')
    
    # Remove rows where all metric columns are null
    metrics = ['winner_match_mean', 'top3set_mean', 'top3exact_mean']
    df_clean = df.dropna(subset=metrics, how='all')
    
    # Sort by date
    df_clean = df_clean.sort_values(date_col)
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    fig.tight_layout(pad=5.0)
    
    # Plot winner match mean and rolling average
    valid_winner = df_clean.dropna(subset=['winner_match_mean'])
    ax1.plot(valid_winner[date_col], valid_winner['winner_match_mean'], 
             label='Daily Mean', color='blue')
    
    # Calculate and plot 7-day rolling average
    rolling_mean = valid_winner['winner_match_mean'].rolling(window=7, min_periods=1).mean()
    ax1.plot(valid_winner[date_col], rolling_mean, 
             label='7-day Rolling Average', color='red', linestyle='--')
    
    ax1.set_title('Winner Match Mean Over Time', pad=20)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Winner Match Mean')
    ax1.grid(True)
    ax1.legend()
    
    # Plot top3 set mean
    valid_top3set = df_clean.dropna(subset=['top3set_mean'])
    ax2.plot(valid_top3set[date_col], valid_top3set['top3set_mean'])
    ax2.set_title('Top 3 Set Mean Over Time', pad=20)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Top 3 Set Mean')
    ax2.grid(True)
    
    # Plot top3 exact mean
    valid_top3exact = df_clean.dropna(subset=['top3exact_mean'])
    ax3.plot(valid_top3exact[date_col], valid_top3exact['top3exact_mean'])
    ax3.set_title('Top 3 Exact Mean Over Time', pad=20)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Top 3 Exact Mean')
    ax3.grid(True)
    
    # Rotate x-axis labels for better readability
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.show()
    
    return df_clean


