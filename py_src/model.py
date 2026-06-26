import lightgbm as lgb

FEATURES = ['micro_price', 'obi', 'obi_5', 'obi_10', 'spread', 'rolling_vol']

# Primary targets: MID-price forward returns -- the tradable mid.
TARGETS = ['ret_10c', 'ret_50c', 'ret_200c']

# Mechanical baseline: micro-price forward returns. The micro-price is pulled
# toward the heavier queue, so it co-moves with OBI almost algebraically; its
# IC (~0.33) overstates the tradable signal and is reported only for contrast.
MICRO_TARGETS = ['ret_micro_10c', 'ret_micro_50c', 'ret_micro_200c']

# Longest forward-return horizon, in ticks. Rows at the end of the training set
# have labels that look up to this many ticks into the future — i.e. into the
# test set — so we purge an embargo gap of this size between train and test.
EMBARGO_TICKS = 200


def purged_split(df, train_frac=0.7, embargo=EMBARGO_TICKS):
    """Chronological train/test split with an embargo gap.

    The last `embargo` rows before the split boundary are dropped from the
    training set so that no training label overlaps the test period.
    """
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:max(0, split_idx - embargo)]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def train_models(df, features=FEATURES, targets=TARGETS, embargo=EMBARGO_TICKS):
    # Sort chronologically just in case
    df = df.sort_values('time').reset_index(drop=True)

    train_df, test_df = purged_split(df, embargo=embargo)

    X_train = train_df[features]

    params = {
        'objective': 'regression',
        'learning_rate': 0.05,
        'num_leaves': 15,
        'feature_fraction': 0.7,
        'min_data_in_leaf': 200,
        'verbose': -1,
        'seed': 42
    }

    models = {}

    for target in targets:
        y_train = train_df[target]

        train_data = lgb.Dataset(X_train, label=y_train)

        print(f"Training model for {target}...")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100
        )

        models[target] = model

    return models, train_df, test_df
