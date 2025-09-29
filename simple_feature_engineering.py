"""
Windows-Only Feature Engineering
Pure pandas for maximum compatibility
"""

import pandas as pd
import numpy as np
from config import CAT_FEATURES, NAN_VALUE


def simple_feature_engineering(df):
    """
    Simplified but effective feature engineering
    Based on proven 0.793 approach with enhancements
    """
    print("Applying simplified feature engineering...")

    # Convert customer_ID consistently
    def safe_hex_convert(x):
        try:
            # Take last 16 characters and convert to int
            hex_part = str(x)[-16:]
            return int(hex_part, 16)
        except:
            # Fallback: use hash of full string
            return hash(str(x)) % (2**63)

    if df['customer_ID'].dtype == 'object':
        df = df.copy()
        df['customer_ID'] = df['customer_ID'].apply(safe_hex_convert)

    all_cols = [c for c in list(df.columns) if c not in ['customer_ID', 'S_2']]
    cat_features = CAT_FEATURES
    num_features = [col for col in all_cols if col not in cat_features]

    print(f"Processing {len(num_features)} numerical and {len(cat_features)} categorical features")

    feature_dfs = []

    # 1. BASIC STATISTICAL AGGREGATIONS (Enhanced)
    print("  -> Basic statistical features...")
    try:
        num_agg = df.groupby("customer_ID")[num_features].agg([
            'mean', 'std', 'min', 'max', 'last', 'first'
        ])
        num_agg.columns = ['_'.join(x) for x in num_agg.columns]
        feature_dfs.append(num_agg)
        print(f"    Added {num_agg.shape[1]} basic numerical features")
    except Exception as e:
        print(f"    Basic aggregation failed: {e}")

    # 2. CATEGORICAL FEATURES
    print("  -> Categorical features...")
    try:
        cat_agg = df.groupby("customer_ID")[cat_features].agg([
            'count', 'last', 'nunique', 'first'
        ])
        cat_agg.columns = ['_'.join(x) for x in cat_agg.columns]
        feature_dfs.append(cat_agg)
        print(f"    Added {cat_agg.shape[1]} categorical features")
    except Exception as e:
        print(f"    Categorical aggregation failed: {e}")

    # 3. ENHANCED STATISTICAL FEATURES
    print("  -> Enhanced statistical features...")
    try:
        # More statistical measures
        enhanced_agg = df.groupby("customer_ID")[num_features[:50]].agg([
            'median', 'sum'
        ])
        enhanced_agg.columns = ['_'.join(x) for x in enhanced_agg.columns]
        feature_dfs.append(enhanced_agg)
        print(f"    Added {enhanced_agg.shape[1]} enhanced statistical features")
    except Exception as e:
        print(f"    Enhanced aggregation failed: {e}")

    # 4. SIMPLE TEMPORAL FEATURES
    print("  -> Simple temporal features...")
    try:
        # Sort for temporal analysis
        df_sorted = df.sort_values(['customer_ID', 'S_2'])

        temporal_features = []
        for feature in num_features[:30]:  # Limit for performance
            try:
                # Simple temporal aggregations
                temp_data = df_sorted.groupby('customer_ID')[feature].agg([
                    lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0,  # Change
                    lambda x: len(x),  # Count of records
                ]).reset_index()
                temp_data.columns = ['customer_ID', f'{feature}_change', f'{feature}_count']
                temporal_features.append(temp_data.set_index('customer_ID'))
            except:
                continue

        if temporal_features:
            temporal_df = pd.concat(temporal_features, axis=1)
            feature_dfs.append(temporal_df)
            print(f"    Added temporal features for {len(temporal_features)} variables")

    except Exception as e:
        print(f"    Temporal features failed: {e}")

    # 5. SIMPLE INTERACTION FEATURES
    print("  -> Simple interaction features...")
    try:
        # Group features by prefix
        payment_features = [f for f in num_features if f.startswith('P_')]
        balance_features = [f for f in num_features if f.startswith('B_')]
        delinquency_features = [f for f in num_features if f.startswith('D_')]
        spend_features = [f for f in num_features if f.startswith('S_')]

        interaction_df = pd.DataFrame(index=df['customer_ID'].unique())

        # Simple ratios
        if payment_features and balance_features:
            payment_mean = df.groupby('customer_ID')[payment_features].mean().sum(axis=1)
            balance_mean = df.groupby('customer_ID')[balance_features].mean().sum(axis=1)
            interaction_df['payment_balance_ratio'] = payment_mean / (balance_mean + 1e-8)

        if spend_features and balance_features:
            spend_mean = df.groupby('customer_ID')[spend_features].mean().sum(axis=1)
            balance_mean = df.groupby('customer_ID')[balance_features].mean().sum(axis=1)
            interaction_df['spend_balance_ratio'] = spend_mean / (balance_mean + 1e-8)

        if delinquency_features:
            delinq_mean = df.groupby('customer_ID')[delinquency_features].mean()
            interaction_df['delinq_concentration'] = delinq_mean.max(axis=1) / (delinq_mean.mean(axis=1) + 1e-8)

        # Add to features (pure pandas)
        if len(interaction_df.columns) > 0:
            interaction_df = interaction_df.reset_index()
            interaction_df = interaction_df.rename(columns={'index': 'customer_ID'})
            feature_dfs.append(interaction_df.set_index('customer_ID'))
            print(f"    Added {interaction_df.shape[1]-1} interaction features")

    except Exception as e:
        print(f"    Interaction features failed: {e}")

    # COMBINE ALL FEATURES (pure pandas)
    if feature_dfs:
        print("  -> Combining all features...")
        result = feature_dfs[0]
        for feat_df in feature_dfs[1:]:
            result = pd.concat([result, feat_df], axis=1)

        print(f"Final feature set: {result.shape[1]} features for {result.shape[0]} customers")
        return result.reset_index()
    else:
        print("❌ No features were generated successfully!")
        return pd.DataFrame({'customer_ID': df['customer_ID'].unique()})


def process_and_feature_engineer(df):
    """
    Main feature engineering function - uses simplified approach
    """
    return simple_feature_engineering(df)