#!/usr/bin/env python3
"""
Market Direction Predictor - Turnkey Solution
A complete Python application for predicting S&P 500 weekly direction using volatility indices.

Author: Manus AI
Date: October 2025

Usage:
    python market_predictor_turnkey.py --mode [collect|train|predict|backtest|all]
    
Requirements:
    pip install yfinance pandas numpy scikit-learn matplotlib seaborn joblib
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import argparse
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys

warnings.filterwarnings('ignore')

class MarketPredictor:
    """Complete market direction prediction system."""
    
    def __init__(self, data_dir="./market_data", model_dir="./models"):
        """Initialize the predictor with data and model directories."""
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.symbols = self._get_symbols()
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.selected_features = None
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
    
    def _get_symbols(self):
        """Define the symbols for data collection."""
        return {
            # US Market
            '^VIX': 'VIX',
            '^GSPC': 'S&P 500',
            'SPY': 'S&P 500 ETF',
            'VXX': 'VIX ETF',
            'UVXY': 'Ultra VIX Short-Term Futures ETF',
            
            # International Markets
            'EWA': 'Australia ETF (ASX 200 proxy)',
            'FXA': 'Australian Dollar ETF',
            'EWG': 'Germany ETF (Euro STOXX proxy)',
            'FXE': 'Euro ETF',
            'EWC': 'Canada ETF (TSX proxy)',
            'FXC': 'Canadian Dollar ETF',
            'EWH': 'Hong Kong ETF (Hang Seng proxy)',
            'FXY': 'Japanese Yen ETF',
            
            # Currency pairs
            'EURUSD=X': 'EUR/USD',
            'GBPUSD=X': 'GBP/USD',
            'USDJPY=X': 'USD/JPY',
            
            # Commodities
            'GLD': 'Gold ETF',
            'GC=F': 'Gold Futures',
            'UUP': 'US Dollar Index ETF',
            
            # Additional volatility
            'VIXM': 'VIX Mid-Term Futures ETF',
            'VIXY': 'VIX Short-Term Futures ETF'
        }
    
    def collect_data(self, period="2y"):
        """Collect market data using yfinance."""
        print("🔄 Collecting market data...")
        
        all_data = {}
        failed_symbols = []
        
        for symbol, name in self.symbols.items():
            try:
                print(f"  📊 Fetching {symbol} ({name})...")
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                
                if not df.empty:
                    df = df.fillna(method='ffill').fillna(method='bfill')
                    all_data[symbol] = df
                    print(f"    ✅ Got {len(df)} records")
                else:
                    failed_symbols.append(symbol)
                    print(f"    ❌ No data")
                    
            except Exception as e:
                failed_symbols.append(symbol)
                print(f"    ❌ Error: {str(e)}")
        
        if failed_symbols:
            print(f"⚠️  Failed to collect: {failed_symbols}")
        
        # Combine data
        if all_data:
            combined_df = self._combine_datasets(all_data)
            combined_df.to_csv(os.path.join(self.data_dir, "raw_data.csv"))
            print(f"✅ Collected data for {len(all_data)} symbols, saved to {self.data_dir}/raw_data.csv")
            return combined_df
        else:
            print("❌ No data collected!")
            return None
    
    def _combine_datasets(self, data_dict):
        """Combine individual datasets into one DataFrame."""
        combined_data = pd.DataFrame()
        
        for symbol, df in data_dict.items():
            symbol_name = self.symbols[symbol].replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
            df_renamed = df.copy()
            df_renamed.columns = [f"{symbol_name}_{col}" for col in df_renamed.columns]
            
            if combined_data.empty:
                combined_data = df_renamed
            else:
                combined_data = combined_data.join(df_renamed, how='outer')
        
        return combined_data.sort_index().fillna(method='ffill')
    
    def engineer_features(self, df=None):
        """Create technical indicators and features."""
        print("🔧 Engineering features...")
        
        if df is None:
            try:
                df = pd.read_csv(os.path.join(self.data_dir, "raw_data.csv"), index_col=0, parse_dates=True)
            except FileNotFoundError:
                print("❌ Raw data not found. Run data collection first.")
                return None
        
        # Find Close price columns
        close_columns = [col for col in df.columns if 'Close' in col]
        print(f"  🎯 Processing {len(close_columns)} price series...")
        
        result_df = df.copy()
        
        for close_col in close_columns:
            base_name = close_col.replace('_Close', '')
            price_data = df[close_col]
            
            # Skip if all NaN
            if price_data.isnull().all():
                continue
            
            print(f"    📈 Processing {base_name}...")
            
            # Moving Averages
            for window in [5, 10, 20, 50]:
                result_df[f'{base_name}_SMA_{window}'] = price_data.rolling(window=window).mean()
                result_df[f'{base_name}_EMA_{window}'] = price_data.ewm(span=window).mean()
            
            # RSI
            delta = price_data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result_df[f'{base_name}_RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = price_data.ewm(span=12).mean()
            ema26 = price_data.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            result_df[f'{base_name}_MACD'] = macd_line
            result_df[f'{base_name}_MACD_Signal'] = signal_line
            result_df[f'{base_name}_MACD_Histogram'] = macd_line - signal_line
            
            # Bollinger Bands
            sma20 = price_data.rolling(window=20).mean()
            std20 = price_data.rolling(window=20).std()
            result_df[f'{base_name}_BB_Upper'] = sma20 + (std20 * 2)
            result_df[f'{base_name}_BB_Lower'] = sma20 - (std20 * 2)
            result_df[f'{base_name}_BB_Width'] = (sma20 + (std20 * 2)) - (sma20 - (std20 * 2))
            
            # Returns and Volatility
            result_df[f'{base_name}_Returns'] = price_data.pct_change()
            result_df[f'{base_name}_Volatility'] = price_data.pct_change().rolling(window=20).std() * np.sqrt(252)
            
            # Momentum
            result_df[f'{base_name}_Momentum_5'] = price_data.pct_change(periods=5)
            result_df[f'{base_name}_Momentum_10'] = price_data.pct_change(periods=10)
            
            # Lagged features
            for lag in [1, 2, 3]:
                result_df[f'{base_name}_Lag_{lag}'] = price_data.shift(lag)
        
        # Create target variable (S&P 500 direction in 5 days)
        sp500_cols = [col for col in close_columns if 'S&P_500' in col and 'ETF' not in col]
        if not sp500_cols:
            sp500_cols = [col for col in close_columns if 'S&P_500_ETF' in col]
        
        if sp500_cols:
            target_col = sp500_cols[0]
            price_data = df[target_col]
            future_price = price_data.shift(-5)
            result_df['Target'] = (future_price > price_data).astype(int)
            print(f"  🎯 Created target using {target_col}")
        else:
            print("  ⚠️  No S&P 500 column found for target creation")
        
        # Remove last 5 rows (no target)
        result_df = result_df[:-5]
        
        # Save processed features
        processed_path = os.path.join(self.data_dir, "processed_features.csv")
        result_df.to_csv(processed_path)
        print(f"✅ Feature engineering complete. Saved to {processed_path}")
        print(f"   📊 Final shape: {result_df.shape}")
        
        if 'Target' in result_df.columns:
            target_dist = result_df['Target'].value_counts()
            print(f"   🎯 Target distribution: Up={target_dist.get(1, 0)}, Down={target_dist.get(0, 0)}")
        
        return result_df
    
    def train_models(self, df=None):
        """Train machine learning models."""
        print("🤖 Training models...")
        
        if df is None:
            try:
                df = pd.read_csv(os.path.join(self.data_dir, "processed_features.csv"), index_col=0, parse_dates=True)
            except FileNotFoundError:
                print("❌ Processed features not found. Run feature engineering first.")
                return None
        
        if 'Target' not in df.columns:
            print("❌ Target column not found!")
            return None
        
        # Prepare data
        y = df['Target']
        X = df.drop('Target', axis=1)
        
        # Remove high-NaN columns
        nan_ratio = X.isnull().sum() / len(X)
        valid_columns = nan_ratio[nan_ratio < 0.3].index
        X = X[valid_columns]
        
        # Handle remaining NaN
        X = X.fillna(method='ffill').fillna(method='bfill')
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in ['float64', 'int64']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0)
        
        # Remove rows with NaN targets
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
        
        print(f"  📊 Training data shape: {X.shape}")
        print(f"  🎯 Target distribution: {y.value_counts().to_dict()}")
        
        # Feature selection
        print("  🔍 Selecting features...")
        n_features = min(100, len(X.columns))
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        self.selected_features = X.columns[selector.get_support()].tolist()
        self.feature_selector = selector
        
        X_selected = pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
        print(f"    ✅ Selected {len(self.selected_features)} features")
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Train Random Forest
        print("  🌲 Training Random Forest...")
        rf_params = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
            'min_samples_split': [5, 10]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            rf_params, cv=tscv, scoring='accuracy', n_jobs=-1
        )
        rf_grid.fit(X_selected, y)
        self.models['RandomForest'] = rf_grid.best_estimator_
        self.scalers['RandomForest'] = None
        
        print(f"    ✅ RF Best Score: {rf_grid.best_score_:.4f}")
        
        # Train Logistic Regression
        print("  📊 Training Logistic Regression...")
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index
        )
        
        lr_params = {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }
        
        lr_grid = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=1000),
            lr_params, cv=tscv, scoring='accuracy', n_jobs=-1
        )
        lr_grid.fit(X_scaled, y)
        self.models['LogisticRegression'] = lr_grid.best_estimator_
        self.scalers['LogisticRegression'] = scaler
        
        print(f"    ✅ LR Best Score: {lr_grid.best_score_:.4f}")
        
        # Evaluate on test set
        print("  📈 Evaluating models...")
        split_idx = int(len(X_selected) * 0.8)
        X_train, X_test = X_selected.iloc[:split_idx], X_selected.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        results = {}
        for model_name, model in self.models.items():
            if self.scalers[model_name] is not None:
                scaler_final = StandardScaler()
                X_train_final = scaler_final.fit_transform(X_train)
                X_test_final = scaler_final.transform(X_test)
                self.scalers[model_name] = scaler_final
            else:
                X_train_final = X_train
                X_test_final = X_test
            
            model.fit(X_train_final, y_train)
            y_pred = model.predict(X_test_final)
            y_pred_proba = model.predict_proba(X_test_final)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[model_name] = {
                'accuracy': accuracy,
                'auc_score': auc_score
            }
            
            print(f"    📊 {model_name}: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}")
        
        # Save models
        self._save_models()
        print("✅ Model training complete!")
        
        return results
    
    def _save_models(self):
        """Save trained models and preprocessing objects."""
        for model_name, model in self.models.items():
            joblib.dump(model, os.path.join(self.model_dir, f'{model_name}_model.pkl'))
        
        for model_name, scaler in self.scalers.items():
            if scaler is not None:
                joblib.dump(scaler, os.path.join(self.model_dir, f'{model_name}_scaler.pkl'))
        
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, os.path.join(self.model_dir, 'feature_selector.pkl'))
        
        if self.selected_features is not None:
            joblib.dump(self.selected_features, os.path.join(self.model_dir, 'selected_features.pkl'))
    
    def _load_models(self):
        """Load trained models and preprocessing objects."""
        try:
            for model_name in ['RandomForest', 'LogisticRegression']:
                model_path = os.path.join(self.model_dir, f'{model_name}_model.pkl')
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                
                scaler_path = os.path.join(self.model_dir, f'{model_name}_scaler.pkl')
                if os.path.exists(scaler_path):
                    self.scalers[model_name] = joblib.load(scaler_path)
                else:
                    self.scalers[model_name] = None
            
            selector_path = os.path.join(self.model_dir, 'feature_selector.pkl')
            if os.path.exists(selector_path):
                self.feature_selector = joblib.load(selector_path)
            
            features_path = os.path.join(self.model_dir, 'selected_features.pkl')
            if os.path.exists(features_path):
                self.selected_features = joblib.load(features_path)
            
            return len(self.models) > 0
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict_latest(self):
        """Make prediction for the latest available data."""
        print("🔮 Making latest prediction...")
        
        if not self._load_models():
            print("❌ No trained models found. Run training first.")
            return None
        
        try:
            df = pd.read_csv(os.path.join(self.data_dir, "processed_features.csv"), index_col=0, parse_dates=True)
        except FileNotFoundError:
            print("❌ Processed features not found.")
            return None
        
        if 'Target' in df.columns:
            df = df.drop('Target', axis=1)
        
        # Get latest data
        latest_data = df.tail(1)
        
        # Preprocess
        latest_processed = self._preprocess_for_prediction(latest_data)
        if latest_processed is None:
            return None
        
        # Make predictions
        results = {'date': latest_data.index[0]}
        
        for model_name, model in self.models.items():
            try:
                if self.scalers[model_name] is not None:
                    X_scaled = self.scalers[model_name].transform(latest_processed)
                else:
                    X_scaled = latest_processed.values
                
                prediction = model.predict(X_scaled)[0]
                probability = model.predict_proba(X_scaled)[0]
                
                results[model_name] = {
                    'prediction': 'UP' if prediction == 1 else 'DOWN',
                    'probability_up': probability[1],
                    'probability_down': probability[0],
                    'confidence': max(probability)
                }
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        # Display results
        print(f"📅 Prediction for {results['date'].strftime('%Y-%m-%d')}:")
        for model_name, result in results.items():
            if model_name != 'date':
                if 'error' not in result:
                    print(f"  🤖 {model_name}: {result['prediction']} "
                          f"(Confidence: {result['confidence']:.1%})")
                else:
                    print(f"  ❌ {model_name}: {result['error']}")
        
        return results
    
    def _preprocess_for_prediction(self, X):
        """Preprocess features for prediction."""
        # Handle NaN
        X_processed = X.fillna(method='ffill').fillna(method='bfill')
        for col in X_processed.columns:
            if X_processed[col].isnull().any():
                if X_processed[col].dtype in ['float64', 'int64']:
                    X_processed[col] = X_processed[col].fillna(X_processed[col].median())
                else:
                    X_processed[col] = X_processed[col].fillna(0)
        
        # Apply feature selection
        if self.feature_selector is not None:
            try:
                X_selected = self.feature_selector.transform(X_processed)
                return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
            except Exception as e:
                print(f"⚠️  Feature selection failed: {e}")
                # Fallback
                available_features = [f for f in self.selected_features if f in X_processed.columns]
                missing_features = [f for f in self.selected_features if f not in X_processed.columns]
                
                if missing_features:
                    for feature in missing_features:
                        X_processed[feature] = 0.0
                
                return X_processed[self.selected_features]
        
        return X_processed
    
    def backtest(self, days=180):
        """Perform backtesting."""
        print(f"📊 Running backtest for last {days} days...")
        
        if not self._load_models():
            print("❌ No trained models found. Run training first.")
            return None
        
        try:
            df = pd.read_csv(os.path.join(self.data_dir, "processed_features.csv"), index_col=0, parse_dates=True)
        except FileNotFoundError:
            print("❌ Processed features not found.")
            return None
        
        if 'Target' not in df.columns:
            print("❌ Target column not found.")
            return None
        
        # Get recent data
        end_date = df.index.max()
        start_date = end_date - timedelta(days=days)
        recent_data = df[df.index >= start_date].copy()
        
        y_true = recent_data['Target']
        X = recent_data.drop('Target', axis=1)
        
        # Remove NaN targets
        mask = ~y_true.isnull()
        X = X[mask]
        y_true = y_true[mask]
        
        print(f"  📊 Backtesting on {len(X)} samples")
        print(f"  📅 Period: {X.index.min().strftime('%Y-%m-%d')} to {X.index.max().strftime('%Y-%m-%d')}")
        
        results = {}
        
        for model_name, model in self.models.items():
            try:
                # Preprocess
                X_processed = self._preprocess_for_prediction(X)
                if X_processed is None:
                    continue
                
                if self.scalers[model_name] is not None:
                    X_scaled = self.scalers[model_name].transform(X_processed)
                else:
                    X_scaled = X_processed.values
                
                # Predictions
                y_pred = model.predict(X_scaled)
                y_pred_proba = model.predict_proba(X_scaled)[:, 1]
                
                # Metrics
                accuracy = accuracy_score(y_true, y_pred)
                auc_score = roc_auc_score(y_true, y_pred_proba)
                
                # Simulated returns (1% for correct, -1% for incorrect)
                returns = np.where(y_pred == y_true, 0.01, -0.01)
                cumulative_returns = np.cumprod(1 + returns) - 1
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'auc_score': auc_score,
                    'total_return': cumulative_returns[-1],
                    'max_drawdown': np.min(cumulative_returns),
                    'predictions': y_pred,
                    'returns': returns,
                    'cumulative_returns': cumulative_returns,
                    'dates': X.index
                }
                
                print(f"  🤖 {model_name}:")
                print(f"    📈 Accuracy: {accuracy:.1%}")
                print(f"    📊 AUC: {auc_score:.3f}")
                print(f"    💰 Total Return: {cumulative_returns[-1]:+.1%}")
                print(f"    📉 Max Drawdown: {np.min(cumulative_returns):+.1%}")
                
            except Exception as e:
                print(f"  ❌ {model_name} failed: {e}")
        
        # Create visualization
        if results:
            self._create_backtest_chart(results, y_true)
        
        return results
    
    def _create_backtest_chart(self, results, y_true):
        """Create backtest visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cumulative returns
        for model_name, result in results.items():
            axes[0, 0].plot(result['dates'], result['cumulative_returns'], 
                           label=model_name, linewidth=2)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy comparison
        models = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in models]
        bars = axes[0, 1].bar(models, accuracies)
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim(0, 1)
        
        for bar, acc in zip(bars, accuracies):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.1%}', ha='center', va='bottom')
        
        # Prediction distribution
        best_model = max(models, key=lambda x: results[x]['accuracy'])
        y_pred = results[best_model]['predictions']
        
        true_dist = y_true.value_counts()
        pred_dist = pd.Series(y_pred).value_counts()
        
        x = ['Down', 'Up']
        x_pos = np.arange(len(x))
        width = 0.35
        
        axes[1, 0].bar(x_pos - width/2, [true_dist.get(0, 0), true_dist.get(1, 0)], 
                      width, label='Actual', alpha=0.7)
        axes[1, 0].bar(x_pos + width/2, [pred_dist.get(0, 0), pred_dist.get(1, 0)], 
                      width, label='Predicted', alpha=0.7)
        axes[1, 0].set_title(f'Predictions vs Actual ({best_model})')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(x)
        axes[1, 0].legend()
        
        # Performance metrics
        metrics = ['Accuracy', 'AUC', 'Total Return']
        best_results = results[best_model]
        values = [best_results['accuracy'], best_results['auc_score'], 
                 (best_results['total_return'] + 1)]  # Normalize return
        
        axes[1, 1].bar(metrics, values)
        axes[1, 1].set_title(f'Performance Metrics ({best_model})')
        axes[1, 1].set_ylim(0, max(values) * 1.1)
        
        for i, v in enumerate(values):
            if i == 2:  # Return
                axes[1, 1].text(i, v + 0.01, f'{(v-1):+.1%}', ha='center', va='bottom')
            else:
                axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        chart_path = os.path.join(self.data_dir, "backtest_results.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Backtest chart saved to {chart_path}")
    
    def run_full_pipeline(self):
        """Run the complete pipeline."""
        print("🚀 Running full market prediction pipeline...")
        print("=" * 60)
        
        # Step 1: Data Collection
        df = self.collect_data()
        if df is None:
            return
        
        # Step 2: Feature Engineering
        df = self.engineer_features(df)
        if df is None:
            return
        
        # Step 3: Model Training
        results = self.train_models(df)
        if results is None:
            return
        
        # Step 4: Latest Prediction
        self.predict_latest()
        
        # Step 5: Backtesting
        self.backtest()
        
        print("\n🎉 Pipeline complete!")
        print("📁 Check the following directories:")
        print(f"   📊 Data: {self.data_dir}")
        print(f"   🤖 Models: {self.model_dir}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Market Direction Predictor")
    parser.add_argument("--mode", choices=["collect", "train", "predict", "backtest", "all"], 
                       default="all", help="Mode to run")
    parser.add_argument("--data-dir", default="./market_data", help="Data directory")
    parser.add_argument("--model-dir", default="./models", help="Model directory")
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        import yfinance
        import sklearn
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install yfinance pandas numpy scikit-learn matplotlib seaborn joblib")
        sys.exit(1)
    
    # Initialize predictor
    predictor = MarketPredictor(data_dir=args.data_dir, model_dir=args.model_dir)
    
    print("📈 Market Direction Predictor")
    print("=" * 40)
    
    if args.mode == "all":
        predictor.run_full_pipeline()
    elif args.mode == "collect":
        predictor.collect_data()
    elif args.mode == "train":
        predictor.train_models()
    elif args.mode == "predict":
        predictor.predict_latest()
    elif args.mode == "backtest":
        predictor.backtest()


if __name__ == "__main__":
    main()
