#!/usr/bin/env python3
"""
SPX Trading System - FINAL WORKING VERSION
Complete trading system with configurable parameters, SPX price charts, and trade analysis.
Includes strict forward-looking bias prevention and comprehensive error handling.

Author: Manus AI
Date: October 2025
"""

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# ============================================================================

# Data Collection Parameters
DATA_START_DATE = "2020-01-01"      # Start date for data collection
DATA_END_DATE = "2024-12-31"        # End date for data collection

# Training Parameters
TRAIN_START_DATE = "2020-01-01"     # Training data start
TRAIN_END_DATE = "2023-12-31"       # Training data end (must be before backtest)
PREDICTION_HORIZON = 5               # Days ahead to predict (1 week = 5 trading days)

# Backtesting Parameters
BACKTEST_START_DATE = "2024-01-01"  # Backtest start (AFTER training end)
BACKTEST_END_DATE = "2024-12-31"    # Backtest end
INITIAL_CAPITAL = 100000             # Starting capital for backtesting
CONFIDENCE_THRESHOLD = 0.55          # Minimum confidence to enter trades
POSITION_SIZE = 0.02                 # Risk 2% of capital per trade

# Model Parameters
N_FEATURES = 30                      # Number of features to select
RANDOM_STATE = 42                    # For reproducibility

# Chart Parameters
SAVE_CHARTS = True                   # Save charts to files

# ============================================================================
# IMPORTS AND SETUP
# ============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class SPXTradingSystem:
    """Complete SPX trading system with forward-looking bias prevention."""
    
    def __init__(self):
        """Initialize the trading system."""
        self.data_dir = "./spx_data"
        self.model_dir = "./spx_models"
        self.chart_dir = "./spx_charts"
        
        # Create directories
        for directory in [self.data_dir, self.model_dir, self.chart_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize components
        self.symbols = self._get_symbols()
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.selected_features = None
        self.raw_data = None
        self.processed_data = None
        self.spx_prices = None
        
        print("SPX Trading System Initialized")
        print(f"Data Period: {DATA_START_DATE} to {DATA_END_DATE}")
        print(f"Training Period: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
        print(f"Backtest Period: {BACKTEST_START_DATE} to {BACKTEST_END_DATE}")
        print(f"Prediction Horizon: {PREDICTION_HORIZON} days")
        print("=" * 60)
    
    def _get_symbols(self):
        """Define symbols for data collection."""
        return {
            '^GSPC': 'SPX',
            'SPY': 'SPY',
            '^VIX': 'VIX',
            'VXX': 'VXX',
            'UVXY': 'UVXY',
            'EWA': 'EWA',
            'EWG': 'EWG',
            'FXA': 'FXA',
            'FXE': 'FXE',
            'FXY': 'FXY',
            'UUP': 'UUP',
            'EURUSD=X': 'EURUSD',
            'GLD': 'GLD'
        }
    
    def collect_data(self):
        """Collect market data."""
        print("Collecting market data...")
        
        all_data = {}
        
        for symbol, name in self.symbols.items():
            try:
                print(f"  Fetching {symbol}...")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=DATA_START_DATE, end=DATA_END_DATE)
                
                if not df.empty:
                    df = df.fillna(method='ffill').fillna(method='bfill')
                    all_data[symbol] = df
                    print(f"    Got {len(df)} records")
                else:
                    print(f"    No data")
                    
            except Exception as e:
                print(f"    Error: {str(e)}")
        
        if all_data:
            self.raw_data = self._combine_datasets(all_data)
            
            # Extract SPX prices
            if '^GSPC' in all_data:
                self.spx_prices = all_data['^GSPC']['Close'].copy()
            elif 'SPY' in all_data:
                self.spx_prices = all_data['SPY']['Close'].copy()
            
            raw_path = os.path.join(self.data_dir, "raw_data.csv")
            self.raw_data.to_csv(raw_path)
            
            print(f"Data collection complete! Shape: {self.raw_data.shape}")
            return self.raw_data
        else:
            print("No data collected!")
            return None
    
    def _combine_datasets(self, data_dict):
        """Combine datasets."""
        combined_data = pd.DataFrame()
        
        for symbol, df in data_dict.items():
            symbol_name = self.symbols[symbol]
            df_renamed = df.copy()
            df_renamed.columns = [f"{symbol_name}_{col}" for col in df_renamed.columns]
            
            if combined_data.empty:
                combined_data = df_renamed
            else:
                combined_data = combined_data.join(df_renamed, how='outer')
        
        return combined_data.sort_index().fillna(method='ffill')
    
    def engineer_features(self):
        """Create technical indicators."""
        print("Engineering features...")
        
        if self.raw_data is None:
            try:
                self.raw_data = pd.read_csv(os.path.join(self.data_dir, "raw_data.csv"), 
                                          index_col=0, parse_dates=True)
            except FileNotFoundError:
                print("Raw data not found.")
                return None
        
        close_columns = [col for col in self.raw_data.columns if 'Close' in col]
        print(f"Processing {len(close_columns)} price series...")
        
        result_df = self.raw_data.copy()
        
        for close_col in close_columns:
            base_name = close_col.replace('_Close', '')
            price_data = self.raw_data[close_col]
            
            if price_data.isnull().all():
                continue
            
            print(f"  Processing {base_name}...")
            
            # Moving averages
            for window in [5, 10, 20]:
                result_df[f'{base_name}_SMA_{window}'] = price_data.rolling(window=window).mean()
                result_df[f'{base_name}_EMA_{window}'] = price_data.ewm(span=window).mean()
            
            # Returns
            result_df[f'{base_name}_Return_1d'] = price_data.pct_change()
            result_df[f'{base_name}_Return_5d'] = price_data.pct_change(periods=5)
            
            # RSI
            delta = price_data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)  # Avoid division by zero
            result_df[f'{base_name}_RSI'] = 100 - (100 / (1 + rs))
            
            # Volatility
            result_df[f'{base_name}_Vol'] = price_data.pct_change().rolling(window=20).std()
        
        # Create target (SPX direction in PREDICTION_HORIZON days)
        spx_cols = [col for col in close_columns if 'SPX' in col]
        if not spx_cols:
            spx_cols = [col for col in close_columns if 'SPY' in col]
        
        if spx_cols:
            target_col = spx_cols[0]
            current_price = self.raw_data[target_col]
            future_price = current_price.shift(-PREDICTION_HORIZON)
            result_df['Target'] = (future_price > current_price).astype(int)
            print(f"Created target using {target_col}")
        
        # Remove last PREDICTION_HORIZON rows
        result_df = result_df[:-PREDICTION_HORIZON]
        
        # Clean data
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        result_df = result_df.fillna(method='ffill').fillna(method='bfill')
        
        self.processed_data = result_df
        processed_path = os.path.join(self.data_dir, "processed_features.csv")
        result_df.to_csv(processed_path)
        
        print(f"Feature engineering complete! Shape: {result_df.shape}")
        
        if 'Target' in result_df.columns:
            target_dist = result_df['Target'].value_counts()
            print(f"Target distribution: UP={target_dist.get(1, 0)}, DOWN={target_dist.get(0, 0)}")
        
        return result_df
    
    def train_models(self):
        """Train models."""
        print("Training models...")
        
        if self.processed_data is None:
            try:
                self.processed_data = pd.read_csv(os.path.join(self.data_dir, "processed_features.csv"), 
                                                index_col=0, parse_dates=True)
            except FileNotFoundError:
                print("Processed features not found.")
                return None
        
        # Use only training period
        train_data = self.processed_data[(self.processed_data.index >= TRAIN_START_DATE) & 
                                       (self.processed_data.index <= TRAIN_END_DATE)].copy()
        
        print(f"Training period: {train_data.index.min().date()} to {train_data.index.max().date()}")
        print(f"Training samples: {len(train_data)}")
        
        if 'Target' not in train_data.columns:
            print("Target column not found!")
            return None
        
        y = train_data['Target']
        X = train_data.drop('Target', axis=1)
        
        # Remove high-NaN columns
        nan_ratio = X.isnull().sum() / len(X)
        valid_columns = nan_ratio[nan_ratio < 0.5].index
        X = X[valid_columns]
        
        # Handle NaN
        X = X.fillna(method='ffill').fillna(method='bfill')
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())
        
        # Remove NaN targets
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
        
        print(f"Final training shape: {X.shape}")
        
        # Feature selection
        try:
            n_features = min(N_FEATURES, len(X.columns))
            selector = SelectKBest(score_func=f_classif, k=n_features)
            X_selected = selector.fit_transform(X, y)
            self.selected_features = X.columns[selector.get_support()].tolist()
            self.feature_selector = selector
            X_selected = pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
            print(f"Selected {len(self.selected_features)} features")
        except:
            print("Feature selection failed, using all features")
            X_selected = X
            self.selected_features = X.columns.tolist()
        
        # Train Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_selected, y)
        self.models['RandomForest'] = rf
        self.scalers['RandomForest'] = None
        print(f"  RF Score: {rf.score(X_selected, y):.4f}")
        
        # Train Logistic Regression
        print("Training Logistic Regression...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        lr.fit(X_scaled, y)
        self.models['LogisticRegression'] = lr
        self.scalers['LogisticRegression'] = scaler
        print(f"  LR Score: {lr.score(X_scaled, y):.4f}")
        
        # Save models
        self._save_models()
        print("Model training complete!")
        return True
    
    def _save_models(self):
        """Save models."""
        for model_name, model in self.models.items():
            joblib.dump(model, os.path.join(self.model_dir, f'{model_name}_model.pkl'))
        
        for model_name, scaler in self.scalers.items():
            if scaler is not None:
                joblib.dump(scaler, os.path.join(self.model_dir, f'{model_name}_scaler.pkl'))
        
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, os.path.join(self.model_dir, 'feature_selector.pkl'))
            joblib.dump(self.selected_features, os.path.join(self.model_dir, 'selected_features.pkl'))
    
    def _load_models(self):
        """Load models."""
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
            print(f"Error loading models: {e}")
            return False
    
    def backtest_with_charts(self):
        """Run backtest with charts."""
        print("Running backtest...")
        
        if not self._load_models():
            print("No models found.")
            return None
        
        if self.processed_data is None:
            try:
                self.processed_data = pd.read_csv(os.path.join(self.data_dir, "processed_features.csv"), 
                                                index_col=0, parse_dates=True)
            except FileNotFoundError:
                print("Processed features not found.")
                return None
        
        # Use only backtest period
        backtest_data = self.processed_data[(self.processed_data.index >= BACKTEST_START_DATE) & 
                                          (self.processed_data.index <= BACKTEST_END_DATE)].copy()
        
        print(f"Backtest period: {backtest_data.index.min().date()} to {backtest_data.index.max().date()}")
        print(f"Backtest samples: {len(backtest_data)}")
        
        if len(backtest_data) == 0:
            print("No backtest data!")
            return None
        
        # Get SPX prices for backtest period
        spx_backtest = self.spx_prices.loc[backtest_data.index].copy()
        
        y_true = backtest_data['Target']
        X = backtest_data.drop('Target', axis=1)
        
        # Remove NaN targets
        mask = ~y_true.isnull()
        X = X[mask]
        y_true = y_true[mask]
        spx_backtest = spx_backtest[mask]
        
        print(f"Valid samples: {len(X)}")
        
        results = {}
        all_trades = []
        
        for model_name, model in self.models.items():
            print(f"\nBacktesting {model_name}...")
            
            try:
                # Preprocess
                X_processed = self._preprocess_for_prediction(X)
                if X_processed is None:
                    continue
                
                # Scale if needed
                if self.scalers[model_name] is not None:
                    X_scaled = self.scalers[model_name].transform(X_processed)
                else:
                    X_scaled = X_processed.values
                
                # Predictions
                y_pred = model.predict(X_scaled)
                y_pred_proba = model.predict_proba(X_scaled)[:, 1]
                confidence = np.maximum(y_pred_proba, 1 - y_pred_proba)
                
                # Trading simulation
                trades = self._simulate_trading(X.index, spx_backtest, y_pred, confidence, y_true, model_name)
                all_trades.extend(trades)
                
                # Metrics
                accuracy = accuracy_score(y_true, y_pred)
                auc_score = roc_auc_score(y_true, y_pred_proba)
                
                if trades:
                    total_return = sum(t['pnl_pct'] for t in trades)
                    win_rate = sum(1 for t in trades if t['pnl_pct'] > 0) / len(trades)
                    avg_trade = np.mean([t['pnl_pct'] for t in trades])
                else:
                    total_return = win_rate = avg_trade = 0
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'auc_score': auc_score,
                    'trades': trades,
                    'total_return': total_return,
                    'win_rate': win_rate,
                    'avg_trade': avg_trade,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'confidence': confidence,
                    'dates': X.index
                }
                
                print(f"  Accuracy: {accuracy:.1%}")
                print(f"  AUC: {auc_score:.3f}")
                print(f"  Trades: {len(trades)}")
                print(f"  Win Rate: {win_rate:.1%}")
                print(f"  Total Return: {total_return:+.1f}%")
                
            except Exception as e:
                print(f"  {model_name} failed: {e}")
        
        # Create charts
        if results:
            self._create_charts(results, spx_backtest, y_true)
            self._analyze_trades(all_trades)
        
        return results
    
    def _simulate_trading(self, dates, spx_prices, predictions, confidence, actual, model_name):
        """Simulate trading."""
        trades = []
        capital = INITIAL_CAPITAL
        
        for i, (date, pred, conf, actual_outcome) in enumerate(zip(dates, predictions, confidence, actual)):
            if conf < CONFIDENCE_THRESHOLD:
                continue
            
            if date not in spx_prices.index:
                continue
            
            entry_price = spx_prices.loc[date]
            
            # Find exit date
            future_dates = spx_prices.index[spx_prices.index > date]
            if len(future_dates) < PREDICTION_HORIZON:
                continue
            
            exit_date = future_dates[min(PREDICTION_HORIZON-1, len(future_dates)-1)]
            exit_price = spx_prices.loc[exit_date]
            
            # Calculate P&L
            if pred == 1:  # Long
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                position_type = 'LONG'
            else:  # Short
                pnl_pct = (entry_price - exit_price) / entry_price * 100
                position_type = 'SHORT'
            
            capital += (capital * POSITION_SIZE * pnl_pct / 100)
            
            trades.append({
                'model': model_name,
                'entry_date': date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': position_type,
                'prediction': pred,
                'confidence': conf,
                'actual': actual_outcome,
                'correct': pred == actual_outcome,
                'pnl_pct': pnl_pct,
                'capital': capital
            })
        
        return trades
    
    def _preprocess_for_prediction(self, X):
        """Preprocess features."""
        X_processed = X.copy()
        X_processed = X_processed.replace([np.inf, -np.inf], np.nan)
        X_processed = X_processed.fillna(method='ffill').fillna(method='bfill')
        
        for col in X_processed.columns:
            if X_processed[col].isnull().any():
                X_processed[col] = X_processed[col].fillna(X_processed[col].median())
        
        # Feature selection
        if self.feature_selector is not None and self.selected_features is not None:
            try:
                missing_features = [f for f in self.selected_features if f not in X_processed.columns]
                for feature in missing_features:
                    X_processed[feature] = 0.0
                
                X_selected = X_processed[self.selected_features]
                return X_selected
            except:
                return X_processed
        
        return X_processed
    
    def _create_charts(self, results, spx_prices, y_true):
        """Create trading charts."""
        print("Creating charts...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('SPX Trading System Analysis', fontsize=16)
            
            # 1. SPX Price with signals
            ax1 = axes[0, 0]
            ax1.plot(spx_prices.index, spx_prices.values, 'k-', linewidth=1, label='SPX Price')
            
            if results:
                best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
                best_results = results[best_model]
                
                for trade in best_results['trades'][:20]:  # Show first 20 trades
                    color = 'green' if trade['position'] == 'LONG' else 'red'
                    marker = '^' if trade['position'] == 'LONG' else 'v'
                    ax1.scatter(trade['entry_date'], trade['entry_price'], 
                               color=color, marker=marker, s=30, alpha=0.7)
                
                ax1.set_title(f'SPX with Trade Signals ({best_model})')
            
            ax1.set_ylabel('SPX Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Cumulative returns
            ax2 = axes[0, 1]
            
            for model_name, result in results.items():
                if result['trades']:
                    trade_dates = [t['entry_date'] for t in result['trades']]
                    returns = [t['pnl_pct'] for t in result['trades']]
                    cumulative_returns = np.cumsum(returns)
                    
                    ax2.plot(trade_dates, cumulative_returns, marker='o', markersize=2, 
                            linewidth=2, label=f'{model_name} ({result["total_return"]:+.1f}%)')
            
            # Buy and hold
            buy_hold_return = (spx_prices.iloc[-1] - spx_prices.iloc[0]) / spx_prices.iloc[0] * 100
            ax2.axhline(y=buy_hold_return, color='gray', linestyle='--', 
                       label=f'Buy & Hold ({buy_hold_return:+.1f}%)')
            
            ax2.set_title('Cumulative Returns')
            ax2.set_ylabel('Return (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Performance metrics
            ax3 = axes[1, 0]
            
            models = list(results.keys())
            accuracies = [results[m]['accuracy'] for m in models]
            win_rates = [results[m]['win_rate'] for m in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax3.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.7)
            ax3.bar(x + width/2, win_rates, width, label='Win Rate', alpha=0.7)
            
            ax3.set_title('Performance Metrics')
            ax3.set_ylabel('Rate')
            ax3.set_xticks(x)
            ax3.set_xticklabels(models)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Return distribution
            ax4 = axes[1, 1]
            
            all_returns = []
            for result in results.values():
                if result['trades']:
                    all_returns.extend([t['pnl_pct'] for t in result['trades']])
            
            if all_returns:
                ax4.hist(all_returns, bins=15, alpha=0.7, color='skyblue')
                ax4.axvline(x=0, color='red', linestyle='--', label='Break-even')
                ax4.axvline(x=np.mean(all_returns), color='green', linestyle='-', 
                           label=f'Mean ({np.mean(all_returns):+.2f}%)')
            
            ax4.set_title('Trade Return Distribution')
            ax4.set_xlabel('Return (%)')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if SAVE_CHARTS:
                chart_path = os.path.join(self.chart_dir, 'trading_analysis.png')
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                print(f"Chart saved: {chart_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Chart creation failed: {e}")
    
    def _analyze_trades(self, all_trades):
        """Analyze trades."""
        if not all_trades:
            return
        
        print("\nTrade Analysis:")
        print("=" * 40)
        
        df = pd.DataFrame(all_trades)
        
        total_trades = len(df)
        winning_trades = len(df[df['pnl_pct'] > 0])
        
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades} ({winning_trades/total_trades:.1%})")
        print(f"Average Return: {df['pnl_pct'].mean():+.2f}%")
        print(f"Best Trade: {df['pnl_pct'].max():+.2f}%")
        print(f"Worst Trade: {df['pnl_pct'].min():+.2f}%")
        
        # By position type
        for position in ['LONG', 'SHORT']:
            pos_trades = df[df['position'] == position]
            if len(pos_trades) > 0:
                win_rate = len(pos_trades[pos_trades['pnl_pct'] > 0]) / len(pos_trades)
                avg_return = pos_trades['pnl_pct'].mean()
                print(f"{position}: {len(pos_trades)} trades, {win_rate:.1%} win rate, {avg_return:+.2f}% avg")
        
        # Save details
        trades_path = os.path.join(self.data_dir, 'trade_details.csv')
        df.to_csv(trades_path, index=False)
        print(f"Trade details saved: {trades_path}")
    
    def get_next_prediction(self):
        """Get next prediction."""
        print("Getting next prediction...")
        
        if not self._load_models():
            print("No models found.")
            return None
        
        if self.processed_data is None:
            try:
                self.processed_data = pd.read_csv(os.path.join(self.data_dir, "processed_features.csv"), 
                                                index_col=0, parse_dates=True)
            except FileNotFoundError:
                print("Processed features not found.")
                return None
        
        # Get latest data
        latest_data = self.processed_data.tail(1).drop('Target', axis=1, errors='ignore')
        latest_date = latest_data.index[0]
        
        print(f"Prediction based on data as of: {latest_date.date()}")
        
        # Preprocess
        latest_processed = self._preprocess_for_prediction(latest_data)
        if latest_processed is None:
            return None
        
        predictions = {'date': latest_date}
        
        for model_name, model in self.models.items():
            try:
                if self.scalers[model_name] is not None:
                    X_scaled = self.scalers[model_name].transform(latest_processed)
                else:
                    X_scaled = latest_processed.values
                
                prediction = model.predict(X_scaled)[0]
                probabilities = model.predict_proba(X_scaled)[0]
                confidence = max(probabilities)
                
                predictions[model_name] = {
                    'prediction': 'UP' if prediction == 1 else 'DOWN',
                    'probability_up': probabilities[1],
                    'confidence': confidence,
                    'trade_signal': 'BUY' if prediction == 1 and confidence >= CONFIDENCE_THRESHOLD else 
                                   'SELL' if prediction == 0 and confidence >= CONFIDENCE_THRESHOLD else 'HOLD'
                }
                
            except Exception as e:
                predictions[model_name] = {'error': str(e)}
        
        # Display results
        print(f"\nNext {PREDICTION_HORIZON}-Day Prediction:")
        print("=" * 40)
        
        for model_name, result in predictions.items():
            if model_name != 'date':
                if 'error' not in result:
                    print(f"\n{model_name}:")
                    print(f"  Direction: {result['prediction']}")
                    print(f"  Confidence: {result['confidence']:.1%}")
                    print(f"  Trade Signal: {result['trade_signal']}")
                    
                    if result['trade_signal'] != 'HOLD':
                        print(f"  Position Size: {POSITION_SIZE:.1%} of capital")
                else:
                    print(f"\n{model_name}: ERROR - {result['error']}")
        
        return predictions
    
    def run_full_system(self):
        """Run complete system."""
        print("Running Complete SPX Trading System")
        print("=" * 60)
        
        # Data collection
        print("\n1. Data Collection")
        print("-" * 20)
        if self.collect_data() is None:
            return None, None
        
        # Feature engineering
        print("\n2. Feature Engineering")
        print("-" * 20)
        if self.engineer_features() is None:
            return None, None
        
        # Model training
        print("\n3. Model Training")
        print("-" * 20)
        if not self.train_models():
            return None, None
        
        # Backtesting
        print("\n4. Backtesting")
        print("-" * 20)
        backtest_results = self.backtest_with_charts()
        
        # Next prediction
        print("\n5. Next Prediction")
        print("-" * 20)
        next_prediction = self.get_next_prediction()
        
        print("\n" + "=" * 60)
        print("SPX Trading System Complete!")
        print(f"Charts: {self.chart_dir}")
        print(f"Data: {self.data_dir}")
        
        return backtest_results, next_prediction


def main():
    """Main function."""
    print("SPX Trading System - Forward-Looking Bias Prevention")
    print("=" * 60)
    print("Configuration:")
    print(f"  Data: {DATA_START_DATE} to {DATA_END_DATE}")
    print(f"  Training: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
    print(f"  Backtesting: {BACKTEST_START_DATE} to {BACKTEST_END_DATE}")
    print(f"  Prediction Horizon: {PREDICTION_HORIZON} days")
    print(f"  Capital: ${INITIAL_CAPITAL:,}")
    print(f"  Position Size: {POSITION_SIZE:.1%}")
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD:.1%}")
    print("=" * 60)
    
    system = SPXTradingSystem()
    backtest_results, next_prediction = system.run_full_system()
    
    return system, backtest_results, next_prediction


if __name__ == "__main__":
    system, backtest_results, next_prediction = main()
