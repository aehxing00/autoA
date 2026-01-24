from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
import pickle

class QuantModel:
    def __init__(self, n_estimators=100, max_depth=None):
        self.base_model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.model = self.base_model # Final model to use
        self.selector = None # Feature selector
        
        # Initial extensive feature list (match with features.py)
        self.feature_cols = [
            # Momentum
            'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'log_ret',
            'rsi_6', 'rsi_12', 'rsi_24',
            'macd', 'macd_signal', 'macd_diff',
            'kdj_k', 'kdj_d', 'kdj_j',
            'bias_5', 'bias_10', 'bias_20', 'bias_60',
            
            # Volatility
            'volatility_5d', 'volatility_20d', 'volatility_60d',
            'amplitude', 'amplitude_ma5',
            'atr_14', 'natr_14',
            'bb_pct_b', 'bb_width',
            
            # Volume
            'vol_change_1d',
            'volume_ratio_5', 'volume_ratio_20',
            'turnover_delta', 'turnover_volatility',
            
            # Value/Fundamental
            'pe_rank_5y', 'peg', 'pb_rank_5y',
            'relative_pe',
            
            # Trend
            'rel_strength', 'trend_filter',
            'ma_5_slope', 'ma_10_slope', 'ma_20_slope',
            'price_pos_20d', 'price_pos_60d',
            'north_money_flow',
            
            # Composite
            'low_val_high_turn', 'steady_growth', 'vol_price_corr_10',
            'gold_pit', 'run_away'
        ]
        
        # Selected features after training
        self.selected_features = self.feature_cols 

    def train(self, data):
        """
        使用提供的数据训练模型。
        自动执行特征筛选。
        """
        # 1. 数据准备
        # 确保所有列都存在，缺失的用0填充 (防守性编程)
        for col in self.feature_cols:
            if col not in data.columns:
                data[col] = 0
                
        X = data[self.feature_cols]
        y = (data['target_return'] > 0.0).astype(int)
        
        if len(y.unique()) < 2:
            print("警告: 目标变量只有一个类别，模型无法有效学习。")
            return {'accuracy': 0, 'precision': 0, 'feature_importance': {}}

        # 2. 特征筛选 (Feature Selection)
        # 使用随机森林进行初步筛选，保留重要性大于平均值的特征
        # max_features=20 限制最多选20个，防止过拟合
        self.selector = SelectFromModel(estimator=self.base_model, max_features=20, threshold="mean")
        self.selector.fit(X, y)
        
        # 获取筛选后的特征名
        selected_mask = self.selector.get_support()
        self.selected_features = np.array(self.feature_cols)[selected_mask].tolist()
        
        # 3. 使用筛选后的特征进行最终训练
        X_selected = X[self.selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, shuffle=False)
        
        self.model.fit(X_train, y_train)
        
        # 4. 评估
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        
        return {
            'accuracy': acc,
            'precision': prec,
            'feature_importance': dict(zip(self.selected_features, self.model.feature_importances_)),
            'selected_features_count': len(self.selected_features)
        }

    def predict(self, data):
        """
        对新数据进行预测评分。
        """
        # 确保列存在
        for col in self.selected_features:
            if col not in data.columns:
                data[col] = 0
                
        X = data[self.selected_features]
        probs = self.model.predict_proba(X)[:, 1]
        return probs

    def save(self, filepath="model.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump({'model': self.model, 'features': self.selected_features}, f)

    def load(self, filepath="model.pkl"):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.model = data['model']
            self.selected_features = data['features']
