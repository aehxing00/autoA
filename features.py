import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, ma_window=20, vol_window=5):
        self.ma_window = ma_window
        self.vol_window = vol_window

    # --- Technical Indicators Helper Methods ---
    def calculate_rsi(self, series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, series, short=12, long=26, signal=9):
        short_ema = series.ewm(span=short, adjust=False).mean()
        long_ema = series.ewm(span=long, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def calculate_bollinger(self, series, window=20, std_dev=2):
        ma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        # %B indicator: (Price - Lower) / (Upper - Lower)
        pct_b = (series - lower) / (upper - lower)
        # Band Width: (Upper - Lower) / Middle
        width = (upper - lower) / ma
        return pct_b, width

    def calculate_atr(self, high, low, close, window=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr

    def calculate_kdj(self, high, low, close, window=9):
        low_min = low.rolling(window=window).min()
        high_max = high.rolling(window=window).max()
        rsv = (close - low_min) / (high_max - low_min) * 100
        k = rsv.ewm(com=2).mean()
        d = k.ewm(com=2).mean()
        j = 3 * k - 2 * d
        return k, d, j

    def calculate_features(self, df, benchmark_df=None):
        """
        计算单只股票的特征。
        期望输入列：open, high, low, close, volume, turnover, pe (可选)
        """
        df = df.copy()
        
        # --- 1. 动量因子 (Momentum Factors) ---
        # Returns
        df['ret_1d'] = df['close'].pct_change(1)
        df['ret_5d'] = df['close'].pct_change(5)
        df['ret_10d'] = df['close'].pct_change(10)
        df['ret_20d'] = df['close'].pct_change(20)
        
        # Log Returns
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        # RSI
        df['rsi_6'] = self.calculate_rsi(df['close'], window=6)
        df['rsi_12'] = self.calculate_rsi(df['close'], window=12)
        df['rsi_24'] = self.calculate_rsi(df['close'], window=24)
        
        # MACD
        macd, signal = self.calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_diff'] = macd - signal
        
        # KDJ
        df['kdj_k'], df['kdj_d'], df['kdj_j'] = self.calculate_kdj(df['high'], df['low'], df['close'])
        
        # BIAS (乖离率)
        for w in [5, 10, 20, 60]:
            ma = df['close'].rolling(window=w).mean()
            df[f'bias_{w}'] = (df['close'] - ma) / ma

        # --- 2. 波动率因子 (Volatility Factors) ---
        # Historical Volatility
        df['volatility_5d'] = df['ret_1d'].rolling(window=5).std()
        df['volatility_20d'] = df['ret_1d'].rolling(window=20).std()
        df['volatility_60d'] = df['ret_1d'].rolling(window=60).std()
        
        # Amplitude (振幅)
        df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1)
        df['amplitude_ma5'] = df['amplitude'].rolling(window=5).mean()
        
        # ATR (Average True Range)
        df['atr_14'] = self.calculate_atr(df['high'], df['low'], df['close'])
        df['natr_14'] = df['atr_14'] / df['close'] # Normalized ATR
        
        # Bollinger Bands
        df['bb_pct_b'], df['bb_width'] = self.calculate_bollinger(df['close'])

        # --- 3. 成交量因子 (Volume Factors) ---
        # Volume Change
        df['vol_change_1d'] = df['volume'].pct_change(1)
        
        # Volume Ratio (量比)
        df['vol_ma5'] = df['volume'].rolling(window=5).mean()
        df['vol_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio_5'] = df['volume'] / df['vol_ma5']
        df['volume_ratio_20'] = df['volume'] / df['vol_ma20']
        
        # Turnover (换手率)
        if 'turnover' in df.columns:
            df['turnover_ma5'] = df['turnover'].rolling(window=5).mean()
            df['turnover_ma20'] = df['turnover'].rolling(window=20).mean()
            df['turnover_delta'] = df['turnover'] / df['turnover_ma5']
            df['turnover_volatility'] = df['turnover'].rolling(window=20).std()

        # --- 4. 价值与基本面因子 (Value/Fundamental Factors) ---
        if 'pe' in df.columns:
            # PE Rank
            window_5y = 252 * 5
            df['pe_rank_5y'] = df['pe'].rolling(window=window_5y, min_periods=50).rank(pct=True)
            
            # PEG (Simplified)
            if 'profit_growth' not in df.columns:
                df['profit_growth'] = 0.15 
            df['peg'] = df['pe'] / (df['profit_growth'] * 100)
        else:
            df['pe_rank_5y'] = 0.5
            df['peg'] = 1.0
            
        if 'pb' in df.columns:
             df['pb_rank_5y'] = df['pb'].rolling(window=252*5, min_periods=50).rank(pct=True)
        else:
             df['pb_rank_5y'] = 0.5

        # --- 5. 趋势因子 (Trend Factors) ---
        # MA Slope
        for w in [5, 10, 20]:
            ma = df['close'].rolling(window=w).mean()
            df[f'ma_{w}_slope'] = (ma - ma.shift(1)) / ma.shift(1)
            
        # Price Position relative to High/Low
        df['price_pos_20d'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min())
        df['price_pos_60d'] = (df['close'] - df['low'].rolling(60).min()) / (df['high'].rolling(60).max() - df['low'].rolling(60).min())

        # --- 6. 特征交叉与复合因子 (Composite Factors) ---
        # Low Value & High Turnover
        df['low_val_high_turn'] = (df['pe_rank_5y'] < 0.3).astype(int) * (df['turnover_delta'] > 1.5).astype(int)
        
        # High Momentum & Low Volatility (Steady Growth)
        df['steady_growth'] = (df['ret_20d'] > 0.05).astype(int) * (df['volatility_20d'] < df['volatility_20d'].rolling(60).mean()).astype(int)
        
        # Volume Price Correlation
        df['vol_price_corr_10'] = df['close'].rolling(10).corr(df['volume'])

        # 删除因滚动窗口产生的NaN值
        df.dropna(inplace=True)
        
        # 替换无限值
        df.replace([np.inf, -np.inf], 0, inplace=True)
        
        return df

    def prepare_training_data(self, df, future_days=5):
        """
        创建训练数据的目标变量 (未来收益率)。
        目标: 如果未来收益率 > 0 则为 1，否则为 0 (分类问题)
        """
        # 目标: 未来N天的收益率
        df['target_return'] = df['close'].shift(-future_days) / df['close'] - 1
        
        # 清除最后N行没有目标数据的行
        data = df.dropna().copy()
        
        return data
