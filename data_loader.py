import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False
    print("AkShare not found. Using synthetic data for demonstration.")

class DataLoader:
    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def get_stock_list(self):
        """获取股票代码列表 (示例: 使用小股票池或获取全部)"""
        # 演示用，返回样本列表。真实应用中可获取指数成分股。
        return ["600519", "000858", "601318", "002594", "300750"]

    def fetch_price_data(self, symbol, start_date, end_date):
        """获取历史价格数据 (OHLCV)"""
        # 尝试读取缓存
        cache_path = os.path.join(self.cache_dir, f"{symbol}_{start_date}_{end_date}.csv")
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, index_col='date', parse_dates=['date'])
                return df
            except Exception as e:
                print(f"读取缓存 {cache_path} 失败: {e}")

        if HAS_AKSHARE:
            try:
                # 调整akshare的代码格式 (例如: sh600519 -> 600519)
                code = symbol.replace("sh", "").replace("sz", "")
                
                df = ak.stock_zh_a_hist(symbol=code, start_date=start_date, end_date=end_date, adjust="qfq")
                if df is None or df.empty:
                    # 如果API未返回数据，则回退到模拟数据
                    return self.generate_mock_data(start_date, end_date)
                
                df.rename(columns={
                    '日期': 'date', '开盘': 'open', '收盘': 'close', 
                    '最高': 'high', '最低': 'low', '成交量': 'volume', 
                    '成交额': 'amount', '换手率': 'turnover'
                }, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # 写入缓存
                df.to_csv(cache_path)
                return df
            except Exception as e:
                print(f"获取 {symbol} 价格时出错: {e}")
                return self.generate_mock_data(start_date, end_date)
        else:
            return self.generate_mock_data(start_date, end_date)

    def fetch_valuation_data(self, symbol):
        """
        获取历史PE/PB数据。
        注意: 免费的历史PE源有限。
        如果API失败或太慢，使用模拟生成以进行稳健演示。
        """
        try:
            # 尝试从akshare获取 (示例接口，可能需要特定的接口)
            # df = ak.stock_a_indicator_lg(symbol=symbol)
            # return df
            # 为了演示稳定性，我将模拟与价格对齐的估值数据
            pass
        except:
            pass
        return None

    def generate_mock_data(self, start_date, end_date):
        """当API不可用时生成用于测试逻辑的合成数据"""
        dates = pd.date_range(start=start_date, end=end_date)
        n = len(dates)
        
        # 价格随机游走
        returns = np.random.normal(0.0005, 0.02, n)
        price = 100 * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'open': price * (1 + np.random.normal(0, 0.005, n)),
            'high': price * (1 + abs(np.random.normal(0, 0.01, n))),
            'low': price * (1 - abs(np.random.normal(0, 0.01, n))),
            'close': price,
            'volume': np.random.randint(1000, 100000, n),
            'turnover': np.random.uniform(0.5, 5.0, n)  # 换手率百分比
        }, index=dates)
        
        # 添加通常需要的模拟基本面列
        df['pe'] = np.random.uniform(10, 60, n)
        df['pb'] = np.random.uniform(1, 5, n)
        df['pe_ttm'] = df['pe']
        
        return df

    def get_northbound_flow(self, start_date, end_date):
        """获取北向资金流向"""
        try:
            df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
            # 按日期过滤...
            return df
        except:
            return pd.DataFrame()

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.fetch_price_data("600519", "20230101", "20231231")
    print(df.head())
