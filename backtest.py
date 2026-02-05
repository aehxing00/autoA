import pandas as pd
import numpy as np
import random
from strategy import QuantModel
from features import FeatureEngineer
from data_loader import DataLoader

class Backtester:
    def __init__(self, stock_codes, start_date, end_date, initial_capital=50000, 
                 stop_loss=0.05, take_profit=0.15, max_positions=3, rebalance_days=5,
                 commission_rate=0.00025, min_commission=5.0, stamp_duty_rate=0.0005,
                 slippage_rate=0.001, buy_signal_threshold=0.35, max_down_risk=0.45,
                 score_mix_limit=0.6, fallback_signal_threshold=0.2):
        self.stock_codes = stock_codes
        self.start_date = start_date
        self.end_date = end_date
        self.capital = initial_capital
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_positions = max_positions
        self.rebalance_days = rebalance_days
        
        # 交易成本与滑点参数
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_duty_rate = stamp_duty_rate
        self.slippage_rate = slippage_rate # 滑点率，模拟成交价偏差
        self.buy_signal_threshold = buy_signal_threshold
        self.max_down_risk = max_down_risk
        self.score_mix_limit = score_mix_limit
        self.fallback_signal_threshold = fallback_signal_threshold
        
        self.loader = DataLoader()
        # 注意: FeatureEngineer 需要在外部初始化或在这里传入参数，
        # 为了简单起见，我们假设 FeatureEngineer 在 app.py 中处理并传入数据
        self.engineer = FeatureEngineer() 
        self.model = QuantModel() 
        self.portfolio_value = []
        self.positions = {} 

    def _apply_slippage(self, price, direction):
        """
        应用滑点计算真实成交价
        买入: 价格上涨 (Price * (1 + slippage))
        卖出: 价格下跌 (Price * (1 - slippage))
        """
        if direction == 'buy':
            return price * (1 + self.slippage_rate)
        elif direction == 'sell':
            return price * (1 - self.slippage_rate)
        return price

    def _calculate_buy_cost(self, price, shares):
        """计算买入总成本 (包含佣金)"""
        # 应用滑点
        exec_price = self._apply_slippage(price, 'buy')
        amount = exec_price * shares
        commission = max(self.min_commission, amount * self.commission_rate)
        return amount + commission, commission, exec_price

    def _calculate_sell_revenue(self, price, shares):
        """计算卖出净收入 (扣除佣金和印花税)"""
        # 应用滑点
        exec_price = self._apply_slippage(price, 'sell')
        amount = exec_price * shares
        commission = max(self.min_commission, amount * self.commission_rate)
        stamp_duty = amount * self.stamp_duty_rate
        # 净收入 = 成交额 - 佣金 - 印花税
        return amount - commission - stamp_duty, commission, stamp_duty, exec_price

    def run_with_data(self, data_map, trained_model):
        """
        使用预加载的数据和已训练的模型运行回测
        返回: (净值DataFrame, 交易记录DataFrame)
        """
        self.model = trained_model
        
        all_dates = sorted(list(set(d for df in data_map.values() for d in df.index)))
        # 简单起见，我们从第20天开始（预留特征窗口）
        test_dates = all_dates[20:]
        
        current_capital = self.capital
        history = []
        transactions = []
        diagnostics = []
        
        print(f"开始回测 | 止损: {self.stop_loss:.1%} | 止盈: {self.take_profit:.1%} | 最大持仓: {self.max_positions} | 调仓周期: {self.rebalance_days}天")
        
        for i, date in enumerate(test_dates):
            # 1. 每日盯市 (Mark to Market) & 风控检查
            # port_val = current_capital # 错误：这里current_capital是现金
            # 应计算: 现金 + 持仓市值
            
            # 检查持仓
            codes_to_sell = []
            for code, pos_info in self.positions.items():
                if date in data_map[code].index:
                    current_price = data_map[code].loc[date, 'close']
                    shares = pos_info['shares']
                    entry_price = pos_info['entry_price']
                    
                    # 检查止盈止损
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    if pnl_pct <= -self.stop_loss:
                        # 触发止损
                        revenue, comm, stamp, exec_price = self._calculate_sell_revenue(current_price, shares)
                        current_capital += revenue
                        codes_to_sell.append((code, '止损卖出', exec_price, comm, stamp))
                    elif pnl_pct >= self.take_profit:
                        # 触发止盈
                        revenue, comm, stamp, exec_price = self._calculate_sell_revenue(current_price, shares)
                        current_capital += revenue
                        codes_to_sell.append((code, '止盈卖出', exec_price, comm, stamp))
                else:
                    # 停牌，简单处理
                    pass
            
            # 执行风控卖出
            for code, reason, price, comm, stamp in codes_to_sell:
                if code in self.positions:
                    shares = self.positions[code]['shares']
                    transactions.append({
                        '日期': date,
                        '股票代码': code,
                        '操作': '卖出',
                        '价格': price,
                        '数量': shares,
                        '金额': shares * price,
                        '手续费': comm,
                        '印花税': stamp,
                        '原因': reason
                    })
                    del self.positions[code]
            
            # 2. 定期调仓 (仅当还有空位或资金时)
            if i % self.rebalance_days == 0: 
                # 生成信号
                candidates = []
                fallback_candidates = []
                for code, df in data_map.items():
                    if date in df.index:
                        row = df.loc[[date]]
                        # 使用选定特征进行预测，如果列不存在会自动补0 (在predict方法中处理)
                        if not row.empty:
                            probs = self.model.predict_proba(row).iloc[0]
                            limit_prob = probs.get("limit_up", 0.0)
                            sharp_up_prob = probs.get("sharp_up", 0.0)
                            sharp_down_prob = probs.get("sharp_down", 0.0)
                            score = (limit_prob * self.score_mix_limit) + (sharp_up_prob * (1 - self.score_mix_limit))
                            if sharp_down_prob <= self.max_down_risk and code not in self.positions:
                                if score >= self.buy_signal_threshold:
                                    candidates.append((code, score, row['close'].values[0]))
                                elif score >= self.fallback_signal_threshold:
                                    fallback_candidates.append((code, score, row['close'].values[0]))
                
                # 选股
                candidates.sort(key=lambda x: x[1], reverse=True)
                fallback_candidates.sort(key=lambda x: x[1], reverse=True)
                
                # 能够持有的最大空位数
                available_slots = self.max_positions - len(self.positions)
                
                # 如果有空位且有资金
                pick_list = candidates if candidates else fallback_candidates
                used_fallback = len(candidates) == 0
                if pick_list and available_slots > 0 and current_capital > 0:
                    top_picks = pick_list[:available_slots]
                    # 简单资金分配：剩余资金均分给空位
                    # 注意：如果已有持仓，current_capital是剩余现金
                    alloc_per_stock = current_capital / available_slots
                    
                    for code, score, price in top_picks:
                        # 预估最大可买股数 (考虑手续费和滑点)
                        # Cost = Price * (1+Slippage) * Shares * (1+Rate) + MinComm
                        estimated_price = price * (1 + self.slippage_rate)
                        max_shares = int((alloc_per_stock - self.min_commission) / (estimated_price * (1 + self.commission_rate)))
                        
                        # 向下取整到100的倍数
                        shares = (max_shares // 100) * 100
                        
                        # 至少买1手 (100股)
                        if shares >= 100:
                            cost, comm, exec_price = self._calculate_buy_cost(price, shares)
                            if current_capital >= cost:
                                self.positions[code] = {'shares': shares, 'entry_price': exec_price}
                                current_capital -= cost
                                transactions.append({
                                    '日期': date,
                                    '股票代码': code,
                                    '操作': '买入',
                                    '价格': exec_price,
                                    '数量': shares,
                                    '金额': shares * exec_price, # 成交额，不含费
                                    '手续费': comm,
                                    '印花税': 0.0,
                                    '原因': f'上涨信号:{score:.2f}'
                                })
                all_candidate_count = len(candidates) + len(fallback_candidates)
                avg_score = 0.0
                if all_candidate_count > 0:
                    avg_score = (sum([x[1] for x in candidates]) + sum([x[1] for x in fallback_candidates])) / all_candidate_count
                diagnostics.append({
                    '日期': date,
                    '候选数量': all_candidate_count,
                    '主阈值命中': len(candidates),
                    '兜底命中': len(fallback_candidates),
                    '是否兜底': int(used_fallback),
                    '平均评分': avg_score,
                    '空位数量': available_slots,
                    '买入数量': min(available_slots, len(pick_list)) if pick_list else 0
                })
            
            # 3. 记录当日净值 (收盘后)
            todays_equity = current_capital
            for code, pos_info in self.positions.items():
                if date in data_map[code].index:
                    price = data_map[code].loc[date, 'close']
                    todays_equity += pos_info['shares'] * price
                else:
                    todays_equity += pos_info['shares'] * pos_info['entry_price'] 
            
            history.append({'date': date, 'value': todays_equity})

        return pd.DataFrame(history).set_index('date'), pd.DataFrame(transactions), pd.DataFrame(diagnostics)

    def run(self):
        # 兼容旧接口，虽然现在主要用 run_with_data
        pass
