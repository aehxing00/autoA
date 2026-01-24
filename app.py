import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import concurrent.futures
from data_loader import DataLoader
from features import FeatureEngineer
from strategy import QuantModel
from backtest import Backtester

st.set_page_config(page_title="AI量化交易助手 Pro", layout="wide")

st.title("🚀 AI量化交易助手 Pro")
st.markdown("""
> **隐私承诺**: 本系统所有数据处理、模型训练与策略回测均在**您的本地机器**上完成，不会上传任何策略参数或私钥至云端，请放心使用。
""")

# --- 侧边栏: 高级配置 ---
st.sidebar.header("1. 基础设置")
initial_capital = st.sidebar.number_input("初始资金", value=100000, step=10000)
data_source_mode = st.sidebar.radio("数据源模式", ["实盘数据 (AkShare)", "模拟数据 (Demo)"], index=0)

# 训练/回测 时间分割
st.sidebar.header("2. 时间与数据分割")
split_date = st.sidebar.date_input("训练/测试分割日期", pd.to_datetime("2024-01-01"), help="在此日期之前的数据用于训练，之后的数据用于回测")

st.sidebar.header("3. 模型超参数 (Model)")
n_estimators = st.sidebar.slider("决策树数量 (n_estimators)", 10, 500, 200, help="树越多越抗噪，但计算越慢")
max_depth = st.sidebar.slider("最大树深 (max_depth)", 3, 20, 5, help="深度越深越容易过拟合，建议保持在3-10之间")
ma_window = st.sidebar.slider("均线窗口 (MA Window)", 5, 60, 20, help="计算趋势指标的窗口大小")
vol_window = st.sidebar.slider("波动率窗口 (Vol Window)", 5, 30, 5, help="计算波动率的窗口大小")

st.sidebar.header("4. 交易风控 (Risk)")
stop_loss_pct = st.sidebar.slider("止损比例 (Stop Loss %)", 0.0, 20.0, 5.0, step=0.5) / 100.0
take_profit_pct = st.sidebar.slider("止盈比例 (Take Profit %)", 0.0, 50.0, 15.0, step=1.0) / 100.0
max_positions = st.sidebar.slider("最大持仓数 (Max Positions)", 1, 10, 3, help="同时持有的最大股票数量")
rebalance_days = st.sidebar.slider("调仓周期 (天)", 1, 20, 5, help="每隔多少个交易日检查一次换股信号")

with st.sidebar.expander("💸 交易成本设置 (Advanced)"):
        commission_rate = st.number_input("佣金费率 (如万分之2.5)", value=0.00025, step=0.00005, format="%.5f")
        min_commission = st.number_input("最低佣金 (元)", value=5.0, step=1.0)
        stamp_duty_rate = st.number_input("印花税率 (卖出收取)", value=0.0005, step=0.0001, format="%.4f", help="2023年8月28日起，A股印花税减半征收为0.05%")
        slippage_rate = st.number_input("滑点率 (Slippage)", value=0.001, step=0.001, format="%.3f", help="模拟成交价与决策价的偏差，0.001代表0.1%")

# 初始化Session State
if 'data_map' not in st.session_state:
    st.session_state['data_map'] = {}
if 'trained_model' not in st.session_state:
    st.session_state['trained_model'] = None
if 'feature_engineer' not in st.session_state:
    st.session_state['feature_engineer'] = FeatureEngineer(ma_window=ma_window, vol_window=vol_window)
else:
    # 更新参数
    st.session_state['feature_engineer'].ma_window = ma_window
    st.session_state['feature_engineer'].vol_window = vol_window

def calculate_max_drawdown(series):
    if series.empty:
        return 0.0
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    return drawdown.min()

def fetch_single_stock(code, start_str, end_str, loader):
    df = loader.fetch_price_data(code, start_str, end_str)
    if df is not None and not df.empty:
        # 必须在主线程或这里计算特征? 
        # 为了利用多核，最好在这里计算，但FeatureEngineer在session_state中
        # 我们可以创建一个临时的FeatureEngineer
        # 或者只返回原始数据，后续并行计算特征
        return code, df
    return code, None

# --- 选项卡布局 ---
tab1, tab2, tab3 = st.tabs(["1️⃣ 选股与数据", "2️⃣ 模型训练与诊断", "3️⃣ 策略回测"])

# === Tab 1: 选股与数据 ===
with tab1:
    st.subheader("第一步：定义股票池并获取数据")
    
    with st.expander("❓ 为什么需要自定义股票池？为何不使用全市场建模？"):
        st.info("""
        **全市场建模的挑战**：
        1. **数据量大**：A股有5000+只股票，每日下载全量数据需要较长时间和大量网络流量。
        2. **内存限制**：单机（个人电脑）内存有限，同时处理5000只股票的高频特征矩阵容易导致内存溢出。
        3. **计算效率**：在单机上对全市场进行随机森林训练和回测会非常缓慢（可能需要数小时）。
        
        **建议**：先使用一组具有代表性的股票（如自选股或指数成份股）验证策略逻辑，确认有效后再分批次扩大范围。
        """)

    col1, col2 = st.columns([3, 1])
    with col1:
        # 默认股票池：精选A股核心资产（约100只，涵盖主要行业龙头）
        default_stock_pool = (
            "600519, 000858, 601318, 002594, 300750, 600036, 601166, 600030, 600887, 600276, "
            "601012, 603288, 000333, 002415, 601888, 300059, 300015, 603259, 600900, 601633, "
            "002714, 600438, 600436, 600309, 600585, 600690, 002304, 002475, 300124, 300014, "
            "601398, 601288, 601939, 601988, 601328, 600000, 600016, 600015, 601169, 601998, "
            "000001, 000002, 000651, 000725, 600104, 600018, 601857, 601088, 601899, 601668, "
            "601800, 601766, 601989, 601601, 600999, 601688, 600362, 600196, 600547, 603986, "
            "300498, 300601, 300274, 300413, 300433, 300760, 688111, 688012, 688036, 688008, "
            "603501, 600809, 600570, 600298, 601919, 601066, 600703, 600741, 603160, 603799, "
            "002027, 002241, 002142, 002007, 002001, 000963, 000568, 000538, 002352, 002460, "
            "002466, 002493, 002555, 002601, 002607, 002624, 002736, 002812, 002821, 002841"
        )
        stock_pool_input = st.text_area("输入股票代码 (逗号分隔)", default_stock_pool, height=150)
        st.caption("默认已加载约100只A股核心龙头资产。支持自定义增删。")
    with col2:
        # 默认拉取2年数据，以便有足够的数据进行训练和测试
        start_date = st.date_input("开始日期", pd.to_datetime("2023-01-01"))
        end_date = st.date_input("结束日期", pd.to_datetime("2024-12-31"))
    
    if st.button("📥 获取数据"):
        loader = DataLoader()
        stock_codes = [x.strip() for x in stock_pool_input.split(",")]
        
        with st.spinner("正在并行加载数据 (支持本地缓存)..."):
            progress_bar = st.progress(0)
            data_map = {}
            
            # 使用并行处理加速数据获取
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                start_str = start_date.strftime("%Y%m%d")
                end_str = end_date.strftime("%Y%m%d")
                
                future_to_code = {executor.submit(fetch_single_stock, code, start_str, end_str, loader): code for code in stock_codes}
                
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_code):
                    code, df = future.result()
                    if df is not None:
                        # 计算特征 (这里仍然是串行的，如果特征计算慢，也可以并行化)
                        # 为了安全起见，我们使用 session_state 中的 engineer
                        df = st.session_state['feature_engineer'].calculate_features(df)
                        data_map[code] = df
                    
                    completed_count += 1
                    progress_bar.progress(completed_count / len(stock_codes))
            
            st.session_state['data_map'] = data_map
            st.success(f"成功加载 {len(data_map)} 只股票的数据！")
            
    # 展示数据概览
    if st.session_state['data_map']:
        st.write("已加载数据概览 (最后5行):")
        first_code = list(st.session_state['data_map'].keys())[0]
        st.write(f"股票代码: {first_code}")
        st.dataframe(st.session_state['data_map'][first_code].tail())

# === Tab 2: 模型训练 ===
with tab2:
    st.subheader("第二步：训练AI模型")
    if not st.session_state['data_map']:
        st.warning("请先在“选股与数据”标签页获取数据。")
    else:
        st.markdown(f"当前配置: **{n_estimators}** 棵树, 最大深度 **{max_depth}**")
        st.info(f"**训练模式**: 使用 {split_date} 之前的数据训练，之后的数据用于验证准确率。")
        
        if st.button("🧠 开始训练模型"):
            model = QuantModel(n_estimators=n_estimators, max_depth=max_depth)
            
            # 准备训练数据
            # 改进：按股票单独处理时间分割和目标变量计算，避免直接concat导致的shift跨股票问题
            train_dfs = []
            for code, df in st.session_state['data_map'].items():
                # 严格按照时间分割，防止未来函数
                train_mask = df.index < pd.to_datetime(split_date)
                # 使用 copy 避免 SettingWithCopyWarning
                train_part = df[train_mask].copy()
                
                if not train_part.empty:
                    # 计算目标变量 (Shift操作)
                    train_part_ready = st.session_state['feature_engineer'].prepare_training_data(train_part)
                    if not train_part_ready.empty:
                        train_dfs.append(train_part_ready)
            
            if not train_dfs:
                st.error("训练集为空！请检查“训练/测试分割日期”是否早于数据结束日期，或者数据是否包含该日期之前的记录。")
            else:
                train_data = pd.concat(train_dfs)
                
                # 双重检查
                if train_data.empty:
                     st.error("训练数据在预处理后为空（可能是因为数据长度不足以计算未来收益率）。")
                else:
                    with st.spinner("AI正在学习历史规律..."):
                        metrics = model.train(train_data)
                        st.session_state['trained_model'] = model
                        
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("准确率 (Accuracy)", f"{metrics['accuracy']:.2%}")
                    col_m2.metric("精确率 (Precision)", f"{metrics['precision']:.2%}")
                    # 显示筛选后的特征数量
                    selected_count = metrics.get('selected_features_count', len(metrics['feature_importance']))
                    col_m3.metric("特征数量 (Selected/Total)", f"{selected_count} / {len(model.feature_cols)}")
                    
                    st.write("### 因子重要性排行 (Top Features)")
                    st.caption("✨ 系统已自动从50+个候选因子中筛选出最有效的因子进行建模")
                    # 排序因子重要性
                    importance_df = pd.DataFrame(
                        list(metrics['feature_importance'].items()), 
                        columns=['Feature', 'Importance']
                    ).sort_values(by='Importance', ascending=False)
                    
                    st.bar_chart(importance_df.set_index('Feature'))
                    st.info("💡 这里的长条越长，说明该因子对预测涨跌越重要。")

# === Tab 3: 策略回测 ===
with tab3:
    st.subheader("第三步：实盘模拟回测")
    if st.session_state['trained_model'] is None:
        st.warning("请先在“模型训练”标签页训练模型。")
    else:
        st.markdown("#### 模拟交易参数")
        st.info(f"**回测区间**: 仅在 {split_date} 之后的数据上进行回测，模拟真实交易环境。")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("初始资金", f"¥{initial_capital:,}")
        c2.metric("止损线", f"-{stop_loss_pct*100}%")
        c3.metric("止盈线", f"+{take_profit_pct*100}%")
        c4.metric("最大持仓", f"{max_positions} 只")
        
        if st.button("📈 执行回测"):
            # 过滤出回测期的数据
            backtest_data_map = {}
            for code, df in st.session_state['data_map'].items():
                # 筛选出分割日期之后的数据
                mask = df.index >= pd.to_datetime(split_date)
                if mask.any():
                    backtest_data_map[code] = df[mask]
            
            if not backtest_data_map:
                st.error("回测数据集为空！请检查分割日期设置。")
            else:
                # 实例化回测引擎
                bt = Backtester(
                    list(backtest_data_map.keys()), 
                    split_date.strftime("%Y%m%d"), 
                    end_date.strftime("%Y%m%d"), 
                    initial_capital,
                    stop_loss=stop_loss_pct,
                    take_profit=take_profit_pct,
                    max_positions=max_positions,
                    rebalance_days=rebalance_days,
                    commission_rate=commission_rate,
                    min_commission=min_commission,
                    stamp_duty_rate=stamp_duty_rate,
                    slippage_rate=slippage_rate
                )
                
                with st.spinner("正在逐日模拟交易..."):
                    res, transactions = bt.run_with_data(backtest_data_map, st.session_state['trained_model'])
                
                if not res.empty:
                    # 1. 净值曲线
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=res.index, y=res['value'], mode='lines', name='策略净值', line=dict(color='#00ba38', width=2)))
                    fig.update_layout(title="账户权益曲线", xaxis_title="日期", yaxis_title="资产净值 (元)", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 2. 核心指标
                    final_val = res['value'].iloc[-1]
                    ret = (final_val - initial_capital) / initial_capital
                    max_dd = calculate_max_drawdown(res['value'])
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("最终资产", f"¥{final_val:,.2f}")
                    m2.metric("总收益率", f"{ret:.2%}", delta_color="normal")
                    m3.metric("最大回撤", f"{max_dd:.2%}")
                    
                    # 3. 交易明细
                    st.markdown("### 📋 交易明细记录")
                    if not transactions.empty:
                        # 格式化显示
                        transactions['价格'] = transactions['价格'].apply(lambda x: f"¥{x:.2f}")
                        transactions['金额'] = transactions['金额'].apply(lambda x: f"¥{x:,.2f}")
                        transactions['手续费'] = transactions['手续费'].apply(lambda x: f"¥{x:.2f}")
                        transactions['印花税'] = transactions['印花税'].apply(lambda x: f"¥{x:.2f}")
                        st.dataframe(transactions, use_container_width=True)
                    else:
                        st.info("回测期间未触发任何交易。")
                        
                else:
                    st.warning("回测期间无交易产生，可能是选股标准太严或数据不足。")
