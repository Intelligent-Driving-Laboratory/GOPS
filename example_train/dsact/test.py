import pandas as pd
import matplotlib.pyplot as plt
import argparse

from finrl.meta.preprocessor.preprocessors import data_split,FeatureEngineer
from Alpha101_code_1 import get_alpha
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import PPO
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.config import INDICATORS
from finrl.plot import backtest_stats

# Contestants are welcome to split the data in their own way for model tuning
TRADE_START_DATE = '2020-07-01'
TRADE_END_DATE = '2023-10-24'
FILE_PATH = '/home/qinshentao/code/FinRL-Contest/task-1-stock-trading-starter-kit/train_data.csv'
INDICATORS_processed = ['macd_x', 'boll_ub_x', 'boll_lb_x', 'rsi_30_x', 'cci_30_x',
       'dx_30_x', 'close_30_sma_x', 'close_60_sma_x', 'vix_x', 'turbulence_x',
       'macd_y', 'boll_ub_y', 'boll_lb_y', 'rsi_30_y', 'cci_30_y', 'dx_30_y',
       'close_30_sma_y', 'close_60_sma_y','vix_y', 'turbulence_y','Returns',
       'Typical_Price', 'VWAP']


# PPO configs
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.0003,
    "batch_size": 128,
}


if __name__ == '__main__':
    # We will use unseen, post-deadline data for testing
    parser = argparse.ArgumentParser(description='Description of program')
    parser.add_argument('--start_date', default=TRADE_START_DATE, help='Trade start date (default: {})'.format(TRADE_START_DATE))
    parser.add_argument('--end_date', default=TRADE_END_DATE, help='Trade end date (default: {})'.format(TRADE_END_DATE))
    parser.add_argument('--data_file', default=FILE_PATH, help='Trade data file')
    parser.add_argument('--model_path', default='/2023-11-12 02:12:26.401475/trained_ppo_alpha', help='Trade data file')

    args = parser.parse_args()
    TRADE_START_DATE = args.start_date
    TRADE_END_DATE = args.end_date
    
    # data = pd.read_csv(args.data_file)
    # fe = FeatureEngineer(use_technical_indicator=True,
    #             tech_indicator_list = INDICATORS,
    #             use_vix=False,
    #             use_turbulence=True,
    #             user_defined_feature = False)
    # processed = fe.preprocess_data(data)
    # alpha = get_alpha(processed)
    # print(alpha.head)
    # # list_ticker = processed["tic"].unique().tolist()
    # # list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    # # combination = list(itertools.product(list_date,list_ticker))
    # # processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    # processed_full = alpha
    # processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    # processed_full = processed_full.sort_values(['date','tic'])
    # processed_full = processed_full.fillna(0)
    
    filename = './processed_full.pkl'
    import pickle
    # 使用 'rb' 模式打开文件（'rb' 代表 "read binary"）
    with open(filename, 'rb') as file:
        # 反序列化对象
        data = pickle.load(file)
    processed_full = data

    trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
    
    stock_dimension = len(trade.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS_processed)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    # please do not change initial_amount
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS_processed,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    check_and_make_directories([TRAINED_MODEL_DIR])

    # Environment
    e_trade_gym = StockTradingEnv(df = trade, **env_kwargs)
    
    # PPO agent
    agent = DRLAgent(env = e_trade_gym)
    model_ppo = agent.get_model("ppo", model_kwargs = PPO_PARAMS)
    trained_ppo = PPO.load(TRAINED_MODEL_DIR + args.model_path)

    # Backtesting
    df_result_ppo, df_actions_ppo = DRLAgent.DRL_prediction(model=trained_ppo, environment = e_trade_gym)

    print("==============Get Backtest Results===========")
    perf_stats_all = backtest_stats(account_value=df_result_ppo)
    
    """Plotting"""
    plt.rcParams["figure.figsize"] = (15,5)
    plt.figure()
    
    df_result_ppo.plot()
    plt.savefig("plot.png")
    
    df_result_ppo.to_csv("results.csv", index=False)