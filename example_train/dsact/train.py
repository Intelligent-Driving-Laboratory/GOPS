import pandas as pd
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from Alpha101_code_1 import get_alpha
import itertools

INDICATORS_processed = ['macd_x', 'boll_ub_x', 'boll_lb_x', 'rsi_30_x', 'cci_30_x',
       'dx_30_x', 'close_30_sma_x', 'close_60_sma_x', 'vix_x', 'turbulence_x',
       'macd_y', 'boll_ub_y', 'boll_lb_y', 'rsi_30_y', 'cci_30_y', 'dx_30_y',
       'close_30_sma_y', 'close_60_sma_y','vix_y', 'turbulence_y','Returns',
       'Typical_Price', 'VWAP']

# INDICATORS_processed = ['macd_x', 'boll_ub_x', 'boll_lb_x', 'rsi_30_x', 'cci_30_x',
#        'dx_30_x', 'close_30_sma_x', 'close_60_sma_x',  'turbulence_x',
#        'macd_y', 'boll_ub_y', 'boll_lb_y', 'rsi_30_y', 'cci_30_y', 'dx_30_y',
#        'close_30_sma_y', 'close_60_sma_y', 'turbulence_y','Returns',
#        'Typical_Price', 'VWAP']

# Contestants are welcome to split the data in their own way for model tuning
TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2020-06-30'

# data = pd.read_csv('/home/qinshentao/code/FinRL-Contest/task-1-stock-trading-starter-kit/train_data.csv')
# ## feature
# fe = FeatureEngineer(use_technical_indicator=True,
#                 tech_indicator_list = INDICATORS,
#                 use_vix=False,
#                 use_turbulence=True,
#                 user_defined_feature = False)
# processed = fe.preprocess_data(data)
# alpha = get_alpha(processed)
# print(alpha.head)
# list_ticker = alpha["tic"].unique().tolist()
# list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
# combination = list(itertools.product(list_date,list_ticker))
# processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
# processed_full = processed_full
# processed_full = processed_full[processed_full['date'].isin(processed['date'])]
# processed_full = processed_full.sort_values(['date','tic'])
# processed_full = processed_full.fillna(0)
# 假设 'example.pkl' 是你的 .pkl 文件
filename = './processed_full.pkl'
import pickle
# 使用 'rb' 模式打开文件（'rb' 代表 "read binary"）
with open(filename, 'rb') as file:
    # 反序列化对象
    data = pickle.load(file)
processed_full = data
train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)

# Environment configs
stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS_processed)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

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


# PPO configs
DSAC_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.0003,
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
    "trainer":"off_serial_trainer",
    "cnn_shared":True,
    'obsv_dim':state_space,
    "action_num": stock_dimension,
    "action_dim":stock_dimension,
    "action_type":"continu",'algorithm': 'DSACT', 'enable_cuda': False, 'reward_scale': 0.2, 'is_render': True, 'is_adversary': False, 'value_func_name': 'ActionValueDistri', 'value_func_type': 'MLP', 'value_hidden_sizes': [256, 256, 256], 'value_hidden_activation': 'gelu', 'value_output_activation': 'linear', 'policy_func_name': 'StochaPolicy', 'policy_func_type': 'MLP', 'policy_act_distribution': 'TanhGaussDistribution', 'policy_hidden_sizes': [256, 256, 256], 'policy_hidden_activation': 'gelu', 'policy_min_log_std': -20, 'policy_max_log_std': 0.5, 'value_learning_rate': 0.0001, 'policy_learning_rate': 0.0001, 'alpha_learning_rate': 0.0003, 'gamma': 0.99, 'tau': 0.005, 'auto_alpha': True, 'alpha': 0.2, 'delay_update': 2, 'trainer': 'off_serial_trainer', 'max_iteration': 1500000, 'ini_network_dir': None, 'buffer_name': 'replay_buffer', 'buffer_warm_size': 10000, 'buffer_max_size': 1000000, 'replay_batch_size': 256, 'sample_interval': 1, 'sampler_name': 'off_sampler', 'sample_batch_size': 20, 'noise_params': None, 'evaluator_name': 'evaluator', 'num_eval_episode': 10, 'eval_interval': 2500, 'eval_save': False, 'save_folder': None, 'apprfunc_save_interval': 50000, 'log_save_interval': 10000}

if __name__ == '__main__':
    check_and_make_directories([TRAINED_MODEL_DIR])
    # Environment
    e_train_gym = StockTradingEnv(df = train,turbulence_threshold = 70,**env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))

    # PPO agent
    agent = DRLAgent(env = env_train)
    
    model_ppo = agent.get_model("DSACT",model_kwargs = DSAC_PARAMS)
    import datetime

    # 获取当前时间
    now = datetime.datetime.now()
    # set up logger
    tmp_path = RESULTS_DIR + '/ppo_alpha101' + f'{now}'
    new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_ppo.set_logger(new_logger_ppo)
    
    trained_ppo = agent.train_model(model=model_ppo,
                                tb_log_name='ppo_alpha101',
                                total_timesteps=80000)
    
    trained_ppo.save(TRAINED_MODEL_DIR + f'/{now}' + '/trained_ppo_alpha')
