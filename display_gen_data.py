from utils.synthetic_data_service import SyntheticWienerGenerator, SyntheticOHLCVGenerator
from matplotlib import pyplot as plt
ohlc_gen_service = SyntheticOHLCVGenerator(n_steps=1000, mu=0.005, sigma=0.1, dt=1, seed=42)

test_df = ohlc_gen_service.generate(start=100.0)

plt.plot(test_df['time'], test_df['close'], label='Synthetic Close Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Synthetic OHLCV Data')
plt.legend()
plt.show()