from api_wrappers.kraken_wrapper import KrakenWrapper

wrapper = KrakenWrapper(lb_interval="1day")
wrapper.pull_kraken_hist_usd()