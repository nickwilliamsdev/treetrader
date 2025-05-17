from api_wrappers.kraken_wrapper import KrakenWrapper

wrapper = KrakenWrapper(lb_interval="4hr")
wrapper.pull_kraken_hist_usd()