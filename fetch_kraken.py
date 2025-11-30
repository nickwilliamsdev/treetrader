from api_wrappers.kraken_wrapper import KrakenWrapper

wrapper = KrakenWrapper(lb_interval="1week")
#wrapper.pull_kraken_hist_usd()
wrapper.update_or_create_hist_files()