import gc
import os
import time
import pytz
import traceback
import numerapi
import yfinance
import simplejson
from catboost import CatBoost

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import requests
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta, FR


# line notify APIのトークン
line_notify_token = os.environ.get("LINE_NOTIFY_TOKEN")
# line notify APIのエンドポイントの設定
line_notify_api = 'https://notify-api.line.me/api/notify'

jst_nowtime = ""


def line_post(notification_message):
    # 現在時刻
    now = datetime.now(tz=timezone.utc)
    tokyo = pytz.timezone('Asia/Tokyo')
    # 東京のローカル時間に変換
    jst_now = tokyo.normalize(now.astimezone(tokyo))
    jst_nowtime = jst_now.strftime("%m/%d %H:%M")
    # ヘッダーの指定
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    # 送信するデータの指定
    data = {'message': f'{notification_message}'}

    # line notify apiにpostリクエストを送る
    requests.post(line_notify_api, headers=headers, data=data)


# notification_message = content0 +'\n' + '\n\n'.join(content_text)
notification_message = jst_nowtime + '\n' + "NumeraiSignals:予測開始"
line_post(notification_message)


try:
    """
    Catboostを呼び出しnumeraisignalsに予測を提出する
    """
    #!pip install numerapi
    #!pip install yfinance
    #!pip install simplejson
    #
    #!pip install catboost


    # Tickers that Numerai signals want. These are bloomberg tickers. yfinance asks for yahoo finance tickers.

    # Data acquisition
    napi = numerapi.SignalsAPI()

    eligible_tickers = pd.Series(napi.ticker_universe(), name="bloomberg_ticker")
    print(f"Number of eligible tickers : {len(eligible_tickers)}")

    print(eligible_tickers.head(10))

    # This file has mapping from bloomberg to yahoo finance tickers. So, we can use yfinance tickers to download and then map/rename them back to bloomberg tickers.

    ticker_map = pd.read_csv(
        'https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv'
    )

    print(len(ticker_map))

    ticker_map.head()

    #Yahoo <-> Bloomberg mapping
    yfinance_tickers = eligible_tickers.map(
        dict(zip(ticker_map["bloomberg_ticker"], ticker_map["yahoo"]))
    ).dropna()

    bloomberg_tickers = ticker_map["bloomberg_ticker"]

    print(f"Number of eligible, mapped tickers: {len(yfinance_tickers)}")

    # These are tickers that Numerai signals wants and are also in the mapping dictionary

    def get_prices(tickers, threads=False, n=1000):

        chunk_df = [
            tickers.iloc[i:i+n]
            for i in range(0, len(tickers), n)
        ]

        concat_dfs = []

        for chunk in chunk_df:
            try:
                #['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

                temp_df = yfinance.download(chunk.str.cat(sep=' '),
                                            start='2005-12-01',
                                            threads=threads)

                #We'll use adjusted close here
                temp_df = temp_df['Adj Close'].stack().reset_index()
                concat_dfs.append(temp_df)

            except:
                pass

        return pd.concat(concat_dfs)

    #this will take some time
    full_data = get_prices(yfinance_tickers)

    full_df = full_data.copy()
    full_df.columns = ['date', 'ticker', 'price']
    full_df.set_index('date', inplace=True)

    full_df['ticker'] = full_df.ticker.map(
        dict(zip(ticker_map["yahoo"], bloomberg_tickers))
    )


    full_df.to_csv("full_data.csv")


    #A day's data
    full_df.groupby("date").get_group("2005-12-01")


    print(f"Number of tickers with data: {len(full_df.ticker.unique())}")


    # Technical Indicators

    def RSI(prices, interval=14):
        '''Computes Relative Strength Index given a price series and lookback interval
        Modified from https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas
        See more here https://www.investopedia.com/terms/r/rsi.asp'''
        delta = prices.diff()

        dUp, dDown = delta.copy(), delta.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0

        RolUp = dUp.rolling(interval).mean()
        RolDown = dDown.rolling(interval).mean().abs()

        RS = RolUp / RolDown
        RSI = 100.0 - (100.0 / (1.0 + RS))
        return RSI


    def sma(prices, window=10):

        return (sum(prices, window))/window


    ticker_groups = full_df.groupby('ticker')
    full_df["RSI"] = ticker_groups["price"].transform(lambda x: RSI(x))
    full_df["SMA_10"] = ticker_groups["price"].transform(lambda x: sma(x, 10))

    #a list of tickers used in full_df.columns to ease the calculation.
    indicators = ['RSI', 'SMA_10']

    full_df

    """
    plt.plot(full_df[full_df['ticker'] == "1 HK"]["RSI"][-100:], label='RSI')
    plt.plot(full_df[full_df['ticker'] == "1 HK"]["price"][-100:], label='price')
    plt.legend(loc='upper left')

    plt.show()
    """

    # %%time

    date_groups = full_df.groupby(full_df.index)

    for indicator in indicators:
        print(indicator)
        full_df[f"{indicator}_quintile"] = date_groups[indicator].transform(
            lambda group: pd.qcut(group, 5, labels=False, duplicates='drop')
        ).astype(np.float32)
        gc.collect()

    full_df.head()


    # Let's encode historical RSI into features

    ticker_groups = full_df.groupby("ticker")

    #create lagged features, lag 0 is that day's value, lag 1 is yesterday's value, etc

    for indicator in indicators:
        num_days = 5
        for day in range(num_days + 1):
            full_df[f"{indicator}_quintile_lag_{day}"] = ticker_groups[
                f"{indicator}_quintile"
            ].transform(lambda group: group.shift(day))

        gc.collect()

    full_df.tail()

    full_df[full_df["ticker"] == "ZEL NZ"]

    feature_names = [f for f in full_df.columns for y in ['lag', 'diff'] if y in f]

    # Historical Numerai Targets
    TARGET_NAME = "target"
    PREDICTION_NAME = "signal"

    url = "https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_train_val.csv"
    targets = pd.read_csv(url)
    targets.head()

    targets['date'] = pd.to_datetime(targets['friday_date'], format='%Y%m%d')

    targets[targets["data_type"] == "train"]

    targets[targets["data_type"] == "validation"]

    targets.target.value_counts()

    ML_data = pd.merge(full_df.reset_index(), targets,
            on=["date", "ticker"]).set_index("date")

    ML_data.dropna(inplace=True)
    ML_data = ML_data[ML_data.index.weekday==4]
    ML_data = ML_data[ML_data.index.value_counts() > 200]

    gc.collect()

    print(f'Number of eras in data: {len(ML_data.index.unique())}')

    ML_data

    gc.collect()

    # Modelling
    train_data = ML_data[ML_data['data_type'] == 'train']
    test_data = ML_data[ML_data['data_type'] == 'validation']
    gc.collect()

    feature_names = [f for f in train_data.columns for y in ['lag', 'diff'] if y in f]

    params = {
        "objective": "RMSE",
        "iterations": 1000,
        "task_type": "GPU"
    }
    model = CatBoost(params)

    gc.collect()

    model.fit(train_data[feature_names],
            train_data[TARGET_NAME],
            eval_set=(test_data[feature_names],
                        test_data[TARGET_NAME]))

    gc.collect()

    """
    plt.figure(figsize=(15, 3))
    plt.bar(feature_names, model.feature_importances_)
    plt.xticks(rotation=70)
    plt.show()
    """

    # Evaluation on historic data
    train_data[PREDICTION_NAME] = model.predict(train_data[feature_names])
    test_data[PREDICTION_NAME] = model.predict(test_data[feature_names])

    #show prediction distribution, most should around the center
    test_data[PREDICTION_NAME].hist(bins=30)

    #From Jason Rosenfeld's notebook
    #https://twitter.com/jrosenfeld13/status/1315749231387443202?s=20

    def score(df):
        '''Takes df and calculates spearm correlation from pre-defined cols'''
        # method="first" breaks ties based on order in array
        return np.corrcoef(
            df[TARGET_NAME],
            df[PREDICTION_NAME].rank(pct=True, method="first")
        )[0,1]

    def run_analytics(era_scores):
        print(f"Mean Correlation: {era_scores.mean():.4f}")
        print(f"Median Correlation: {era_scores.median():.4f}")
        print(f"Standard Deviation: {era_scores.std():.4f}")
        print('\n')
        print(f"Mean Pseudo-Sharpe: {era_scores.mean()/era_scores.std():.4f}")
        print(f"Median Pseudo-Sharpe: {era_scores.median()/era_scores.std():.4f}")
        print('\n')
        print(f'Hit Rate (% positive eras): {era_scores.apply(lambda x: np.sign(x)).value_counts()[1]/len(era_scores):.2%}')

        era_scores.rolling(10).mean().plot(kind='line', title='Rolling Per Era Correlation Mean', figsize=(15,4))
        plt.axhline(y=0.0, color="r", linestyle="--"); plt.show()

        era_scores.cumsum().plot(title='Cumulative Sum of Era Scores', figsize=(15,4))
        plt.axhline(y=0.0, color="r", linestyle="--"); plt.show()


    # spearman scores by era
    train_era_scores = train_data.groupby(train_data.index).apply(score)
    test_era_scores = test_data.groupby(test_data.index).apply(score)

    #train scores, in-sample and will be significantly overfit
    run_analytics(train_era_scores)

    #test scores, out of sample
    run_analytics(test_era_scores)


    # Prediction on live data

    # choose data as of most recent friday
    last_friday = datetime.now() + relativedelta(weekday=FR(-1))
    date_string = last_friday.strftime('%Y-%m-%d')

    live_data = full_df.loc[date_string].copy()
    live_data.dropna(subset=feature_names, inplace=True)

    live_data

    print(f"Number of live tickers to submit: {len(live_data)}")
    live_data[PREDICTION_NAME] = model.predict(live_data[feature_names])

    live_data[PREDICTION_NAME].hist()

    # You can simply run this without any modification
    diagnostic_df = pd.concat([test_data, live_data])
    diagnostic_df['friday_date'] = diagnostic_df.friday_date.fillna(
        last_friday.strftime('%Y%m%d')).astype(int)
    diagnostic_df['data_type'] = diagnostic_df.data_type.fillna('live')
    diagnostic_df[['ticker', 'friday_date', 'data_type',
                    'signal']].reset_index(drop=True).to_csv(
        'example_signal_upload.csv', index=False)
    """
    print(
        'Example submission completed. Upload to signals.numer.ai for scores and live submission'
    )
    """

    diagnostic_df

    ps = diagnostic_df.groupby('date')[PREDICTION_NAME].rank(
        pct=True, method="first")

    diagnostic_df["signal"] = ps

    diagnostic_df[["ticker", "friday_date", "data_type", "signal"]].reset_index(
        drop=True
    ).to_csv("example_signal_upload.csv", index=False)

    """
    print("Upload to signals.numer.ai for scores and live submission")
    """

    # Uploading predictions using your API keys

    # Find your Numerapi public and private keys from https://numer.ai/account

    # NameOfYourAI
    # Add keys between the quotes
    public_id = os.environ.get('PUBLIC_ID')
    secret_key = os.environ.get('SECRET_KEY')
    model_id = os.environ.get('MODEL_ID')
    napi = numerapi.SignalsAPI(public_id=public_id, secret_key=secret_key)


    submission_id = napi.upload_predictions(
        f"/content/example_signal_upload.csv", model_id=model_id)

    # And its done. Congratulations. Your predictions for latest round are submitted!
    #
    # Check some information about your latest predictions on[Signals Tournament](https: // signals.numer.ai/tournament). It will show some metrics like this,

    # LINE通知（終了：成功）
    notification_message = jst_nowtime + '\n' + "NumeraiSignals:予測提出完了"
    line_post(notification_message)

except:
    # LINE通知（終了：失敗）
    notification_message = jst_nowtime + '\n' + \
        "NumeraiSignals:予測失敗" + '\n\n' + traceback.format_exc()
    line_post(notification_message)
