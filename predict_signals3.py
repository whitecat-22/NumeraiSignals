import os
import time
import pytz
import traceback
import pandas as pd
import numerapi
import sys
# from google.colab import drive
# import datetime
#!pip install pytrends numerapi
import pytrends
from pytrends.request import TrendReq

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
notification_message = jst_nowtime + '\n' + "NumeraiSignals3:予測開始"
line_post(notification_message)


try:
    today = datetime.now().strftime('%Y%m%d')
    hdf_name = f'trend_{today}.h5'
    hdf_name

    """
    drive.mount('/content/drive')
    %cd "/content/drive/My Drive/signals"
    sys.path.append('/content/drive/My Drive/signals')
    """

    napi = numerapi.SignalsAPI()

    eligible_tickers = pd.Series(napi.ticker_universe(), name='numerai_ticker')

    ticker_map = pd.read_csv(
        'https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv')

    trend_search_map = pd.read_csv('trend_search_list.csv')

    #trend_search_map

    #ticker_map

    trend_search_map = pd.merge(ticker_map, trend_search_map, on='ticker')

    #trend_search_map

    trend_dfs = []

    for x in range(len(trend_search_map)):
        try:
            df = pd.read_hdf(hdf_name, f'trenddata_{x}')
            trend_dfs.append(df)
        except:
            break

    ticker2bloomberg = {}
    for x in ticker_map.itertuples():
        # print(x.ticker, x.bloomberg_ticker)
        ticker2bloomberg[x.ticker] = x.bloomberg_ticker

    pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25), retries=2,
                        backoff_factor=0.1)  # requests_args={'verify':False}
    # pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25), retries=10, backoff_factor=1) # requests_args={'verify':False}

    have_map = {}
    for df in trend_dfs:
        have_map[df.columns[0]] = True

    for x in trend_search_map.itertuples():
        #     print(x)
        if x.search_word in ['NASDAQ:CZR']:
            print(f'skip {x.search_word}')
            continue
        if have_map.get(x.ticker) is None:
            ok = False
            print(x.search_word, x.trend_topics)
            #         print(x)
            for trycount in range(5):
                try:
                    kw_list = [x.trend_topics]
                    # time.sleep(0.5)
                    time.sleep(2)
                    pytrends.build_payload(
                        kw_list=kw_list, timeframe='today 12-m', geo='', gprop='', cat=0)
                    z = pytrends.interest_over_time()
                    if len(z) > 0:
                    #             print(z)
                        z = z[x.trend_topics]
                        z = z.rename(x.ticker)
                        z = z.to_frame()
                        trend_dfs.append(z)
                    ok = True
                    break
                except Exception as e:
                    print(e)
            if not ok:
                continue
                #save_trends()
                #gotofail()
    #     break
    trend_dfs[-1]

    #save_trends()

    signals_dfs = []
    for df in trend_dfs:
        ticker = df.columns[0]
        print(ticker)
        df = df.rename(columns={ticker: 'trend'})
        df = df.replace(0, 1)
        df['trend_ave13'] = df.rolling(13).mean()
        df['trend_kairi13'] = (df['trend'] - df['trend_ave13']) / df['trend_ave13']
        df['ticker'] = ticker
        df['numerai_ticker'] = ticker2bloomberg[ticker]
        df['friday_date'] = df.index + datetime.timedelta(days=5)
        today = df.tail(1)['friday_date'][0]
        df['data_type'] = df['friday_date'].apply(
            lambda x: 'live' if x == today else 'validation')
        signals_dfs.append(df)
    # df

    signal_df = pd.concat(signals_dfs)
    signal_df = signal_df.dropna()
    signal_df = signal_df.sort_index()
    # signal_df

    signal_df = signal_df.reset_index()

    def make_rank(pred):
        return pred.rank(pct=True, method="first")
    signal_df['signal'] = signal_df.groupby('date')['trend_kairi13'].apply(make_rank)
    # signal_df

    signal_df['friday_date'] = signal_df['friday_date'].dt.strftime('%Y%m%d')

    signal_df[['numerai_ticker', 'friday_date', 'data_type', 'signal']]

    signal_df[['numerai_ticker', 'friday_date', 'data_type', 'signal']].to_csv(
        'trend_signal_upload.csv', index=False)

    public_id = os.environ.get('PUBLIC_ID3')
    secret_key = os.environ.get('SECRET_KEY3')
    model_id = os.environ.get('MODEL_ID3')
    napi = numerapi.SignalsAPI(public_id=public_id, secret_key=secret_key)
    napi.upload_predictions('trend_signal_upload.csv', model_id=model_id)

   # LINE通知（終了：成功）
    notification_message = jst_nowtime + '\n' + "NumeraiSignals3:予測提出完了"
    line_post(notification_message)

except:
    # LINE通知（終了：失敗）
    notification_message = jst_nowtime + '\n' + \
        "NumeraiSignals3:予測失敗" + '\n\n' + traceback.format_exc()
    line_post(notification_message)
