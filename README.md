# NumeraiSignals


[Numerai Signals](https://signals.numer.ai/tournament) 提出用の株価予測は、WebhookからAWS APIGwatwayを通じて、Lambda経由でEC2(g4dn.xlearge)を起動。  
EC2起動時に、インスタンス内： home/(user)/numerai/predict.py を起動する。

　

- 参照元：  
[AWS EC2 で Numerai Compute](https://zenn.dev/kunigaku/articles/50c079b033e6051bc764)

　

- 株価取得用の参考データ：
[YFinance Stock Price Data for Numerai Signals](https://www.kaggle.com/code1110/yfinance-stock-price-data-for-numerai-signals)

　

<a href="https://signals.numer.ai/tournament">
  <img src="https://github.com/whitecat-22/NumeraiSignals/blob/main/signals-logo-white.6b048f21.png">
</a>
