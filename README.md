# NumeraiSignals


・[Numerai Signals](https://signals.numer.ai/tournament) 提出用の株価予測は、Webhookから[AWS APIGwatway](https://aws.amazon.com/jp/api-gateway/?nc2=h_ql_prod_serv_apig)を通じて、  
　[Lambda](https://aws.amazon.com/jp/lambda/?nc2=h_ql_prod_serv_lbd)経由で[EC2](https://aws.amazon.com/jp/ec2/?nc2=h_ql_prod_fs_ec2)(g4dn.xlearge)を起動する。  
・EC2起動時に、インスタンス内： home/(user)/numerai/predict_signals.py を起動する。

　

- 参照元：  
[AWS EC2 で Numerai Compute](https://zenn.dev/kunigaku/articles/50c079b033e6051bc764)

　

- 株価取得用の参考データ：  
[YFinance Stock Price Data for Numerai Signals](https://www.kaggle.com/code1110/yfinance-stock-price-data-for-numerai-signals)

　

<a href="https://signals.numer.ai/tournament">
  <img src="https://github.com/whitecat-22/NumeraiSignals/blob/main/signals-logo-white.6b048f21.png">
</a>
