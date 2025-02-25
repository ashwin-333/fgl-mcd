# Future Guided Learning

## Project Structure

```plaintext
├── baseline-model/
│   ├── main.py
│   ├── visualize.py
│   ├── utils.py
├── rnn/
│   ├── main.py
│   ├── visualize.py
│   ├── utils.py
├── models/
|   ├── layers/
|   |   ├── AutoCorrelation.py
|   |   ├── Autoformer_EncDec.py
|   |   ├── Embed.py
|   |   ├── PatchTST_backbone.py
|   |   ├── PatchTST_layers.py
|   |   ├── RevIN.py
│   ├── Autoformer.py
│   ├── LSTM.py
│   ├── PatchTST.py
│   ├── S-Mamba.py
├── datasets/
│   ├── traffic.zip
│   ├── electricity.zip
│   ├── exchange.zip
│   ├── weather.zip