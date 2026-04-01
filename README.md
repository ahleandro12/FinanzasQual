# FinanzasQual

📈 FinDash — Personal Finance Dashboard
A personal project built out of curiosity and the need to have all my investments in one place.
I'm an international trade professional who wanted to track his CEDEAR portfolio without switching between 5 different tabs. So I built this.

🧠 What is this?
FinDash is a personal finance dashboard built with Python and Streamlit. It pulls real-time market data and lets you visualize your portfolio with the same technical indicators that professional traders use — even if, like me, you had to Google what RSI meant before implementing it.

⚠️ Note: The portfolio data shown in the app is fictional. Real personal financial data is not included for privacy reasons.


✨ Features

📊 Candlestick charts — visualize price movements over time
📉 RSI (Relative Strength Index) — momentum indicator
📈 MACD (Moving Average Convergence Divergence) — trend indicator
📐 Bollinger Bands — volatility indicator
💼 CEDEAR Portfolio tracker — Argentine market instruments
☁️ Google Sheets backend — portfolio data stored and managed in the cloud
⚡ Real-time data — powered by yfinance API


🛠️ Tech Stack
ToolPurposePythonCore languageStreamlitWeb app frameworkyfinanceReal-time market dataPandasData manipulationPlotlyInteractive chartsGoogle Sheets APIPortfolio data backend

🚀 Live Demo
👉 Open the app on Streamlit Cloud

The app may take a few seconds to load if it's been inactive — just click "Wake up" and it'll be ready in 30 seconds.


💡 Why I built this
I invest in CEDEARs through Bull Market and I got tired of checking prices manually across different platforms. I wanted one single view with everything integrated.
I had no background in financial analysis — I learned about RSI, MACD and Bollinger Bands while building this. The math is handled by the libraries; what I focused on was making it usable and personal.
It's not perfect. But it's mine, it works, and I use it.

📁 Project Structure
FinanzasQual/
├── app.py              # Main Streamlit app
├── requirements.txt    # Dependencies
└── README.md           # This file

🔧 Run locally
bashgit clone https://github.com/ahleandro12/FinanzasQual
cd FinanzasQual
pip install -r requirements.txt
streamlit run app.py

👤 Author
Leandro Apodaca Aleman
Supply Chain & International Trade Specialist | Buenos Aires, Argentina
LinkedIn
