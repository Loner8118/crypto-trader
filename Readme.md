Project Overview:-
This project analyzes the relationship between market sentiment (Fear & Greed Index) and crypto trader behavior.

The objective was to:
-Clean and align sentiment + trading datasets
-Evaluate trader performance across sentiment regimes
-Develop actionable strategy recommendations
-Build a simple predictive model
-Create an interactive Streamlit dashboard
-The final output includes reproducible analysis and an interactive dashboard.

📂 Repository Structure
crypto-trader-sentinel/
│
├── notebook/
│   └── analysis.ipynb
│
├── data/
│   ├── fear_greed_index.csv
│   ├── overall_trader_perf.csv
│   ├── sentiment_pnl.csv
│   └── final_df.csv
│   └── historical_data.csv
│
├── app.py
├── requirements.txt
└── README.md

⚙️ Setup Instructions
1️⃣ Clone Repository
git clone https://github.com/Loner8118/crypto-trader.git
cd crypto-trader
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Notebook
Open:
notebook/analysis.ipynb
(Run all cells from top to bottom.)

4️⃣ Run Interactive Dashboard
streamlit run app.py

Dashboard will open at:
http://localhost:8501

Methodology
1. Data Cleaning & Alignment:-
-Removed missing values
-Standardized date formats
-Converted timestamps to daily frequency
-Merged trader performance with sentiment data using date alignment
-Used relative file paths for reproducibility

Created:
-Daily PnL per trader
-Win rate per trader
-Sentiment regime buckets
-Normalized PnL scores
-Composite trader performance score

Scoring formula:
-Score = 0.7 × Normalized PnL + 0.3 × Win Rate

2. Sentiment Regime Analysis:-
-Market regimes categorized as:
-Extreme Fear
-Fear
-Neutral
-Greed
-Extreme Greed
-Performance metrics were compared across regimes.

3. Predictive Modeling:-
-Built a classification model to predict:
-Next-Day Profitability Bucket

Features used:
-Current day PnL
-Sentiment score
-Behavioral statistics

Model:
-Random Forest Classifier

Objective:

Predict P(Profitability Bucket | Sentiment + Behavior)
🔍 Key Insights
1️⃣ Profitability is Driven by Sentiment Intensity

Extreme Fear and Extreme Greed periods showed the highest dispersion in trader returns.
Volatility intensity had stronger predictive power than sentiment direction alone.

2️⃣ Trader Skill is Regime-Dependent

Some traders outperform during crisis conditions (Fear).
Others perform better during Greed-driven markets.
Performance is not uniform across market regimes.

3️⃣ Neutral Markets Produce Lower Edge

During neutral sentiment periods, average PnL declines, indicating lower opportunity.

🚀 Strategy Recommendations
Strategy 1 — Volatility Amplification Allocation

Increase capital exposure during extreme sentiment regimes (|sentiment| high).

Reduce exposure during neutral regimes.

Rationale: Volatility creates opportunity dispersion.

Strategy 2 — Regime-Based Capital Rotation

Allocate capital dynamically to trader segments based on regime:

Fear → Crisis Alpha traders

Greed → Momentum traders

Neutral → Balanced / conservative traders

This improves capital efficiency across cycles.

📈 Dashboard Features

The Streamlit dashboard provides:

Interactive trader leaderboard

Sentiment vs PnL visualization

Trader-level performance tracking

Distribution of PnL across traders

Dynamic filtering by date and sentiment

📌 Reproducibility

Uses relative paths

requirements.txt included

All processed datasets included in data/ folder

Notebook runs top-to-bottom without modification

🧠 Conclusion

This analysis demonstrates that trader profitability is significantly influenced by sentiment regimes and volatility intensity.

By combining sentiment analytics with behavioral scoring, adaptive capital allocation strategies can be developed.

The project integrates:

Data cleaning
Behavioral analytics
Strategy design
Predictive modeling

Dashboard deployment

👤 Author

Your Name- Tanish AKre
Email: tanishakre15@gmail.com
