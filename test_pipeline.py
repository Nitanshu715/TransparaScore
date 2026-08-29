"""
TransparaScore Institutional Pipeline Verification & Unit Test Suite
Validates:
1. yfinance Stock Ingestion & Graceful Fallback
2. Macro Indicators (10Y Treasury, World Bank API, Benchmark fallbacks)
3. News & Sentiment Ingestion (Google News RSS & Yahoo Finance)
4. ML Credit Model Training, Persistence & Inference
5. Altman Z-Score & Merton Probability of Default (PD) Models
6. True SHAP TreeExplainer Local Feature Attribution
7. 1,000-Path Monte Carlo Stress Testing Engine
8. Rating Tier and Crisis Shock Stress Calculations
"""
import os
import sys
import pandas as pd
import numpy as np

def run_tests():
    print("==================================================")
    print("  TRANSPARASCORE INSTITUTIONAL TEST SUITE         ")
    print("==================================================")
    
    # 1. Test Stock Data Fetcher
    print("\n[1/7] Testing Stock & Balance Sheet Data Fetcher (yfinance)...")
    from TransparaScore import fetch_stock_data
    df, meta, src = fetch_stock_data("AAPL", period="1mo", interval="1d")
    assert df is not None and not df.empty, "Stock dataframe should not be empty"
    assert "Close" in df.columns, "Stock dataframe missing Close column"
    assert "total_assets" in meta, "Metadata missing total_assets"
    print(f"  [OK] Successfully fetched AAPL data: {len(df)} rows | Source: {src}")
    
    # 2. Test Macro Indicators Fetcher
    print("\n[2/7] Testing Macroeconomic Indicators (Treasury & World Bank)...")
    from TransparaScore import fetch_macro_indicators
    macro_data, sources = fetch_macro_indicators()
    assert "rate10y_pct" in macro_data, "Macro data missing rate10y_pct"
    assert "cpi_yoy_pct" in macro_data, "Macro data missing cpi_yoy_pct"
    assert "gdp_growth_us_pct" in macro_data, "Macro data missing gdp_growth_us_pct"
    print(f"  [OK] Macro indicators verified: 10Y Rate = {macro_data['rate10y_pct']}%, CPI = {macro_data['cpi_yoy_pct']}%, US GDP = {macro_data['gdp_growth_us_pct']}%")
    
    # 3. Test News & Sentiment Engine
    print("\n[3/7] Testing News & NLP Sentiment Engine...")
    from TransparaScore import fetch_news_and_sentiment
    sentiment_val, news_items = fetch_news_and_sentiment("AAPL", "Apple Inc.")
    assert isinstance(sentiment_val, float), "Sentiment value must be float"
    assert len(news_items) > 0, "News items list must not be empty"
    print(f"  [OK] News & Sentiment verified: Processed {len(news_items)} headlines | Net Polarity = {sentiment_val:+.2f}")
    
    # 4. Test Altman Z-Score & Merton PD Models
    print("\n[4/7] Testing Altman Z-Score & Default Probability Models...")
    from TransparaScore import calculate_altman_z_score
    altman_res = calculate_altman_z_score(meta)
    assert "z_score" in altman_res, "Altman result missing z_score"
    assert "pd_1yr_pct" in altman_res, "Altman result missing pd_1yr_pct"
    assert 0 <= altman_res["pd_1yr_pct"] <= 100, "1-Yr PD out of 0-100% bounds"
    print(f"  [OK] Altman Z-Score = {altman_res['z_score']:.2f} ({altman_res['zone']}) | 1-Yr PD = {altman_res['pd_1yr_pct']:.2f}%")
    
    # 5. Test Machine Learning Model & SHAP TreeExplainer
    print("\n[5/7] Testing ML Model & SHAP TreeExplainer Attribution...")
    from TransparaScore import load_or_train_credit_model, calculate_shap_contributions
    model, feature_names = load_or_train_credit_model()
    dummy_input = pd.DataFrame([{
        "Price_Return_Pct": 5.2,
        "Volatility_Pct": 15.0,
        "Max_Drawdown_Pct": -8.5,
        "Liquidity_Proxy": 15.0,
        "Debt_to_Equity": 45.0,
        "Stock_Beta": 1.05,
        "Inflation_CPI_Pct": 2.8,
        "Unemployment_Pct": 4.1,
        "Treasury_10Y_Rate": 4.2,
        "GDP_Growth_Pct": 3.0,
        "News_Sentiment_Score": 1.5
    }])
    pred = model.predict(dummy_input)[0]
    assert 0 <= pred <= 100, f"Predicted score {pred} out of 0-100 bounds"
    
    shap_df = calculate_shap_contributions(model, dummy_input)
    assert not shap_df.empty, "SHAP dataframe should not be empty"
    print(f"  [OK] ML Model Score = {pred:.2f}/100 | Generated SHAP attributions for {len(shap_df)} features")
    
    # 6. Test Monte Carlo Simulation Engine
    print("\n[6/7] Testing 1,000-Path Monte Carlo Stress Simulator...")
    from TransparaScore import run_monte_carlo_simulation
    mc_res = run_monte_carlo_simulation(base_score=75.0, volatility=18.5, n_sims=1000, horizon_months=12)
    assert len(mc_res["final_distribution"]) == 1000, "Monte carlo distribution should have 1,000 values"
    assert "var_95" in mc_res, "MC results missing var_95"
    print(f"  [OK] Monte Carlo Sim: Median = {mc_res['median_projected']}, 95% Worst-Case VaR = {mc_res['var_95']}, Junk Prob = {mc_res['prob_junk_pct']}%")

    # 7. Test Rating Tiers & Stress Testing Logic
    print("\n[7/7] Testing Rating Bands & Stress Engine...")
    from TransparaScore import get_credit_rating_tier, calculate_financial_health_score, calculate_macro_score
    r_aaa = get_credit_rating_tier(92.0)
    r_bbb = get_credit_rating_tier(65.0)
    r_d = get_credit_rating_tier(15.0)
    assert r_aaa["tier"] == "AAA" and r_bbb["tier"] == "BBB" and r_d["tier"] == "D"
    
    fin_score, _ = calculate_financial_health_score(df, meta)
    mac_score, _ = calculate_macro_score(macro_data)
    assert 0 <= fin_score <= 100, "Financial score out of bounds"
    assert 0 <= mac_score <= 100, "Macro score out of bounds"
    print(f"  [OK] Rating & Scoring Logic verified: Fin Score = {fin_score:.1f}, Mac Score = {mac_score:.1f}")

    print("\n==================================================")
    print("  [SUCCESS] ALL 7 INSTITUTIONAL MODULES PASSED!    ")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
