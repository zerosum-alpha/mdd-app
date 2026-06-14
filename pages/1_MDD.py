import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykrx import stock as krx_stock

# 차트 그리기 함수 중 PER 패널 부분 (한국 주식 맞춤형)
def make_kr_comprehensive_chart(df, ticker):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2.5, 1.2, 1.2]}, sharex=True)
    
    # 1. 주가 패널
    ax1.plot(df.index, df["Close"], label="Price", color="#1F77B4", linewidth=2)
    ax1.set_title(f"[{ticker}] KRX Trading & Valuation", fontsize=14, fontweight='bold')
    
    # 2. PER 패널 (한국 주식 전용)
    if 'PER' in df.columns:
        ax2.plot(df.index, df['PER'], color="#D69E2E", linewidth=2, label="KRX PER")
        ax2.axhline(df['PER'].mean(), color="gray", linestyle="--", label="Avg PER")
        ax2.set_ylabel("PER (x)", color="#D69E2E", fontweight='bold')
        ax2.set_ylim(0, 50) # PER 왜곡 방지 범위 설정
        ax2.legend()
        ax2.grid(True, linestyle=':', alpha=0.6)
    
    # 3. MDD 패널
    ax3.fill_between(df.index, df["Current_Drawdown"] * 100, 0, color="red", alpha=0.2)
    ax3.set_ylabel("Drawdown (%)", fontweight='bold')
    
    plt.tight_layout()
    return fig

# 사용 시 호출 예시:
# per_df = load_krx_per_data(ticker, start_date)
# df = df.join(per_df, how='left')
# st.pyplot(make_kr_comprehensive_chart(df, ticker))
