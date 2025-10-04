import streamlit as st
import pandas as pd
from io import StringIO
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# Streamlit page config
st.set_page_config(page_title="InvAI - Pro-Level AI Auditor", layout="wide")

# -------------------------------
# Constants
CII = {2015:254, 2016:264, 2017:272, 2018:280, 2019:289, 2020:301, 
       2021:317, 2022:331, 2023:348, 2024:362, 2025:375}

EQUITY_LTCG_EXEMPTION = 100000  # ₹1L per FY
EQUITY_LTCG_RATE = 0.20
EQUITY_STCG_RATE = 0.125
REALESTATE_LTCG_RATE = 0.20
REALESTATE_STCG_SLABS = [(250000,0),(500000,0.05),(1000000,0.20),(float('inf'),0.30)]
CESS = 0.04

# -------------------------------
# CSV Parsing
def parse_csv(file_bytes):
    s = file_bytes.decode('utf-8-sig')
    df = pd.read_csv(StringIO(s))
    expected_cols = ['client_id','asset_type','asset_id','transaction_type','trade_date','quantity','price','currency']
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        return None
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['transaction_type','trade_date','client_id','asset_id','quantity','price'])
    df['transaction_type'] = df['transaction_type'].str.strip().str.upper()
    df['asset_type'] = df['asset_type'].str.strip()
    return df

# -------------------------------
# Tax Calculations
def calculate_indexed_ltcg(purchase_price, buy_year, sell_year, sale_price):
    cii_buy = CII.get(buy_year, CII[max(CII.keys())])
    cii_sell = CII.get(sell_year, CII[max(CII.keys())])
    indexed_cost = purchase_price * (cii_sell / cii_buy)
    ltcg = sale_price - indexed_cost
    return ltcg, indexed_cost

def stcg_real_estate_tax(stcg_amount, slab_rate=None):
    if slab_rate is not None:
        tax = stcg_amount * slab_rate * (1+CESS)
        return tax
    tax = 0
    remaining = stcg_amount
    for limit, rate in REALESTATE_STCG_SLABS:
        taxable = min(remaining, limit)
        tax += taxable * rate
        remaining -= taxable
        if remaining <= 0: break
    tax *= (1+CESS)
    return tax

# -------------------------------
# AI Auditor Pro Suggestions
def professional_suggestions_pro(row):
    asset_type = str(row['Asset Type']).lower()
    gain = row.get('Gain', 0)
    gain_type = row.get('Gain Type','LTCG')
    strategies = []

    # -------------------------
    # Real Estate / Land / Property
    # -------------------------
    if asset_type in ['realestate','land','property']:
        base_tax = gain * REALESTATE_LTCG_RATE*(1+CESS) if gain_type=='LTCG' else stcg_real_estate_tax(gain)
        strategies.append({'Strategy':'Base Tax','Tax':round(base_tax,2),'Savings':0})
        
        # Sec 54 / 54F / 54EC
        if gain_type=='LTCG' and gain>0:
            strategies.append({'Strategy':'Sec 54/54F/54EC Reinvestment','Tax':0,'Savings':round(base_tax,2)})
        
        # Indexation benefit
        if gain_type=='LTCG' and gain>0:
            strategies.append({'Strategy':'Indexation Benefit','Tax':0,'Savings':round(base_tax,2)})
        
        # Joint/HUF ownership
        strategies.append({'Strategy':'Joint Ownership / HUF Ownership','Tax':0,'Savings':round(base_tax,2)})

    # -------------------------
    # Equity / Mutual Funds / Stocks
    # -------------------------
    elif asset_type in ['equity','mutualfund','mf','stock']:
        if gain_type=='STCG':
            base_tax = gain * EQUITY_STCG_RATE
            strategies.append({'Strategy':'Base Tax','Tax':round(base_tax,2),'Savings':0})
        else:
            taxable = max(0, gain-EQUITY_LTCG_EXEMPTION)
            base_tax = taxable * EQUITY_LTCG_RATE
            strategies.append({'Strategy':'Base Tax','Tax':round(base_tax,2),'Savings':0})
        # Optional: offset ST/LT losses
        if gain_type=='STCG' and row.get('Has ST Loss', False):
            strategies.append({'Strategy':'Offset short-term losses','Tax':round(base_tax*0.7,2),'Savings':round(base_tax*0.3,2)})
        if gain_type=='LTCG' and row.get('Has LT Loss', False):
            strategies.append({'Strategy':'Offset long-term losses','Tax':round(base_tax*0.7,2),'Savings':round(base_tax*0.3,2)})

    # -------------------------
    # Other Assets (Gold, Crypto, FD, Startup)
    # -------------------------
    else:
        base_tax = gain * 0.2*(1+CESS)
        strategies.append({'Strategy':'Base Tax','Tax':round(base_tax,2),'Savings':0})
        strategies.append({'Strategy':'Increase Cost Base','Tax':round(base_tax*0.8,2),'Savings':round(base_tax*0.2,2)})
        if row.get('Has ST Loss', False):
            strategies.append({'Strategy':'Offset Losses','Tax':round(base_tax*0.75,2),'Savings':round(base_tax*0.25,2)})

    best_strategy = min(strategies, key=lambda x: x['Tax'])
    return strategies, round(best_strategy['Tax'],2), best_strategy['Strategy']

# -------------------------------
# Build lots & Gains
def build_lots(df):
    lots = {}
    results = []
    df_sorted = df.sort_values('trade_date')
    for _, r in df_sorted.iterrows():
        key = (r['client_id'], r['asset_id'])
        if r['transaction_type']=='BUY':
            lots.setdefault(key, []).append({'qty':r['quantity'],'price':r['price'],'date':r['trade_date']})
        elif r['transaction_type']=='SELL':
            sell_qty = r['quantity']
            proceeds = r['price']*sell_qty
            matched = []
            while sell_qty>0 and lots.get(key):
                lot = lots[key][0]
                matched_qty = min(lot['qty'], sell_qty)
                lot['qty'] -= matched_qty
                if lot['qty']==0: lots[key].pop(0)
                sell_qty -= matched_qty
                cost = matched_qty * lot['price']
                gain = matched_qty*r['price'] - cost
                holding_days = (r['trade_date'] - lot['date']).days
                gain_type = 'LTCG' if ((asset_type := r['asset_type'].lower()) in ['realestate','land','property'] and holding_days>730) or (holding_days>365 and asset_type not in ['realestate','land','property']) else 'STCG'
                if asset_type in ['realestate','land','property']:
                    tax = calculate_indexed_ltcg(lot['price'], lot['date'].year, r['trade_date'].year, r['price'])[0]*REALESTATE_LTCG_RATE*(1+CESS) if gain_type=='LTCG' else stcg_real_estate_tax(gain)
                elif asset_type in ['equity','mutualfund','mf','stock']:
                    tax = max(0,gain-EQUITY_LTCG_EXEMPTION)*EQUITY_LTCG_RATE if gain_type=='LTCG' else gain*EQUITY_STCG_RATE
                else:
                    tax = gain*0.2*(1+CESS)
                matched.append({'matched_qty':matched_qty,'cost':cost,'gain':gain,'holding_days':holding_days,'gain_type':gain_type,'tax_estimate':tax})
            results.append({'client_id':r['client_id'],'asset_id':r['asset_id'],'asset_type':r['asset_type'],
                            'sell_date':r['trade_date'],'proceeds':proceeds,'matches':matched})
    return results

# -------------------------------
# Streamlit UI
st.title("💹 InvAI - Professional AI Auditor Dashboard (Pro Strategies)")

uploaded_file = st.file_uploader("Upload CSV of client transactions", type="csv")
if uploaded_file:
    df = parse_csv(uploaded_file.read())
    if df is not None:
        # Calculate holding period in days
        df['holding_days'] = 0
        buys = df[df['transaction_type']=='BUY']
        sells = df[df['transaction_type']=='SELL']
        for idx, sell in sells.iterrows():
            key = (sell['client_id'], sell['asset_id'])
            buy_rows = buys[(buys['client_id']==sell['client_id']) & (buys['asset_id']==sell['asset_id'])]
            if not buy_rows.empty:
                first_buy = buy_rows.iloc[0]
                df.at[idx,'holding_days'] = (sell['trade_date'] - first_buy['trade_date']).days

        st.success(f"CSV Loaded: {len(df)} rows")
        st.subheader("📄 Uploaded Transactions with Holding Period")
        st.dataframe(df)

        results = build_lots(df)
        clients = df['client_id'].unique()
        summary = []

        for client in clients:
            st.header(f"👤 Client: {client}")
            client_df = df[df['client_id']==client]

            # Current Holdings
            st.subheader("📈 Current Holdings")
            holdings = client_df[client_df['transaction_type']=='BUY'].groupby(['asset_id','asset_type']).agg(total_qty=('quantity','sum'), avg_price=('price','mean')).reset_index()
            st.dataframe(holdings)

            # Flatten results for client
            flat_rows = []
            for r in results:
                if r['client_id'] != client: continue
                for m in r['matches']:
                    has_st_loss = m['gain']<0 and m['gain_type']=='STCG'
                    has_lt_loss = m['gain']<0 and m['gain_type']=='LTCG'
                    flat_rows.append({'Asset':r['asset_id'],'Asset Type':r['asset_type'],'Sell Date':r['sell_date'].date(),
                                      'Quantity':m['matched_qty'],'Gain Type':m['gain_type'],'Gain':round(m['gain'],2),
                                      'Estimated Tax':round(m['tax_estimate'],2),'Has ST Loss':has_st_loss,'Has LT Loss':has_lt_loss})

            flat_df = pd.DataFrame(flat_rows)
            if flat_df.empty: continue

            # -------------------------------
            # AI Suggestions & Per-Strategy Tax
            optimized_tax_total = 0
            st.subheader("📌 AI Auditor Suggestions with Tax Savings")
            for idx,row in flat_df.iterrows():
                st.markdown(f"**Asset {row['Asset']} | {row['Asset Type']} | Gain Type: {row['Gain Type']} | Gain ₹{row['Gain']}**")
                strategies, best_tax, best_strategy_name = professional_suggestions_pro(row)
                optimized_tax_total += best_tax

                # Display strategy table
                strategy_df = pd.DataFrame(strategies)
                st.table(strategy_df)

                st.success(f"✅ Recommended Strategy: {best_strategy_name} → Tax after strategy: ₹{best_tax}")
                st.markdown("---")

            total_gain = flat_df['Gain'].sum()
            total_estimated_tax = flat_df['Estimated Tax'].sum()
            st.markdown(f"**Total Gain:** ₹{round(total_gain,2)} | **Estimated Tax:** ₹{round(total_estimated_tax,2)} | **Optimized Tax:** ₹{round(optimized_tax_total,2)}")
            summary.append({'Client':client,'Total Gain':total_gain,'Estimated Tax':total_estimated_tax,'Optimized Tax':optimized_tax_total})

        # -------------------------------
        # Dashboard charts
        if summary:
            summary_df = pd.DataFrame(summary)
            summary_df['Tax Savings'] = summary_df['Estimated Tax'] - summary_df['Optimized Tax']

            st.subheader("📊 Client-wise Gain vs Tax Overview")
            fig_overview = px.bar(summary_df,x='Client',y=['Total Gain','Estimated Tax','Optimized Tax'],barmode='group',
                                  text_auto=True,color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_overview,use_container_width=True)

            st.subheader("🥧 Potential Tax Savings per Client")
            fig_savings=px.bar(summary_df,x='Client',y='Tax Savings',text='Tax Savings',color='Tax Savings',
                               color_continuous_scale='Viridis',title="Potential Tax Savings if AI Auditor Suggestions Followed")
            fig_savings.update_layout(yaxis_title="Potential Tax Savings (₹)", xaxis_title="Client")
            st.plotly_chart(fig_savings,use_container_width=True)

            st.subheader("📈 Full Client Financial Dashboard")
            fig_combined=go.Figure()
            fig_combined.add_trace(go.Bar(x=summary_df['Client'],y=summary_df['Total Gain'],name='Total Gain',marker_color='blue'))
            fig_combined.add_trace(go.Bar(x=summary_df['Client'],y=summary_df['Estimated Tax'],name='Estimated Tax',marker_color='red'))
            fig_combined.add_trace(go.Bar(x=summary_df['Client'],y=summary_df['Optimized Tax'],name='Optimized Tax',marker_color='green'))
            fig_combined.add_trace(go.Bar(x=summary_df['Client'],y=summary_df['Tax Savings'],name='Potential Tax Savings',marker_color='orange'))
            fig_combined.update_layout(barmode='group',yaxis_title='Amount (₹)',xaxis_title='Client',title='Client-wise Financial Overview')
            st.plotly_chart(fig_combined,use_container_width=True)
