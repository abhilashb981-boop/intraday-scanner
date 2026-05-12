"""
Intraday Scanner v5 — Streamlit Cloud Edition
Mobile-friendly · Deploy free on share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import pytz
from datetime import datetime, timedelta

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📈 Intraday Scanner v5",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Mobile-friendly CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container{padding-top:1rem;padding-bottom:0rem}
  .stMetric{background:#0c1120;border-radius:10px;padding:10px;border:1px solid #1e293b}
  .stMetric label{color:#475569!important;font-size:12px!important}
  .buy-badge{background:#052e16;color:#4ade80;padding:4px 10px;border-radius:12px;font-weight:700;font-size:13px}
  .sell-badge{background:#450a0a;color:#f87171;padding:4px 10px;border-radius:12px;font-weight:700;font-size:13px}
  .hold-badge{background:#1c1917;color:#fbbf24;padding:4px 10px;border-radius:12px;font-weight:700;font-size:13px}
  .perfect-badge{background:#071a07;color:#fbbf24;padding:4px 12px;border-radius:12px;font-weight:800;font-size:14px;border:1px solid #fbbf24}
  .mkt-live{color:#22c55e;font-weight:700}
  .mkt-closed{color:#ef4444;font-weight:700}
  .mkt-pre{color:#f59e0b;font-weight:700}
  div[data-testid="stDataFrame"]{border-radius:10px}
  @media(max-width:768px){.stTabs [data-baseweb="tab"]{font-size:11px;padding:6px 8px}}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════════════════

def calc_rsi(s, p=14):
    if len(s)<p+1: return None
    d=s.diff().dropna()
    ag=d.clip(lower=0).ewm(com=p-1,min_periods=p).mean().iloc[-1]
    al=(-d.clip(upper=0)).ewm(com=p-1,min_periods=p).mean().iloc[-1]
    return round(100-100/(1+ag/al),2) if al else 100.0

def calc_ema(s,p):
    if len(s)<p: return None
    return round(s.ewm(span=p,adjust=False).mean().iloc[-1],2)

def calc_atr(df,p=14):
    h,l,c=df['High'],df['Low'],df['Close']
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(span=p,adjust=False).mean()

def calc_supertrend(df,p=10,mult=3.0):
    if len(df)<p+2: return None,None
    atr=calc_atr(df,p); hl2=(df['High']+df['Low'])/2
    upper=hl2+mult*atr; lower=hl2-mult*atr; close=df['Close']
    st=pd.Series(index=df.index,dtype=float)
    dr=pd.Series(index=df.index,dtype=int)
    st.iloc[p]=upper.iloc[p]; dr.iloc[p]=-1
    for i in range(p+1,len(df)):
        cu=upper.iloc[i]; cl_=lower.iloc[i]
        pu=upper.iloc[i-1]; pl=lower.iloc[i-1]; pd_=dr.iloc[i-1]
        fl=cl_ if cl_>pl or close.iloc[i-1]<pl else pl
        fu=cu  if cu<pu  or close.iloc[i-1]>pu else pu
        if pd_==-1: dr.iloc[i]=1 if close.iloc[i]<fl else -1
        else:       dr.iloc[i]=-1 if close.iloc[i]>fu else 1
        st.iloc[i]=fl if dr.iloc[i]==1 else fu
    ld=dr.dropna().iloc[-1] if not dr.dropna().empty else None
    lv=round(st.dropna().iloc[-1],2) if not st.dropna().empty else None
    return lv,int(ld) if ld is not None else None

def calc_adx(df,p=14):
    if len(df)<p*2: return None,None,None
    h,l,c=df['High'],df['Low'],df['Close']
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    dmp=h.diff(); dmm=-l.diff()
    dmp=dmp.where((dmp>dmm)&(dmp>0),0)
    dmm=dmm.where((dmm>dmp)&(dmm>0),0)
    a14=tr.ewm(span=p,adjust=False).mean()
    dip=100*dmp.ewm(span=p,adjust=False).mean()/a14
    dim=100*dmm.ewm(span=p,adjust=False).mean()/a14
    dx=100*(dip-dim).abs()/(dip+dim).replace(0,0.001)
    adx=dx.ewm(span=p,adjust=False).mean()
    return round(adx.iloc[-1],2),round(dip.iloc[-1],2),round(dim.iloc[-1],2)

def calc_stoch_rsi(s,rp=14,sp=14,sk=3,sd=3):
    if len(s)<rp+sp+sk: return None,None
    d=s.diff().dropna()
    ag=d.clip(lower=0).ewm(com=rp-1,min_periods=rp).mean()
    al=(-d.clip(upper=0)).ewm(com=rp-1,min_periods=rp).mean()
    rs=100-100/(1+ag/al.replace(0,0.001)); rs=rs.dropna()
    stk=100*(rs-rs.rolling(sp).min())/(rs.rolling(sp).max()-rs.rolling(sp).min()).replace(0,0.001)
    k=stk.rolling(sk).mean(); dd=k.rolling(sd).mean()
    kv=k.dropna().iloc[-1] if not k.dropna().empty else None
    dv=dd.dropna().iloc[-1] if not dd.dropna().empty else None
    return (round(kv,2) if kv else None),(round(dv,2) if dv else None)

def calc_bollinger(s,p=20,std=2.0):
    if len(s)<p: return None,None,None
    mid=s.rolling(p).mean(); sig=s.rolling(p).std()
    return round((mid+std*sig).iloc[-1],2),round(mid.iloc[-1],2),round((mid-std*sig).iloc[-1],2)

def calc_vwap(df):
    tp=(df['High']+df['Low']+df['Close'])/3
    cv=df['Volume'].cumsum()
    return round((tp*df['Volume']).cumsum().iloc[-1]/cv.iloc[-1],2) if cv.iloc[-1]>0 else None

def calc_obv_trend(df):
    c,v=df['Close'].values,df['Volume'].values
    obv=[0]
    for i in range(1,len(c)):
        obv.append(obv[-1]+(v[i] if c[i]>c[i-1] else -v[i] if c[i]<c[i-1] else 0))
    s=pd.Series(obv); slope=s.iloc[-1]-s.iloc[-min(10,len(s))]
    return ("Accumulation ↑","bullish") if slope>0 else ("Distribution ↓","bearish")

def calc_cmf(df,p=14):
    h,l,c,v=df['High'],df['Low'],df['Close'],df['Volume']
    mfm=((c-l)-(h-c))/(h-l).replace(0,0.001)
    cmf=(mfm*v).rolling(p).sum()/v.rolling(p).sum()
    val=cmf.iloc[-1]; return round(val,4) if not pd.isna(val) else 0.0

def calc_pivots(df_d):
    if df_d is None or len(df_d)<2: return None
    p=df_d.iloc[-2]; pp=(p['High']+p['Low']+p['Close'])/3; r=p['High']-p['Low']
    return {k:round(v,2) for k,v in {"PP":pp,"R1":2*pp-p['Low'],"R2":pp+r,"S1":2*pp-p['High'],"S2":pp-r}.items()}

def detect_trend(df):
    if len(df)<10: return "—","neutral"
    h,l=df['High'].values,df['Low'].values; mid=len(h)//2
    hh=h[-1]>h[mid]; hl=l[-1]>l[mid]; lh=h[-1]<h[mid]; ll=l[-1]<l[mid]
    if hh and hl: return "Uptrend ↑","bullish"
    if lh and ll: return "Downtrend ↓","bearish"
    return "Sideways ↔","neutral"

def calc_rvol(df5d):
    try:
        ist=pytz.timezone('Asia/Kolkata'); idx=df5d.index.tz_convert(ist)
        dates=idx.normalize().unique()
        if len(dates)<2: return 0.0
        today_v=df5d[idx.normalize()==dates[-1]]['Volume']
        yest_v =df5d[idx.normalize()==dates[-2]]['Volume']
        n=min(len(today_v),len(yest_v))
        if n==0: return 0.0
        yvol=yest_v.iloc[:n].sum()
        return round(today_v.iloc[:n].sum()/yvol,2) if yvol>0 else 0.0
    except: return 0.0

def get_last_trading_day_data(tk_obj, interval="5m"):
    raw=tk_obj.history(period="5d",interval=interval)
    if raw is None or len(raw)==0: return pd.DataFrame()
    try:
        ist=pytz.timezone('Asia/Kolkata'); idx=raw.index.tz_convert(ist)
        ld=idx.normalize()[-1]; filtered=raw[idx.normalize()==ld]
        if len(filtered)<5:
            ud=idx.normalize().unique()
            if len(ud)>=2: filtered=raw[idx.normalize()==ud[-2]]
        return filtered
    except:
        ld=raw.index.normalize()[-1]; return raw[raw.index.normalize()==ld]

def get_nifty_trend():
    try:
        raw=yf.Ticker("^NSEI").history(period="2d",interval="5m")
        if raw is None or len(raw)<10: return "neutral","NIFTY N/A"
        ist=pytz.timezone('Asia/Kolkata'); idx=raw.index.tz_convert(ist)
        today=raw[idx.normalize()==idx.normalize()[-1]]
        if len(today)<5: return "neutral","Pre-market"
        h,l=today['High'].values,today['Low'].values; mid=len(h)//2
        c=today['Close']; chg=round((c.iloc[-1]-c.iloc[0])/c.iloc[0]*100,2)
        if h[-1]>h[mid] and l[-1]>l[mid]: return "bullish",f"NIFTY ↑ Uptrend {chg:+.2f}%"
        if h[-1]<h[mid] and l[-1]<l[mid]: return "bearish",f"NIFTY ↓ Downtrend {chg:+.2f}%"
        return "neutral",f"NIFTY ↔ Sideways {chg:+.2f}%"
    except: return "neutral","NIFTY N/A"

NSE_H={'User-Agent':'Mozilla/5.0','Referer':'https://www.nseindia.com/','Accept':'*/*'}
_sess=None
def nse_sess():
    global _sess
    if _sess is None:
        _sess=requests.Session()
        try: _sess.get('https://www.nseindia.com',headers=NSE_H,timeout=6)
        except: pass
    return _sess

def get_delivery(sym):
    try:
        r=nse_sess().get(f'https://www.nseindia.com/api/quote-equity?symbol={sym.upper()}',
                         headers=NSE_H,timeout=5)
        d=r.json().get('securityWiseDP',{})
        return round(float(d.get('deliveryToTradedQuantity',0)),2)
    except: return None

POS=["surge","rally","gain","jump","rise","soar","beat","profit","record","high","upgrade",
     "buy","outperform","strong","growth","bullish","positive","boost","breakout","momentum",
     "upside","exceed","robust","dividend","buyback","approval","win","deal","order","contract"]
NEG=["fall","drop","decline","loss","crash","sell","downgrade","underperform","weak","bearish",
     "negative","concern","worry","risk","miss","cut","reduce","penalty","fraud","probe",
     "debt","default","warning","scam","block","ban","rejection","slowdown","slump","plunge"]

def get_news_sentiment(symbol,sfx):
    try:
        raw=yf.Ticker(symbol+sfx).get_news(count=6) or []
        scores=[]
        for item in raw[:6]:
            c=item.get('content',item)
            txt=f"{c.get('title','')} {c.get('summary','')}".lower()
            pos=sum(1 for w in POS if w in txt); neg=sum(1 for w in NEG if w in txt)
            scores.append(pos-neg)
        if not scores: return "⚪ No News","neutral"
        avg=sum(scores)/len(scores)
        if avg>=1.5:  return "🟢 Bullish News","bullish"
        if avg<=-1.5: return "🔴 Bearish News","bearish"
        if avg>0:     return "🟡 Slight Bullish","bullish"
        if avg<0:     return "🟡 Slight Bearish","bearish"
        return "⚪ Neutral","neutral"
    except: return "⚪ No News","neutral"

# ═══════════════════════════════════════════════════════
#  PERFECT TRADE CHECKLIST (25 rules)
# ═══════════════════════════════════════════════════════

def run_checklist(d, side):
    checks={}
    def ck(name,ok,detail): checks[name]=(ok,detail)
    last=d['last']; rsi=d['rsi']; adx=d['adx']; dip=d['dip']; dim=d['dim']
    st_dir=d['st_dir']; k=d['stoch_k']; dk=d['stoch_d']
    bb_mid=d['bb_mid']; vwap=d['vwap']; orb_h=d['orb_h']; orb_l=d['orb_l']
    trend_s=d['trend_sent']; obv_s=d['obv_sent']; cmf=d['cmf']; gap=d['gap']
    ema9=d['ema9']; ema21=d['ema21']; mtf15=d['mtf15']; mtf1h=d['mtf1h']
    news_s=d['news_side']; vol_r=d['vol_r']; pivots=d['pivots']
    nifty_s=d['nifty_sent']; rvol=d['rvol']
    pdh=d['pdh']; pdl=d['pdl']; cbull=d['cbull']; cbear=d['cbear']
    cbullc=d['cbullc']; cbearc=d['cbearc']
    uw=d['uw']; lw=d['lw']; atr_m=d['atr_m']; atr_d=d['atr_d']
    ema9_1h=d['ema9_1h']; ema21_1h=d['ema21_1h']

    if side=='buy':
        if st_dir is not None: ck('① Supertrend BUY ↑',st_dir==1,f"{'↑ BUY' if st_dir==1 else '↓ SELL'} ({d['st_val']})")
        if adx:                 ck('② ADX > 25',adx>25,f"ADX={adx}")
        if dip and dim:         ck('③ DI+ > DI-',dip>dim,f"DI+={dip} DI-={dim}")
        if ema9 and ema21:      ck('④ EMA9 > EMA21 (5m)',ema9>ema21,f"EMA9={ema9} EMA21={ema21}")
        if vwap:                ck('⑤ Price Above VWAP',last>vwap,f"₹{last} vs VWAP ₹{vwap}")
        if rsi:                 ck('⑥ RSI 45–65',45<=rsi<=65,f"RSI={rsi}")
        if k and dk:            ck('⑦ StochRSI K>D & <75',k>dk and k<75,f"K={k} D={dk}")
        if orb_h:               ck('⑧ Above ORB High',last>orb_h,f"₹{last} vs ORB ₹{orb_h}")
        if bb_mid:              ck('⑨ Above BB Mid',last>=bb_mid,f"₹{last} vs BB ₹{bb_mid}")
        ck('⑩ 5m Uptrend HH+HL',trend_s=='bullish',f"{d['trend']}")
        if mtf15:               ck('⑪ 15m Uptrend',mtf15=='bullish',f"15m={mtf15}")
        ck('⑫ OBV Accumulation',obv_s=='bullish',f"{d['obv']}")
        ck('⑬ CMF > +0.05',cmf>0.05,f"CMF={cmf:.3f}")
        ck('⑭ Volume > 1.3x',vol_r>=1.3,f"Vol={vol_r:.1f}x")
        ck('⑮ No Bearish News',news_s!='bearish',f"News: {d['news_lbl']}")
        ck('⑯ No Gap Down',gap>=-0.5,f"Gap={gap:+.2f}%")
        if pivots:              ck('⑰ Above S1 Pivot',last>pivots['S1'],f"₹{last} vs S1 ₹{pivots['S1']}")
        ck('⑱ NIFTY Bullish',nifty_s=='bullish',f"{d['nifty_lbl']}")
        ck('⑲ RVOL > 1.5x',rvol>=1.5,f"RVOL={rvol}x vs yesterday")
        if pdh:                 ck('⑳ Price > Prev Day High',last>pdh,f"₹{last} vs PDH ₹{pdh}")
        ck('㉑ Last 3 Candles All Green',cbull,f"{cbullc}/3 green candles")
        ck('㉒ No Big Upper Wick (<25%)',uw<25,f"Upper wick={uw}%")
        if mtf1h:               ck('㉓ 1H Uptrend',mtf1h=='bullish',f"1H={mtf1h}")
        if ema9_1h and ema21_1h:ck('㉔ 1H EMA9 > EMA21',ema9_1h>ema21_1h,f"1H EMA9={ema9_1h} EMA21={ema21_1h}")
        ck('㉕ ATR Momentum ≥ 0.4x ↑',atr_m>=0.4 and atr_d=='up',f"{atr_m}x ATR dir={atr_d}")
    else:
        if st_dir is not None: ck('① Supertrend SELL ↓',st_dir==-1,f"{'↓ SELL' if st_dir==-1 else '↑ BUY'} ({d['st_val']})")
        if adx:                 ck('② ADX > 25',adx>25,f"ADX={adx}")
        if dip and dim:         ck('③ DI- > DI+',dim>dip,f"DI+={dip} DI-={dim}")
        if ema9 and ema21:      ck('④ EMA9 < EMA21 (5m)',ema9<ema21,f"EMA9={ema9} EMA21={ema21}")
        if vwap:                ck('⑤ Price Below VWAP',last<vwap,f"₹{last} vs VWAP ₹{vwap}")
        if rsi:                 ck('⑥ RSI 35–55',35<=rsi<=55,f"RSI={rsi}")
        if k and dk:            ck('⑦ StochRSI K<D & >25',k<dk and k>25,f"K={k} D={dk}")
        if orb_l:               ck('⑧ Below ORB Low',last<orb_l,f"₹{last} vs ORB ₹{orb_l}")
        if bb_mid:              ck('⑨ Below BB Mid',last<=bb_mid,f"₹{last} vs BB ₹{bb_mid}")
        ck('⑩ 5m Downtrend LH+LL',trend_s=='bearish',f"{d['trend']}")
        if mtf15:               ck('⑪ 15m Downtrend',mtf15=='bearish',f"15m={mtf15}")
        ck('⑫ OBV Distribution',obv_s=='bearish',f"{d['obv']}")
        ck('⑬ CMF < -0.05',cmf<-0.05,f"CMF={cmf:.3f}")
        ck('⑭ Volume > 1.3x',vol_r>=1.3,f"Vol={vol_r:.1f}x")
        ck('⑮ No Bullish News',news_s!='bullish',f"News: {d['news_lbl']}")
        ck('⑯ No Gap Up',gap<=0.5,f"Gap={gap:+.2f}%")
        if pivots:              ck('⑰ Below R1 Pivot',last<pivots['R1'],f"₹{last} vs R1 ₹{pivots['R1']}")
        ck('⑱ NIFTY Bearish',nifty_s=='bearish',f"{d['nifty_lbl']}")
        ck('⑲ RVOL > 1.5x',rvol>=1.5,f"RVOL={rvol}x vs yesterday")
        if pdl:                 ck('⑳ Price < Prev Day Low',last<pdl,f"₹{last} vs PDL ₹{pdl}")
        ck('㉑ Last 3 Candles All Red',cbear,f"{cbearc}/3 red candles")
        ck('㉒ No Big Lower Wick (<25%)',lw<25,f"Lower wick={lw}%")
        if mtf1h:               ck('㉓ 1H Downtrend',mtf1h=='bearish',f"1H={mtf1h}")
        if ema9_1h and ema21_1h:ck('㉔ 1H EMA9 < EMA21',ema9_1h<ema21_1h,f"1H EMA9={ema9_1h} EMA21={ema21_1h}")
        ck('㉕ ATR Momentum ≥ 0.4x ↓',atr_m>=0.4 and atr_d=='down',f"{atr_m}x ATR dir={atr_d}")
    passed=sum(1 for v in checks.values() if v[0])
    return checks,passed,len(checks),passed==len(checks)

# ═══════════════════════════════════════════════════════
#  SCAN ONE STOCK
# ═══════════════════════════════════════════════════════

def scan_stock(symbol, nifty_sent, nifty_lbl):
    for sfx in [".NS",".BO"]:
        try:
            tk=yf.Ticker(symbol+sfx)
            d5m=get_last_trading_day_data(tk,"5m")
            d5d=tk.history(period="5d",interval="5m")
            d15=tk.history(period="5d",interval="15m")
            d1h=tk.history(period="10d",interval="1h")
            d1d=tk.history(period="60d",interval="1d")
            if d5m is None or len(d5m)<15: continue

            closes=d5m['Close'].dropna(); vols=d5m['Volume'].dropna()
            last=round(closes.iloc[-1],2); opens=d5m['Open'].dropna()
            prev_c=d1d['Close'].iloc[-2] if len(d1d)>=2 else closes.iloc[0]
            gap=round((opens.iloc[0]-prev_c)/prev_c*100,2) if prev_c else 0
            avg_vol=vols.iloc[:-1].mean() if len(vols)>1 else vols.mean()
            vol_r=round(vols.iloc[-1]/avg_vol,2) if avg_vol>0 else 0
            atr_ser=calc_atr(d5m); atr_v=round(atr_ser.iloc[-1],2) if len(atr_ser)>0 else last*0.01

            # Indicators
            rsi=calc_rsi(closes); ema9=calc_ema(closes,9); ema21=calc_ema(closes,min(21,len(closes)))
            st_val,st_dir=calc_supertrend(d5m); adx,dip,dim=calc_adx(d5m)
            k,dk=calc_stoch_rsi(closes); _,bb_mid,_=calc_bollinger(closes)
            vwap=calc_vwap(d5m); obv_lbl,obv_sent=calc_obv_trend(d5m)
            cmf=calc_cmf(d5m); pivots=calc_pivots(d1d)
            trend_str,trend_sent=detect_trend(d5m)
            chg=round((last-prev_c)/prev_c*100,2) if prev_c else 0

            # ORB
            try:
                ist=pytz.timezone('Asia/Kolkata'); idx=d5m.index.tz_convert(ist)
                ld=idx.normalize()[-1]; dt=d5m[idx.normalize()==ld]; orb=dt.head(3)
                orb_h=round(orb['High'].max(),2) if len(orb)>=3 else None
                orb_l=round(orb['Low'].min(),2) if len(orb)>=3 else None
            except: orb_h=orb_l=None

            # MTF 15m
            mtf15="neutral"
            if d15 is not None and len(d15)>=10: _,mtf15=detect_trend(d15)

            # MTF 1H
            mtf1h="neutral"; ema9_1h=ema21_1h=None
            if d1h is not None and len(d1h)>=10:
                try:
                    c1h=d1h['Close'].dropna(); _,mtf1h=detect_trend(d1h)
                    ema9_1h=calc_ema(c1h,9); ema21_1h=calc_ema(c1h,min(21,len(c1h)))
                except: pass

            # v5 indicators
            rvol=calc_rvol(d5d)
            pdh=round(d1d.iloc[-2]['High'],2) if len(d1d)>=2 else None
            pdl=round(d1d.iloc[-2]['Low'],2)  if len(d1d)>=2 else None

            last_n=d5m.tail(3)
            cbull=all(r['Close']>r['Open'] for _,r in last_n.iterrows())
            cbear=all(r['Close']<r['Open'] for _,r in last_n.iterrows())
            cbullc=sum(1 for _,r in last_n.iterrows() if r['Close']>r['Open'])
            cbearc=sum(1 for _,r in last_n.iterrows() if r['Close']<r['Open'])

            lc=d5m.iloc[-1]; rng=lc['High']-lc['Low'] or 0.001
            uw=round((lc['High']-max(lc['Open'],lc['Close']))/rng*100,1)
            lw_=round((min(lc['Open'],lc['Close'])-lc['Low'])/rng*100,1)

            try:
                pm=abs(closes.iloc[-1]-closes.iloc[-4]); ratio=round(pm/atr_v,2) if atr_v>0 else 0
                atr_d="up" if closes.iloc[-1]>closes.iloc[-4] else "down"
            except: ratio=0; atr_d="neutral"

            news_lbl,news_side=get_news_sentiment(symbol,sfx)
            delivery=get_delivery(symbol)

            # Intraday signal
            sc=0
            if st_dir==1:  sc+=3
            if st_dir==-1: sc-=3
            if adx and adx>25:
                if dip and dim and dip>dim: sc+=2
                elif dip and dim and dim>dip: sc-=2
            if ema9 and ema21: sc+=(1.5 if ema9>ema21 else -1.5)
            if vwap: sc+=(1 if last>vwap else -1)
            if rsi:
                if rsi<45: sc+=1
                elif rsi>65: sc-=1
            sc+=(1 if obv_sent=='bullish' else -1)
            if   sc>=3.5:  sig,sig_e="BUY","🟢 BUY"
            elif sc<=-3.5: sig,sig_e="SELL","🔴 SELL"
            else:          sig,sig_e="HOLD","🟡 HOLD"

            d={
                'symbol':symbol,'sfx':sfx.replace(".",""),'last':last,'chg':chg,'gap':gap,
                'rsi':rsi,'ema9':ema9,'ema21':ema21,'st_val':st_val,'st_dir':st_dir,
                'adx':adx,'dip':dip,'dim':dim,'stoch_k':k,'stoch_d':dk,'bb_mid':bb_mid,
                'vwap':vwap,'obv':obv_lbl,'obv_sent':obv_sent,'cmf':cmf,'pivots':pivots,
                'orb_h':orb_h,'orb_l':orb_l,'trend':trend_str,'trend_sent':trend_sent,
                'mtf15':mtf15,'mtf1h':mtf1h,'ema9_1h':ema9_1h,'ema21_1h':ema21_1h,
                'vol_r':vol_r,'news_lbl':news_lbl,'news_side':news_side,
                'nifty_sent':nifty_sent,'nifty_lbl':nifty_lbl,
                'rvol':rvol,'pdh':pdh,'pdl':pdl,
                'cbull':cbull,'cbear':cbear,'cbullc':cbullc,'cbearc':cbearc,
                'uw':uw,'lw':lw_,'atr_m':ratio,'atr_d':atr_d,'atr_v':atr_v,
                'delivery':f"{delivery}%" if delivery else "—",
                'sig':sig,'sig_e':sig_e,
            }
            bc,bp,bt,pfb=run_checklist(d,'buy')
            sc2,sp,st_,pfs=run_checklist(d,'sell')

            # Next day prediction
            if bp>=sp: side='buy'; conf=bp/bt*100
            else:      side='sell'; conf=sp/st_*100
            if pfb:          action="🏆 PERFECT BUY ↑ Karchintanga"
            elif pfs:        action="🏆 PERFECT SELL ↓ Karchintanga"
            elif bp>=bt*0.88:action="🟢 Strong BUY"
            elif sp>=st_*0.88:action="🔴 Strong SELL"
            elif bp>=bt*0.75:action="🟡 Possible BUY"
            elif sp>=st_*0.75:action="🟡 Possible SELL"
            else:            action="⚪ SKIP"
            if 'buy'  in action.lower(): tgt=round(last+2.0*atr_v,2); sl=round(last-0.8*atr_v,2)
            elif 'sell' in action.lower(): tgt=round(last-2.0*atr_v,2); sl=round(last+0.8*atr_v,2)
            else: tgt=sl=last
            rr=round(abs(tgt-last)/abs(sl-last),2) if abs(sl-last)>0 else 0

            d.update({'buy_checks':bc,'buy_passed':bp,'buy_total':bt,'perfect_buy':pfb,
                      'sell_checks':sc2,'sell_passed':sp,'sell_total':st_,'perfect_sell':pfs,
                      'action':action,'conf':round(conf,1),'tgt':tgt,'sl':sl,'rr':rr})
            return d
        except: continue
    return {'symbol':symbol,'error':True,'sig':'N/A','sig_e':'⚪ N/A',
            'buy_passed':0,'buy_total':25,'sell_passed':0,'sell_total':25,
            'perfect_buy':False,'perfect_sell':False,'action':'⚪ SKIP'}

# ═══════════════════════════════════════════════════════
#  MARKET STATUS
# ═══════════════════════════════════════════════════════

NSE_HOLIDAYS={(2026,1,26),(2026,2,19),(2026,3,17),(2026,4,2),(2026,4,10),
    (2026,4,14),(2026,4,17),(2026,5,1),(2026,5,28),(2026,6,26),
    (2026,8,15),(2026,9,14),(2026,10,2),(2026,10,20),
    (2026,11,10),(2026,11,24),(2026,12,25)}

def market_status():
    try:
        ist=pytz.timezone('Asia/Kolkata'); now=datetime.now(ist); wd=now.weekday(); t=now.time()
        ot=datetime.strptime("09:15","%H:%M").time(); ct=datetime.strptime("15:30","%H:%M").time()
        key=(now.year,now.month,now.day)
        if wd>=5:         return "🔴 Weekend — Market CLOSED","closed"
        if key in NSE_HOLIDAYS: return f"🔴 NSE Holiday ({now.strftime('%d %b')})","closed"
        if t<ot:
            secs=(datetime.combine(now.date(),ot)-datetime.combine(now.date(),t.replace(tzinfo=None))).seconds
            return f"🟡 Pre-Market — Opens in {secs//60} min","pre"
        if t>ct: return f"🟠 Closed for today — Last session data","closed"
        return f"🟢 MARKET LIVE — {now.strftime('%H:%M')} IST","live"
    except: return "⚪ Unknown","unknown"

# ═══════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════

def signal_badge(sig):
    if "PERFECT" in sig or "🏆" in sig:
        return f'<span class="perfect-badge">{sig}</span>'
    if "BUY" in sig or "🟢" in sig:
        return f'<span class="buy-badge">{sig}</span>'
    if "SELL" in sig or "🔴" in sig:
        return f'<span class="sell-badge">{sig}</span>'
    return f'<span class="hold-badge">{sig}</span>'

def main():
    # Header
    st.markdown("## 📈 Intraday Scanner v5")
    st.markdown("**25-Rule Checklist · NIFTY · RVOL · 1H MTF · PDH/L · Karchintanga Peragaali**")

    # Market status
    mst,mtype=market_status()
    css={"live":"mkt-live","pre":"mkt-pre","closed":"mkt-closed"}.get(mtype,"")
    st.markdown(f'<p class="{css}">{mst}</p>',unsafe_allow_html=True)
    st.divider()

    # Sidebar / Upload
    with st.sidebar:
        st.markdown("### 📂 Upload Stocks")
        uploaded=st.file_uploader("Excel file (symbols list)",type=["xlsx","xls"])
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        min_rules=st.slider("Min rules for Strong signal",10,24,19)
        show_hold=st.checkbox("Show HOLD stocks",value=False)
        st.markdown("---")
        st.markdown("### ℹ️ How to use")
        st.markdown("""
1. Excel lo stock symbols upload cheyyi
2. **Scan** button click cheyyi
3. 🏆 Perfect = ALL 25 rules pass
4. Double check ✅ Checklist tab lo
        """)

    # Main input — also allow manual entry
    col1,col2=st.columns([3,1])
    with col1:
        manual=st.text_input("Or type symbols manually (comma separated)",
                             placeholder="RELIANCE, TCS, INFY, HDFC")
    with col2:
        scan_btn=st.button("🔍 Scan Now",type="primary",use_container_width=True)

    # Parse symbols
    symbols=[]
    SKIP={"SYMBOL","STOCK","NAME","SCRIP","SR","NO","SL","SNO","QTY","QUANTITY","SERIES","DATE","CMP"}
    if uploaded:
        try:
            df=pd.read_excel(uploaded,header=None,dtype=str)
            for cell in df.values.flatten():
                if not isinstance(cell,str): continue
                s=cell.strip().upper().replace(".NS","").replace(".BO","").replace(" ","")
                if 2<=len(s)<=25 and s not in SKIP and s.replace("&","").replace("-","").isalnum():
                    if s not in symbols: symbols.append(s)
        except Exception as e:
            st.error(f"Excel read error: {e}")
    if manual:
        for s in manual.split(","):
            s=s.strip().upper().replace(".NS","").replace(".BO","")
            if s and s not in symbols: symbols.append(s)

    if symbols:
        st.success(f"📋 {len(symbols)} stocks loaded: {', '.join(symbols[:8])}{'...' if len(symbols)>8 else ''}")

    if not scan_btn or not symbols:
        if not symbols:
            st.info("👆 Upload Excel file or type stock symbols above, then click Scan Now")
        return

    # ── SCAN ────────────────────────────────────────────────────────────────
    st.markdown("---")

    # Fetch NIFTY first
    with st.spinner("📡 Fetching NIFTY trend..."):
        nifty_sent,nifty_lbl=get_nifty_trend()
    nifty_col={"bullish":"🟢","bearish":"🔴","neutral":"🟡"}.get(nifty_sent,"🟡")
    st.markdown(f"**{nifty_col} {nifty_lbl}**")

    results=[]
    prog=st.progress(0); status_txt=st.empty()

    for i,sym in enumerate(symbols):
        status_txt.markdown(f"⏳ Scanning **{sym}** ({i+1}/{len(symbols)})...")
        d=scan_stock(sym,nifty_sent,nifty_lbl)
        results.append(d)
        prog.progress((i+1)/len(symbols))

    status_txt.markdown(f"✅ Scan complete! **{len(symbols)}** stocks analyzed at {datetime.now().strftime('%H:%M:%S')}")

    # ── SUMMARY METRICS ──────────────────────────────────────────────────────
    buys   =[d for d in results if d.get('sig')=='BUY' and not d.get('error')]
    sells  =[d for d in results if d.get('sig')=='SELL' and not d.get('error')]
    holds  =[d for d in results if d.get('sig')=='HOLD' and not d.get('error')]
    p_buys =[d for d in results if d.get('perfect_buy')]
    p_sells=[d for d in results if d.get('perfect_sell')]

    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("🟢 BUY",    len(buys))
    c2.metric("🔴 SELL",   len(sells))
    c3.metric("🟡 HOLD",   len(holds))
    c4.metric("🏆 Perfect BUY",  len(p_buys),  delta="Karchintanga ↑" if p_buys else None)
    c5.metric("🏆 Perfect SELL", len(p_sells), delta="Karchintanga ↓" if p_sells else None)

    st.divider()

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab1,tab2,tab3,tab4,tab5=st.tabs([
        "🏆 Perfect Trade","📊 All Results","🆕 v5 Signals","✅ Checklist","💾 Export"])

    # ── TAB 1: Perfect Trade ─────────────────────────────────────────────────
    with tab1:
        perfect=[d for d in results if d.get('perfect_buy') or d.get('perfect_sell')]
        strong =[d for d in results if not d.get('perfect_buy') and not d.get('perfect_sell')
                 and (d.get('buy_passed',0)>=min_rules or d.get('sell_passed',0)>=min_rules)
                 and not d.get('error')]

        if perfect:
            st.markdown(f"### 🏆 Perfect Trades ({len(perfect)}) — Karchintanga Peragaali!")
            for d in perfect:
                pfb=d.get('perfect_buy'); pfs=d.get('perfect_sell')
                side_color="🟢" if pfb else "🔴"
                with st.expander(f"{side_color} **{d['symbol']}**  ₹{d['last']}  |  {d['action']}  |  Conf: {d.get('conf',0):.0f}%  |  R:R {d.get('rr',0)}"):
                    c1,c2,c3,c4=st.columns(4)
                    c1.metric("Entry",f"₹{d.get('last','—')}")
                    c2.metric("Target",f"₹{d.get('tgt','—')}")
                    c3.metric("Stop Loss",f"₹{d.get('sl','—')}")
                    c4.metric("Risk:Reward",d.get('rr','—'))
                    st.markdown(f"**Buy Rules:** {d.get('buy_passed',0)}/{d.get('buy_total',25)} &nbsp;&nbsp; **Sell Rules:** {d.get('sell_passed',0)}/{d.get('sell_total',25)}")
                    st.markdown(f"**NIFTY:** {d.get('nifty_lbl','—')} &nbsp;&nbsp; **RVOL:** {d.get('rvol',0)}x &nbsp;&nbsp; **News:** {d.get('news_lbl','—')}")
        else:
            st.info("🔍 No Perfect Trades today — market conditions not aligned yet. Check Strong signals below.")

        if strong:
            st.markdown(f"### 🟡 Strong Signals (≥{min_rules}/25 rules) — {len(strong)} stocks")
            rows=[]
            for d in strong:
                bp=d.get('buy_passed',0); sp=d.get('sell_passed',0)
                side="🟢 BUY candidate" if bp>sp else "🔴 SELL candidate"
                rows.append({"Symbol":d['symbol'],"LTP":f"₹{d.get('last','—')}",
                             "Signal":d.get('sig_e','—'),"Buy Rules":f"{bp}/25","Sell Rules":f"{sp}/25",
                             "Action":d.get('action','—'),"NIFTY":d.get('nifty_lbl','—'),"RVOL":f"{d.get('rvol',0)}x"})
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    # ── TAB 2: All Results ───────────────────────────────────────────────────
    with tab2:
        rows=[]
        for d in results:
            if d.get('error'): continue
            if not show_hold and d.get('sig')=='HOLD': continue
            rows.append({
                "Symbol":d['symbol'],"Exch":d.get('sfx',''),
                "LTP ₹":d.get('last'),
                "Chg %":f"{'+' if d.get('chg',0)>=0 else ''}{d.get('chg',0)}%",
                "Signal":d.get('sig_e'),
                "Buy":f"{d.get('buy_passed',0)}/25","Sell":f"{d.get('sell_passed',0)}/25",
                "Next Day":d.get('action'),
                "Conf%":f"{d.get('conf',0):.0f}%",
                "Entry":d.get('last'),"Target":d.get('tgt'),"SL":d.get('sl'),"R:R":d.get('rr'),
                "RSI":d.get('rsi'),"ADX":d.get('adx'),
                "NIFTY":d.get('nifty_lbl'),"RVOL":f"{d.get('rvol',0)}x",
                "News":d.get('news_lbl'),"Delivery":d.get('delivery'),
            })
        if rows:
            df_res=pd.DataFrame(rows)
            st.dataframe(df_res,use_container_width=True,hide_index=True,
                        column_config={
                            "Signal":st.column_config.TextColumn(width="small"),
                            "Next Day":st.column_config.TextColumn(width="medium"),
                        })
        else:
            st.info("No results to show.")

    # ── TAB 3: v5 Signals ───────────────────────────────────────────────────
    with tab3:
        st.markdown("#### 🆕 New v5 Indicators — Extra confirmation layers")
        rows=[]
        for d in results:
            if d.get('error'): continue
            rows.append({
                "Symbol":d['symbol'],"LTP":d.get('last'),
                "NIFTY":d.get('nifty_lbl'),
                "RVOL":f"{d.get('rvol',0)}x",
                "PDH ₹":d.get('pdh','—'),"PDL ₹":d.get('pdl','—'),
                "3C Bull":f"{'✅' if d.get('cbull') else str(d.get('cbullc',0))+'/3'}",
                "3C Bear":f"{'✅' if d.get('cbear') else str(d.get('cbearc',0))+'/3'}",
                "Upper Wick":f"{d.get('uw',0)}%","Lower Wick":f"{d.get('lw',0)}%",
                "ATR Mom":f"{d.get('atr_m',0)}x {d.get('atr_d','')}",
                "1H Trend":d.get('mtf1h'),"15m Trend":d.get('mtf15'),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    # ── TAB 4: Checklist ────────────────────────────────────────────────────
    with tab4:
        st.markdown("#### ✅ Full 25-Rule Checklist per Stock")
        sym_list=[d['symbol'] for d in results if not d.get('error')]
        if sym_list:
            sel=st.selectbox("Stock select cheyyi",sym_list)
            d=next((x for x in results if x.get('symbol')==sel),None)
            if d:
                st.markdown(f"**{sel}** ₹{d.get('last','—')} | {d.get('sig_e','—')} | Buy: {d.get('buy_passed',0)}/25 | Sell: {d.get('sell_passed',0)}/25")
                col1,col2=st.columns(2)
                with col1:
                    st.markdown("**🟢 BUY Checklist**")
                    buy_rows=[{"Rule":r,"Status":"✅ Pass" if ok else "❌ Fail","Detail":det}
                              for r,(ok,det) in d.get('buy_checks',{}).items()]
                    if buy_rows:
                        df_b=pd.DataFrame(buy_rows)
                        st.dataframe(df_b,use_container_width=True,hide_index=True,
                                    column_config={"Status":st.column_config.TextColumn(width="small")})
                with col2:
                    st.markdown("**🔴 SELL Checklist**")
                    sell_rows=[{"Rule":r,"Status":"✅ Pass" if ok else "❌ Fail","Detail":det}
                               for r,(ok,det) in d.get('sell_checks',{}).items()]
                    if sell_rows:
                        df_s=pd.DataFrame(sell_rows)
                        st.dataframe(df_s,use_container_width=True,hide_index=True,
                                    column_config={"Status":st.column_config.TextColumn(width="small")})

    # ── TAB 5: Export ────────────────────────────────────────────────────────
    with tab5:
        st.markdown("#### 💾 Export Results to Excel")
        rows=[]
        for d in results:
            rows.append({
                "Symbol":d.get('symbol'),"Exchange":d.get('sfx'),
                "LTP":d.get('last'),"Chg%":d.get('chg'),"Gap%":d.get('gap'),
                "Signal":d.get('sig'),"RSI":d.get('rsi'),"ADX":d.get('adx'),
                "EMA9":d.get('ema9'),"EMA21":d.get('ema21'),
                "VWAP":d.get('vwap'),"CMF":d.get('cmf'),"OBV":d.get('obv'),
                "Delivery":d.get('delivery'),"Vol_Ratio":d.get('vol_r'),
                "NIFTY":d.get('nifty_lbl'),"RVOL":d.get('rvol'),
                "Prev_Day_H":d.get('pdh'),"Prev_Day_L":d.get('pdl'),
                "3C_Bull":d.get('cbullc'),"3C_Bear":d.get('cbearc'),
                "Upper_Wick%":d.get('uw'),"Lower_Wick%":d.get('lw'),
                "ATR_Momentum":d.get('atr_m'),"ATR_Dir":d.get('atr_d'),
                "Trend_1H":d.get('mtf1h'),"Trend_15m":d.get('mtf15'),
                "News":d.get('news_lbl'),
                "Buy_Rules":f"{d.get('buy_passed',0)}/25","Sell_Rules":f"{d.get('sell_passed',0)}/25",
                "Perfect_BUY":d.get('perfect_buy'),"Perfect_SELL":d.get('perfect_sell'),
                "Action":d.get('action'),"Confidence%":d.get('conf'),
                "Entry":d.get('last'),"Target":d.get('tgt'),
                "Stop_Loss":d.get('sl'),"Risk_Reward":d.get('rr'),
            })
        df_exp=pd.DataFrame(rows)
        fname=f"scan_v5_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        df_exp.to_excel(fname,index=False)
        with open(fname,"rb") as f:
            st.download_button("📥 Download Excel",f,file_name=fname,
                              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                              use_container_width=True)
        st.dataframe(df_exp,use_container_width=True,hide_index=True)

if __name__=="__main__":
    main()
