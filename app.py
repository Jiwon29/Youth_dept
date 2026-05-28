"""
청년 부채는 왜 누구에겐 자산이, 누구에겐 함정이 되었는가
대시보드 3 : 같은 부채, 소득분위별 다른 결과

실행 방법:
    streamlit run app.py
"""

import os
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

# ───────────────────────────────────────────────
# 0. 페이지 기본 설정
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="청년 부채 대시보드",
    page_icon="📊",
    layout="wide",
)

# ───────────────────────────────────────────────
# 1. 색상 팔레트 & 공통 스타일
# ───────────────────────────────────────────────
# 하위 집단 = 차분한 청록, 상위 집단 = 강조되는 주황
COLOR_LOW  = "#4C9BE8"   # 하위(Q1+Q2) - 파란 계열
COLOR_HIGH = "#F07B39"   # 상위(Q4+Q5) - 주황 계열
COLOR_GREY = "#B0BAC4"
YEAR_COLORS = {2018: "#A8C4E0", 2021: "#4C9BE8", 2023: "#1A5FAD"}
Q_COLORS = {
    1: "#D9EAF7", 2: "#A8C4E0",
    3: "#7BAFD4", 4: "#F5C086",
    5: "#F07B39",
}

# ───────────────────────────────────────────────
# 2. DB 연결 함수 (에러 처리 포함)
# ───────────────────────────────────────────────
DB_FILENAME = "youth_debt_분석내용포함.db"

@st.cache_resource
def get_connection():
    """DB 파일을 찾아 연결을 반환합니다. 없으면 None을 반환합니다."""
    # 현재 폴더 → 업로드 폴더 순서로 탐색
    candidates = [
        DB_FILENAME,
        os.path.join(os.path.dirname(__file__), DB_FILENAME),
    ]
    for path in candidates:
        if os.path.exists(path):
            return sqlite3.connect(path, check_same_thread=False)
    return None

conn = get_connection()

# ─── DB 없을 때 친절한 안내 화면 ───────────────
if conn is None:
    st.error("⚠️ 데이터베이스 파일을 찾을 수 없습니다.")
    st.markdown(
        f"""
        ### 해결 방법

        `app.py` 파일과 **같은 폴더**에 아래 파일을 넣어 주세요.

        ```
        📁 내 프로젝트 폴더
        ├── app.py                          ← 이 파일
        └── {DB_FILENAME}   ← ★ 여기에 넣으세요
        ```

        파일 이름이 정확히 `{DB_FILENAME}` 인지 확인해 주세요.  
        (띄어쓰기, 한글, 확장자 `.db` 모두 포함)

        파일을 넣은 뒤 터미널에서 다시 실행하세요.
        ```bash
        streamlit run app.py
        ```
        """
    )
    st.stop()   # 이하 코드는 실행하지 않습니다

# ───────────────────────────────────────────────
# 3. 데이터 로딩 함수
# ───────────────────────────────────────────────
@st.cache_data
def load_ttest():
    return pd.read_sql("SELECT * FROM ttest_results", conn)

@st.cache_data
def load_ols():
    return pd.read_sql("SELECT * FROM ols_results", conn)

@st.cache_data
def load_kgss():
    return pd.read_sql("SELECT * FROM kgss_summary", conn)

@st.cache_data
def load_hfws_sample():
    """박스플롯·고위험 집계에 필요한 컬럼만 로드합니다."""
    return pd.read_sql(
        """
        SELECT year,
               소득분위_숫자,
               순자산,
               고위험_플래그,
               부채_금융부채
        FROM hfws_youth
        """,
        conn,
    )

@st.cache_data
def load_median_join():
    """
    순자산 중앙값과 KGSS를 연도 기준 LEFT JOIN합니다.
    SQLite는 MEDIAN()을 지원하지 않으므로
    Python에서 계산 후 kgss_summary와 병합합니다.
    """
    hf = pd.read_sql("SELECT year, 순자산 FROM hfws_youth", conn)
    kgss = pd.read_sql("SELECT * FROM kgss_summary", conn)

    median_df = (
        hf.groupby("year")["순자산"]
        .median()
        .reset_index()
        .rename(columns={"순자산": "순자산_중앙값"})
    )
    merged = median_df.merge(kgss, on="year", how="left")
    return merged


# ───────────────────────────────────────────────
# 4. 헤더
# ───────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2rem; font-weight: 800;
        color: #1A2B45; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem; color: #5A6A7E;
        margin-bottom: 1.5rem;
    }
    .kpi-box {
        background: #F0F5FB;
        border-left: 5px solid #4C9BE8;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .kpi-box.high {
        border-left-color: #F07B39;
        background: #FEF4EC;
    }
    .kpi-label { font-size: 0.78rem; color: #7A8899; font-weight: 600; }
    .kpi-value { font-size: 1.75rem; font-weight: 800; color: #1A2B45; }
    .kpi-sub   { font-size: 0.78rem; color: #5A6A7E; margin-top: 0.1rem; }
    .sig-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .sig-yes { background: #D4EDDA; color: #155724; }
    .sig-no  { background: #F8D7DA; color: #721C24; }
    .section-title {
        font-size: 1.15rem; font-weight: 700;
        color: #1A2B45; margin-top: 2.5rem;
        border-bottom: 2px solid #E2EAF4;
        padding-bottom: 0.4rem;
    }
    .insight-box {
        background: #F7F9FC;
        border: 1px solid #DAE4F0;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        font-size: 0.88rem;
        color: #374151;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">📊 같은 부채, 소득분위별 다른 결과</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">청년 부채는 왜 누구에겐 자산이, 누구에겐 함정이 되었는가 — '
    '가계금융복지조사 2018·2021·2023 × KGSS</p>',
    unsafe_allow_html=True,
)

st.divider()

# ───────────────────────────────────────────────
# 5. 차트 1 — KPI 수치 카드
# ───────────────────────────────────────────────
st.markdown('<p class="section-title">① 핵심 수치 요약</p>', unsafe_allow_html=True)

ttest = load_ttest()
ols   = load_ols()

# 필요한 수치 추출
row_high_18_21 = ttest[(ttest["group"] == "상위(Q4+Q5)") &
                        (ttest["year_from"] == 2018) &
                        (ttest["year_to"] == 2021)].iloc[0]
row_low_18_21  = ttest[(ttest["group"] == "하위(Q1+Q2)") &
                        (ttest["year_from"] == 2018) &
                        (ttest["year_to"] == 2021)].iloc[0]

coef_age_low  = ols[(ols["group"] == "하위(Q1+Q2)") & (ols["var"] == "연령")].iloc[0]["coef"]
coef_age_high = ols[(ols["group"] == "상위(Q4+Q5)") & (ols["var"] == "연령")].iloc[0]["coef"]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="kpi-box high">
          <div class="kpi-label">영끌 시기 상위 집단 순자산 변화 (2018→2021)</div>
          <div class="kpi-value">+{int(row_high_18_21['diff']):,}만원</div>
          <div class="kpi-sub">
            통계적으로 유의 <span class="sig-badge sig-yes">p&lt;0.001 ★★★</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="kpi-box">
          <div class="kpi-label">영끌 시기 하위 집단 순자산 변화 (2018→2021)</div>
          <div class="kpi-value">{int(row_low_18_21['diff']):,}만원</div>
          <div class="kpi-sub">
            통계적으로 유의하지 않음 <span class="sig-badge sig-no">n.s.</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="kpi-box high">
          <div class="kpi-label">연령 1세 증가 시 순자산 증가 (회귀계수)</div>
          <div class="kpi-value">하위 +{int(coef_age_low):,}만 vs 상위 +{int(coef_age_high):,}만</div>
          <div class="kpi-sub">
            나이가 들수록 자산 격차가 구조적으로 더 벌어집니다
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="insight-box">💡 같은 영끌 시기, 빚을 낸 건 모두 같았지만 '
    '<b>자산이 늘어난 것은 상위 집단뿐</b>이었습니다. '
    '연령 계수 격차(535 vs 2,003만원)는 나이가 들수록 두 집단의 격차가 구조적으로 벌어짐을 의미합니다.</div>',
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────
# 6. 차트 2 — 소득분위별 순자산 변화량 바차트 (핵심)
# ───────────────────────────────────────────────
st.markdown('<p class="section-title">② 소득분위별 순자산 변화량 — 영끌 시기 비교 (t-test 검증)</p>', unsafe_allow_html=True)

# 2018→2021 데이터만 사용
ttest_1821 = ttest[ttest["year_from"] == 2018].copy()

fig2 = go.Figure()

colors_bar = []
for _, row in ttest_1821.iterrows():
    if "상위" in row["group"]:
        colors_bar.append(COLOR_HIGH)
    else:
        colors_bar.append(COLOR_LOW)

fig2.add_trace(go.Bar(
    x=ttest_1821["group"],
    y=ttest_1821["diff"],
    marker_color=colors_bar,
    text=[
        f"<b>{int(d):+,}만원</b><br>{s}"
        for d, s in zip(ttest_1821["diff"], ttest_1821["sig"])
    ],
    textposition="outside",
    textfont=dict(size=13),
    width=0.45,
    hovertemplate=(
        "<b>%{x}</b><br>"
        "변화량: %{y:+,.0f}만원<br>"
        "<extra></extra>"
    ),
))

# 기준선 (0)
fig2.add_hline(y=0, line_color="#888", line_width=1.2)

# 유의성 별표 주석
for _, row in ttest_1821.iterrows():
    if row["sig"] != "n.s.":
        fig2.add_annotation(
            x=row["group"],
            y=row["diff"] + 200,
            text=row["sig"],
            showarrow=False,
            font=dict(size=16, color="#155724"),
        )

fig2.update_layout(
    height=420,
    title=dict(
        text="2018→2021 순자산 변화량 (영끌 시기)",
        font=dict(size=15, color="#1A2B45"),
    ),
    yaxis=dict(
        title="순자산 변화량 (만원)",
        zeroline=True,
        zerolinewidth=1.5,
        zerolinecolor="#888",
    ),
    xaxis=dict(title="소득 집단"),
    plot_bgcolor="white",
    showlegend=False,
    margin=dict(t=60, b=40, l=60, r=40),
)
fig2.update_xaxes(showgrid=False)
fig2.update_yaxes(showgrid=True, gridcolor="#E8EFF6")

st.plotly_chart(fig2, use_container_width=True)

# 전체 3구간 비교 테이블
st.caption("📋 전체 비교 구간 (t-test 결과)")
ttest_display = ttest[["group", "year_from", "year_to", "diff", "p_val", "sig"]].copy()
ttest_display.columns = ["집단", "비교 시작", "비교 종료", "변화량(만원)", "p값", "유의성"]
ttest_display["변화량(만원)"] = ttest_display["변화량(만원)"].apply(lambda x: f"{int(x):+,}")
ttest_display["p값"] = ttest_display["p값"].apply(lambda x: f"{x:.4f}")
st.dataframe(ttest_display, use_container_width=True, hide_index=True)

st.markdown(
    '<div class="insight-box">💡 <b>같은 영끌 시기에 상위 집단만 +7,994만원 순자산이 증가했습니다(p&lt;0.001).</b> '
    '하위 집단의 변화량(-40만원)은 통계적으로 의미 없는 수준(n.s.)입니다. '
    '빚은 똑같이 냈지만, 결과는 완전히 달랐습니다.</div>',
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────
# 7. 차트 3 — 소득분위별 순자산 박스플롯
# ───────────────────────────────────────────────
st.markdown('<p class="section-title">③ 소득분위별 순자산 분포 — 박스플롯 (2018·2021·2023)</p>', unsafe_allow_html=True)

hfws = load_hfws_sample()

# 이상치 제거: 시각화 가독성을 위해 99% 분위수로 클리핑
q99 = hfws["순자산"].quantile(0.99)
hfws_clip = hfws[hfws["순자산"] <= q99].copy()
hfws_clip["소득분위"] = hfws_clip["소득분위_숫자"].map(
    {1: "Q1(하위)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5(상위)"}
)

# 소득분위 선택 필터
selected_q = st.multiselect(
    "소득분위 선택 (기본: 전체)",
    options=["Q1(하위)", "Q2", "Q3", "Q4", "Q5(상위)"],
    default=["Q1(하위)", "Q2", "Q3", "Q4", "Q5(상위)"],
)
if not selected_q:
    selected_q = ["Q1(하위)", "Q2", "Q3", "Q4", "Q5(상위)"]

hfws_filtered = hfws_clip[hfws_clip["소득분위"].isin(selected_q)]

fig3 = go.Figure()

year_label = {2018: "2018년 (코로나 이전)", 2021: "2021년 (영끌 시기)", 2023: "2023년 (회복기)"}
year_color_list = [YEAR_COLORS[2018], YEAR_COLORS[2021], YEAR_COLORS[2023]]

for i, yr in enumerate([2018, 2021, 2023]):
    sub = hfws_filtered[hfws_filtered["year"] == yr]
    fig3.add_trace(go.Box(
        x=sub["소득분위"],
        y=sub["순자산"],
        name=year_label[yr],
        marker_color=year_color_list[i],
        boxmean=True,       # 평균값 마름모 표시
        offsetgroup=str(yr),
        hovertemplate=(
            f"<b>{yr}년</b><br>"
            "소득분위: %{x}<br>"
            "순자산: %{y:,.0f}만원<br>"
            "<extra></extra>"
        ),
    ))

fig3.update_layout(
    height=480,
    title=dict(text="소득분위 × 연도별 순자산 분포 (상위 1% 이상치 제거)", font=dict(size=15, color="#1A2B45")),
    yaxis=dict(title="순자산 (만원)", gridcolor="#E8EFF6"),
    xaxis=dict(title="소득분위"),
    boxmode="group",
    plot_bgcolor="white",
    legend=dict(orientation="h", y=1.08),
    margin=dict(t=70, b=40, l=60, r=40),
)

st.plotly_chart(fig3, use_container_width=True)
st.markdown(
    '<div class="insight-box">💡 Q5(상위)는 2021년 박스 전체가 위로 크게 이동한 반면, '
    'Q1(하위)은 3개년 내내 같은 자리에 머물러 있습니다. '
    '<b>격차는 단순한 수치 차이가 아니라 구조적 불균형</b>임을 분포 전체가 보여줍니다.</div>',
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────
# 8. 차트 4 — 고위험 가구 비율 바차트
# ───────────────────────────────────────────────
st.markdown('<p class="section-title">④ 소득분위별 고위험 가구 비율 (DSR 40% 초과)</p>', unsafe_allow_html=True)

# 부채 보유 가구만 집계
hfws_debt = hfws[hfws["부채_금융부채"] > 0].copy()
risk_df = (
    hfws_debt.groupby(["year", "소득분위_숫자"])
    .agg(고위험=("고위험_플래그", "sum"), 전체=("고위험_플래그", "count"))
    .reset_index()
)
risk_df["비율(%)"] = (risk_df["고위험"] / risk_df["전체"] * 100).round(1)
risk_df["소득분위"] = risk_df["소득분위_숫자"].map(
    {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5"}
)

fig4 = px.bar(
    risk_df,
    x="소득분위",
    y="비율(%)",
    color="year",
    barmode="group",
    color_discrete_map=YEAR_COLORS,
    text="비율(%)",
    title="부채 보유 청년 가구 중 DSR 40% 초과 비율",
    labels={"비율(%)": "고위험 가구 비율 (%)", "year": "연도"},
    height=400,
)
fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig4.update_layout(
    plot_bgcolor="white",
    yaxis=dict(gridcolor="#E8EFF6"),
    margin=dict(t=60, b=40, l=60, r=40),
    legend=dict(title="연도", orientation="h", y=1.08),
)
st.plotly_chart(fig4, use_container_width=True)
st.markdown(
    '<div class="insight-box">💡 2023년 고위험 가구 비율이 전반적으로 상승했으며, '
    '특히 Q5(상위)에서 2.4%로 높게 나타납니다. '
    '이는 상위 집단이 더 큰 규모의 담보대출을 활용한 데 따른 상환 부담을 반영합니다. '
    '<b>빚을 감당하지 못할 위험은 2023년 이후 전 계층으로 확산</b>되는 추세입니다.</div>',
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────
# 9. 차트 5 — 집단별 회귀계수 바차트
# ───────────────────────────────────────────────
st.markdown('<p class="section-title">⑤ 집단별 회귀계수 — 순자산에 영향을 미치는 요인</p>', unsafe_allow_html=True)

ols_data = load_ols()
ols_data["유의여부"] = ols_data["sig"].apply(lambda x: "유의" if x != "n.s." else "비유의")
ols_data["opacity"] = ols_data["sig"].apply(lambda x: 1.0 if x != "n.s." else 0.3)
ols_data["집단_색"] = ols_data["group"].apply(
    lambda x: COLOR_HIGH if "상위" in x else COLOR_LOW
)
ols_data["var_label"] = ols_data["var"].map(
    {"금융부채_만": "금융부채(만원)", "연령": "연령(세)", "수도권": "수도권 거주"}
)

fig5 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("하위 집단 (Q1+Q2)", "상위 집단 (Q4+Q5)"),
    shared_yaxes=True,
)

for col_idx, group_name in enumerate(["하위(Q1+Q2)", "상위(Q4+Q5)"], start=1):
    sub = ols_data[ols_data["group"] == group_name]
    bar_color = COLOR_LOW if "하위" in group_name else COLOR_HIGH

    opacities = sub["opacity"].tolist()
    colors_with_opacity = [
        bar_color if op == 1.0 else COLOR_GREY
        for op in opacities
    ]

    fig5.add_trace(
        go.Bar(
            x=sub["var_label"],
            y=sub["coef"],
            marker_color=colors_with_opacity,
            text=[
                f"{c:,.0f}<br>{s}" for c, s in zip(sub["coef"], sub["sig"])
            ],
            textposition="outside",
            name=group_name,
            showlegend=False,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "회귀계수: %{y:,.1f}만원<br>"
                "<extra></extra>"
            ),
        ),
        row=1, col=col_idx,
    )

fig5.update_layout(
    height=430,
    title=dict(text="집단별 순자산 회귀계수 (회색 = 유의하지 않음)", font=dict(size=15, color="#1A2B45")),
    plot_bgcolor="white",
    margin=dict(t=70, b=40, l=60, r=40),
)
fig5.update_yaxes(title_text="회귀계수 (만원)", gridcolor="#E8EFF6", row=1, col=1)
fig5.update_xaxes(showgrid=False)
st.plotly_chart(fig5, use_container_width=True)

# 회귀계수 표
r2_low  = ols_data[ols_data["group"] == "하위(Q1+Q2)"]["r2"].iloc[0]
r2_high = ols_data[ols_data["group"] == "상위(Q4+Q5)"]["r2"].iloc[0]
st.caption(f"📋 하위 집단 R² = {r2_low:.4f} | 상위 집단 R² = {r2_high:.4f}")

st.markdown(
    '<div class="insight-box">💡 <b>연령 계수</b>가 집단 간 격차를 가장 뚜렷하게 보여줍니다. '
    '하위 집단은 1살당 +535만원, 상위 집단은 +2,003만원으로 '
    '<b>나이가 들수록 자산 격차가 구조적으로 벌어집니다.</b> '
    '수도권 거주 프리미엄도 상위 집단(+5,060만원)에서만 통계적으로 유의합니다.</div>',
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────
# 10. 차트 6 — 순자산 중앙값 × 가계만족도 이중축
# ───────────────────────────────────────────────
st.markdown('<p class="section-title">⑥ 순자산 중앙값 vs 가계만족도 (KGSS SATFIN)</p>', unsafe_allow_html=True)

merged = load_median_join()

fig6 = make_subplots(specs=[[{"secondary_y": True}]])

fig6.add_trace(
    go.Scatter(
        x=merged["year"], y=merged["순자산_중앙값"],
        name="순자산 중앙값(만원)",
        mode="lines+markers+text",
        line=dict(color=COLOR_LOW, width=3),
        marker=dict(size=9),
        text=[f"{int(v):,}만" for v in merged["순자산_중앙값"]],
        textposition="top center",
        hovertemplate="<b>%{x}년</b><br>순자산 중앙값: %{y:,.0f}만원<extra></extra>",
    ),
    secondary_y=False,
)

fig6.add_trace(
    go.Scatter(
        x=merged["year"], y=merged["avg_satfin"],
        name="가계만족도 (1=만족 ↔ 5=불만족)",
        mode="lines+markers+text",
        line=dict(color=COLOR_HIGH, width=3, dash="dot"),
        marker=dict(size=9),
        text=[f"{v:.3f}" for v in merged["avg_satfin"]],
        textposition="bottom center",
        hovertemplate="<b>%{x}년</b><br>가계만족도 평균: %{y:.3f}<extra></extra>",
    ),
    secondary_y=True,
)

fig6.update_layout(
    height=400,
    title=dict(text="순자산 중앙값과 가계만족도 추이 (2018·2021·2023)", font=dict(size=15, color="#1A2B45")),
    plot_bgcolor="white",
    legend=dict(orientation="h", y=1.1),
    margin=dict(t=70, b=40, l=60, r=80),
    xaxis=dict(tickvals=[2018, 2021, 2023], gridcolor="#E8EFF6"),
)
fig6.update_yaxes(title_text="순자산 중앙값 (만원)", secondary_y=False, gridcolor="#E8EFF6")
fig6.update_yaxes(
    title_text="SATFIN 평균 (↑ 높을수록 불만족)",
    secondary_y=True,
    showgrid=False,
    range=[2.5, 3.5],
)

st.plotly_chart(fig6, use_container_width=True)
st.markdown(
    '<div class="insight-box">💡 순자산 중앙값이 13,026만원(2018) → 12,516만원(2021) → 11,654만원(2023)으로 '
    '3개년 연속 하락하는 동안, 가계만족도 역시 2.865 → 3.050으로 악화되었습니다. '
    '<b>객관적 자산과 주관적 체감이 모두 나빠졌습니다.</b></div>',
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────
# 11. 차트 7 — 순자산 중앙값 × 미래전망 이중축
# ───────────────────────────────────────────────
st.markdown('<p class="section-title">⑦ 순자산 중앙값 vs 미래 경제 전망 (KGSS FINPROS)</p>', unsafe_allow_html=True)

# FINPROS는 2021·2023만 유효
merged_fin = merged[merged["avg_finpros"].notna()].copy()

fig7 = make_subplots(specs=[[{"secondary_y": True}]])

fig7.add_trace(
    go.Scatter(
        x=merged_fin["year"], y=merged_fin["순자산_중앙값"],
        name="순자산 중앙값(만원)",
        mode="lines+markers+text",
        line=dict(color=COLOR_LOW, width=3),
        marker=dict(size=9),
        text=[f"{int(v):,}만" for v in merged_fin["순자산_중앙값"]],
        textposition="top center",
        hovertemplate="<b>%{x}년</b><br>순자산 중앙값: %{y:,.0f}만원<extra></extra>",
    ),
    secondary_y=False,
)

fig7.add_trace(
    go.Scatter(
        x=merged_fin["year"], y=merged_fin["avg_finpros"],
        name="미래전망 (1=좋아질 것 ↔ 5=나빠질 것)",
        mode="lines+markers+text",
        line=dict(color="#9B59B6", width=3, dash="dot"),
        marker=dict(size=9),
        text=[f"{v:.3f}" for v in merged_fin["avg_finpros"]],
        textposition="bottom center",
        hovertemplate="<b>%{x}년</b><br>미래전망 평균: %{y:.3f}<extra></extra>",
    ),
    secondary_y=True,
)

fig7.update_layout(
    height=400,
    title=dict(text="순자산 중앙값과 미래 경제 전망 (2021·2023)", font=dict(size=15, color="#1A2B45")),
    plot_bgcolor="white",
    legend=dict(orientation="h", y=1.1),
    margin=dict(t=70, b=40, l=60, r=80),
    xaxis=dict(tickvals=[2021, 2023], gridcolor="#E8EFF6"),
)
fig7.update_yaxes(title_text="순자산 중앙값 (만원)", secondary_y=False, gridcolor="#E8EFF6")
fig7.update_yaxes(
    title_text="FINPROS 평균 (↑ 높을수록 비관적)",
    secondary_y=True,
    showgrid=False,
    range=[2.0, 3.5],
)

st.plotly_chart(fig7, use_container_width=True)
st.markdown(
    '<div class="insight-box">💡 2023년 순자산 중앙값이 하락하는 동안 미래 전망 지수도 거의 개선되지 않았습니다(2.534 → 2.503). '
    '⑥번 차트와 함께, <b>"현재도 나쁘고, 미래도 밝지 않다"</b>는 청년 청년 재정 인식을 보여줍니다.</div>',
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────
# 12. 푸터
# ───────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div style="text-align:center; color:#8A96A3; font-size:0.8rem; margin-top:1rem;">
    데이터 출처: 통계청 가계금융복지조사(2018·2021·2023) × KGSS 한국종합사회조사<br>
    청년 = 가구주 만 39세 이하 | 경영정보처리론 8조
    </div>
    """,
    unsafe_allow_html=True,
)
