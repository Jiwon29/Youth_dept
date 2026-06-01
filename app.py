"""
청년 부채는 왜 누구에겐 자산이, 누구에겐 함정이 되었는가
통합 대시보드 — 경영정보처리론 8조
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────
# 0. 페이지 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="청년 부채 양극화 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# 1. 전역 CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* 전체 배경 흰색 */
.main, [data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
}
[data-testid="stSidebar"] {
    background-color: #f0f4fa;
}
/* 제목 */
.page-title {
    font-size: 1.9rem; font-weight: 800;
    color: #1A2B45; margin-bottom: 0.15rem;
}
.page-subtitle {
    font-size: 1.0rem; color: #5A6A7E; margin-bottom: 0.5rem;
}
/* 섹션 구분선 제목 */
.section-title {
    font-size: 1.1rem; font-weight: 700;
    color: #1A2B45; margin-top: 2rem;
    border-bottom: 2px solid #3B82F6;
    padding-bottom: 0.35rem; margin-bottom: 0.8rem;
}
/* KPI 카드 */
.kpi-box {
    background: #EFF6FF;
    border-left: 5px solid #3B82F6;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
}
.kpi-box.red {
    border-left-color: #EF4444;
    background: #FEF2F2;
}
.kpi-box.orange {
    border-left-color: #F07B39;
    background: #FEF4EC;
}
.kpi-label { font-size: 0.78rem; color: #7A8899; font-weight: 600; }
.kpi-value { font-size: 1.75rem; font-weight: 800; color: #1A2B45; }
.kpi-sub   { font-size: 0.78rem; color: #5A6A7E; margin-top: 0.1rem; }
.sig-badge {
    display: inline-block; padding: 0.15rem 0.5rem;
    border-radius: 4px; font-size: 0.72rem; font-weight: 700;
}
.sig-yes { background: #D1FAE5; color: #065F46; }
.sig-no  { background: #FEE2E2; color: #991B1B; }
/* 인사이트 박스 */
.insight-box {
    background: #F8FAFF;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #374151;
    margin-top: 0.5rem;
    margin-bottom: 0.3rem;
}
/* 결론 박스 */
.conclusion-box {
    background: #FFFBEB;
    border-left: 5px solid #F59E0B;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    color: #374151;
    margin-top: 1.5rem;
}
/* 사이드바 버튼 */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.3rem;
    border: none;
}
/* 데이터프레임 헤더 */
[data-testid="stDataFrame"] thead th {
    background-color: #EFF6FF !important;
    color: #1A2B45 !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 2. DB 연결
# ─────────────────────────────────────────────────────────────
DB_FILE = "youth_debt_integrated.db"

@st.cache_resource
def get_conn():
    candidates = [
        DB_FILE,
        os.path.join(os.path.dirname(__file__), DB_FILE),
    ]
    for p in candidates:
        if os.path.exists(p):
            return sqlite3.connect(p, check_same_thread=False)
    return None

conn = get_conn()

if conn is None:
    st.error(f"❌ 데이터베이스 파일 '{DB_FILE}'을 찾을 수 없습니다. app.py와 같은 폴더에 넣어 주세요.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# 3. 공통 유틸
# ─────────────────────────────────────────────────────────────
COLOR_BLUE   = "#3B82F6"
COLOR_RED    = "#EF4444"
COLOR_ORANGE = "#F07B39"
COLOR_GREEN  = "#10B981"
COLOR_GREY   = "#9CA3AF"
COLOR_Q1     = "#EF4444"   # 1분위 강조 — 빨강
COLOR_Q5     = "#1D4ED8"   # 5분위 강조 — 진파랑
Q_COLORS = {1: COLOR_Q1, 2: "#93C5FD", 3: "#6B7280", 4: "#60A5FA", 5: COLOR_Q5}
YEAR_COLORS  = {2018: "#A8C4E0", 2021: "#3B82F6", 2023: "#1A2B45"}

def weighted_median(df, val_col, weight_col):
    if df.empty:
        return 0.0
    df_s = df.sort_values(val_col).copy()
    cs = df_s[weight_col].cumsum()
    cutoff = df_s[weight_col].sum() / 2.0
    return float(df_s[cs >= cutoff].iloc[0][val_col])

@st.cache_data
def load_table(query):
    return pd.read_sql(query, conn)

# ─────────────────────────────────────────────────────────────
# 4. 사이드바 네비게이션
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 청년 부채 분석")
    st.markdown("**경영정보처리론 8조**")
    st.markdown("---")
    st.markdown("### 📌 대시보드 이동")

    if "page" not in st.session_state:
        st.session_state.page = 1

    def nav(n):
        st.session_state.page = n

    pages = {
        1: "📈 대시보드 1\n청년 부채 현상 파악",
        2: "💰 대시보드 2\n부채 속 자산은 쌓였는가?",
        3: "🔬 대시보드 3\n같은 부채, 다른 결과",
    }
    for n, label in pages.items():
        active = st.session_state.page == n
        btn_style = "primary" if active else "secondary"
        if st.button(label, key=f"nav_{n}", type=btn_style):
            nav(n)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.78rem;color:#7A8899'>"
        "데이터 출처<br>"
        "• 통계청 가계금융복지조사<br>"
        "• 한국은행 ECOS<br>"
        "• KGSS 한국종합사회조사<br>"
        "청년 = 가구주 만 39세 이하"
        "</div>",
        unsafe_allow_html=True,
    )

page = st.session_state.page

# ═══════════════════════════════════════════════════════════════
#  대시보드 1 — 청년 부채 현상 파악
# ═══════════════════════════════════════════════════════════════
if page == 1:

    st.markdown('<p class="page-title">📈 대시보드 1: 청년 부채 현상 파악</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">저금리 시대, 청년들은 왜 빚을 냈는가 — 거시 금리 흐름과 청년 부채 급증의 상관관계 추적</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── KPI 카드 ────────────────────────────────────────────────
    st.markdown('<p class="section-title">① 핵심 지표 요약</p>', unsafe_allow_html=True)

    query_kpi1 = """
SELECT ((avg_2021 - avg_2018) / avg_2018) * 100 AS growth_rate
FROM
  (SELECT AVG(financial_debt) AS avg_2018
   FROM table_household_master WHERE year = 2018 AND age <= 39) a,
  (SELECT AVG(financial_debt) AS avg_2021
   FROM table_household_master WHERE year = 2021 AND age <= 39) b
"""
    growth_rate = load_table(query_kpi1).iloc[0, 0]

    query_kpi2 = """
SELECT loan_balance FROM table_youth_loan
WHERE year_quarter = '2026/Q1' AND age_group = '30대'
"""
    try:
        latest_loan_raw = load_table(query_kpi2).iloc[0, 0]
        # loan_balance 단위: 십만원 → 억원으로 환산
        loan_eok = latest_loan_raw / 1000  # 십만원 → 억원
        real_loan_str = f"약 {loan_eok:.1f}억원"
    except Exception:
        real_loan_str = "데이터 없음"

    query_kpi3 = """
SELECT
  CASE
    WHEN reason_code = 1.0 THEN '거주주택 구입 (부동산 영끌)'
    WHEN reason_code = 4.0 THEN '부동산 이외 자산투자/사업자금'
    WHEN reason_code = 9.0 THEN '기타 용도 및 생활비'
    ELSE '기타 사유'
  END AS top_reason
FROM table_household_master
WHERE year = 2021 AND age <= 39 AND reason_code IS NOT NULL
GROUP BY reason_code
ORDER BY COUNT(*) DESC
LIMIT 1
"""
    top_reason = load_table(query_kpi3).iloc[0, 0]

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f"""<div class="kpi-box red">
            <div class="kpi-label">청년 금융부채 증가율 (2018 → 2021)</div>
            <div class="kpi-value">{growth_rate:.1f}%</div>
            <div class="kpi-sub">코로나 초저금리 국면 집중 분석 구간</div>
            </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(
            f"""<div class="kpi-box">
            <div class="kpi-label">최근 청년(30대) 평균 대출액 (2026 Q1)</div>
            <div class="kpi-value">{real_loan_str}</div>
            <div class="kpi-sub">고금리 속 고점 유지 — 부채의 하방경직성</div>
            </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(
            f"""<div class="kpi-box orange">
            <div class="kpi-label">2021년 청년 부채 증가 주원인</div>
            <div class="kpi-value" style="font-size:1.2rem;">{top_reason}</div>
            <div class="kpi-sub">투기보다 생존형 대응이 다수</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 차트 1 : 기준금리 × 청년 대출 ──────────────────────────
    st.markdown('<p class="section-title">② 거시 분석 — 기준금리 변동과 청년층 대출 잔액 추이</p>', unsafe_allow_html=True)

    query_main = """
SELECT
    main_data.year_label,
    AVG(main_data.year_avg_loan)  AS avg_loan,
    MAX(main_data.year_max_rate)  AS avg_base_rate
FROM (
    SELECT
        SUBSTR(l.year_quarter, 1, 4) AS year_label,
        AVG(l.loan_balance)          AS year_avg_loan,
        AVG(b.base_rate)             AS year_max_rate
    FROM table_youth_loan l
    LEFT JOIN table_base_rate b ON l.year_quarter = b.year_quarter
    WHERE l.age_group IN ('20대', '30대')
    GROUP BY l.year_quarter
) main_data
GROUP BY main_data.year_label
ORDER BY main_data.year_label ASC
"""
    df_main = load_table(query_main)

    chart1, info1 = st.columns([2, 1])
    with chart1:
        fig_m = make_subplots(specs=[[{"secondary_y": True}]])
        fig_m.add_trace(
            go.Bar(x=df_main['year_label'] + "년", y=df_main['avg_loan'],
                   name="청년층 평균 대출 잔액 (십만원)",
                   marker_color=COLOR_BLUE, opacity=0.85),
            secondary_y=False)
        fig_m.add_trace(
            go.Scatter(x=df_main['year_label'] + "년", y=df_main['avg_base_rate'],
                       name="한국은행 연평균 기준금리 (%)",
                       line=dict(color=COLOR_RED, width=3)),
            secondary_y=True)
        fig_m.update_layout(
            title_text="연도별 청년 부채 총량 변화와 거시 금리 흐름",
            hovermode="x unified", plot_bgcolor="white",
            legend=dict(orientation="h", y=1.05, x=0))
        fig_m.update_xaxes(showgrid=False)
        fig_m.update_yaxes(title_text="평균 대출 잔액 (십만원)", range=[450, 780],
                           gridcolor="#E8EFF6", secondary_y=False)
        fig_m.update_yaxes(title_text="기준금리 (%)", showgrid=False, secondary_y=True)
        st.plotly_chart(fig_m, use_container_width=True)

    with info1:
        st.markdown(
            '<div class="insight-box">💡 <b>금리 하락과의 강력한 동조성</b><br><br>'
            '기준금리가 <b>2019년 1.56% → 2021년 0.64%로 급락</b>하는 초저금리 국면에서 '
            '청년층(20·30대) 평균 대출 잔액이 가파르게 급증했습니다.<br><br>'
            '💡 <b>고금리에도 부채가 줄지 않는다 (하방경직성)</b><br><br>'
            '2023년 이후 기준금리가 <b>3.50%까지 폭등</b>했음에도 '
            '청년 대출 잔액은 오히려 <b>2026년 Q1 기준 30대 평균 약 11.3억원</b> 수준을 유지합니다. '
            '이자 부담 리스크가 세대 내에 고착화된 것입니다.</div>',
            unsafe_allow_html=True)
        with st.expander("🗄️ SQL 쿼리 보기"):
            st.code(query_main, language="sql")

    st.markdown("---")

    # ── 차트 2 : 사유별 부채 총량 ───────────────────────────────
    st.markdown('<p class="section-title">③ 미시 분석 — 청년 부채 증가의 진짜 원인: 투기인가, 생존인가?</p>', unsafe_allow_html=True)

    query_sub = """
SELECT
    year,
    SUM(CASE WHEN reason_code = 1.0 THEN financial_debt ELSE 0 END) / 100 AS home_purchase_total,
    SUM(CASE WHEN reason_code = 4.0 THEN financial_debt ELSE 0 END) / 100 AS asset_investment_total,
    SUM(CASE WHEN reason_code = 9.0 THEN financial_debt ELSE 0 END) / 100 AS lifestyle_etc_total
FROM table_household_master
WHERE age <= 39
GROUP BY year
"""
    df_sub = load_table(query_sub)

    chart2, info2 = st.columns([2, 1])
    with chart2:
        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(
            x=df_sub['year'].astype(str) + "년", y=df_sub['home_purchase_total'],
            name="거주주택 구입 (영끌)", marker_color=COLOR_BLUE))
        fig_b.add_trace(go.Bar(
            x=df_sub['year'].astype(str) + "년", y=df_sub['asset_investment_total'],
            name="주식/자산투자·사업자금", marker_color=COLOR_GREEN))
        fig_b.add_trace(go.Bar(
            x=df_sub['year'].astype(str) + "년", y=df_sub['lifestyle_etc_total'],
            name="생활비·기타", marker_color=COLOR_ORANGE))
        fig_b.update_layout(
            title="연도별 부채 사유별 총량 비교 (단위: 백만원)",
            barmode='group', plot_bgcolor="white",
            legend=dict(orientation="h", y=1.05, x=0),
            margin=dict(t=100, b=40, l=40, r=40))
        fig_b.update_xaxes(showgrid=False)
        fig_b.update_yaxes(gridcolor="#E8EFF6")
        st.plotly_chart(fig_b, use_container_width=True)

    with info2:
        st.markdown(
            '<div class="insight-box">💡 <b>언론 프레임의 왜곡 검증</b><br><br>'
            '2021년 청년 부채를 사유별로 비교하면, <b>주택구입 총액(약 8,074억원)이 주식·자산투자(약 4,166억원)의 약 2배</b>입니다. '
            '청년 부채 폭발의 본질은 투기가 아닌 <b>폭등하는 주거비에 대한 생존형 대응</b>이었습니다.<br><br>'
            '💡 <b>2023년의 위험 신호</b><br><br>'
            '투자 목적 부채는 줄었지만, <b>생활비형 부채가 오히려 증가</b>합니다. '
            '취약 청년층의 위기가 시작됩니다.</div>',
            unsafe_allow_html=True)
        with st.expander("🗄️ SQL 쿼리 보기"):
            st.code(query_sub, language="sql")

    # ── 결론 ────────────────────────────────────────────────────
    st.markdown(
        '<div class="conclusion-box">📢 <b>대시보드 1 종합 결론</b> — '
        '청년 부채는 저금리와 주거비 폭등이 맞물린 구조적 결과입니다. '
        '2023년 들어 투자 목적 부채는 감소했으나 생활비형 부채가 급증하며 취약 청년층의 위기 징후가 포착됩니다. '
        '<b>이 부채가 세대 내 소득 격차에 따라 어떤 다른 결과를 낳았는지 → 대시보드 2에서 확인하세요.</b></div>',
        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  대시보드 2 — 부채는 늘었는데, 자산은 쌓였는가?
# ═══════════════════════════════════════════════════════════════
elif page == 2:

    st.markdown('<p class="page-title">💰 대시보드 2: 부채는 늘었는데, 자산은 쌓였는가?</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">청년 순자산 평균은 올랐지만 중앙값은 내려갔다 — 숫자 뒤에 숨은 양극화를 추적합니다</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # hfws_youth → household 역할 수행 (조사연도 사용)
    # 컬럼 매핑: 조사연도→조사연도, 소득분위_숫자→소득5분위코드 등
    @st.cache_data
    def load_youth():
        return pd.read_sql(
            """SELECT 조사연도, 가중값, 가구주_만연령,
                      소득분위_숫자 AS 소득5분위코드,
                      순자산, 부채_금융부채,
                      자산_금융자산, 자산_실물자산
               FROM hfws_youth""",
            conn,
        )

    youth_df = load_youth()

    # ── 차트 1 : 순자산 평균 vs 중앙값 ─────────────────────────
    st.markdown('<p class="section-title">① 청년 순자산 평균은 올랐는데, 중앙값은 왜 내려갔을까?</p>', unsafe_allow_html=True)

    query_avg = """
SELECT
    조사연도,
    SUM(부채_금융부채 * 가중값) / SUM(가중값) AS 금융부채_가중평균,
    SUM(순자산 * 가중값) / SUM(가중값) AS 순자산_가중평균
FROM hfws_youth
GROUP BY 조사연도
ORDER BY 조사연도 ASC
"""
    avg_df = load_table(query_avg)

    medians = []
    for yr in [2018, 2021, 2023]:
        sub = youth_df[youth_df["조사연도"] == yr]
        med = weighted_median(sub, "순자산", "가중값")
        medians.append({"조사연도": yr, "순자산_중앙값": med})
    med_df = pd.DataFrame(medians)
    chart1_df = pd.merge(avg_df, med_df, on="조사연도")

    # 변화율 계산
    r18 = chart1_df[chart1_df["조사연도"] == 2018].iloc[0]
    r23 = chart1_df[chart1_df["조사연도"] == 2023].iloc[0]
    rate_debt    = ((r23["금융부채_가중평균"] - r18["금융부채_가중평균"]) / r18["금융부채_가중평균"]) * 100
    rate_avg     = ((r23["순자산_가중평균"]   - r18["순자산_가중평균"])   / r18["순자산_가중평균"])   * 100
    rate_med     = ((r23["순자산_중앙값"]     - r18["순자산_중앙값"])     / abs(r18["순자산_중앙값"])) * 100

    c1, c2 = st.columns([2.5, 1])
    with c1:
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        # 금융부채 (왼쪽)
        fig1.add_trace(
            go.Scatter(x=chart1_df["조사연도"], y=chart1_df["금융부채_가중평균"],
                       name="금융부채 가중평균",
                       mode="lines+markers+text",
                       line=dict(color=COLOR_RED, width=3),
                       marker=dict(size=9),
                       text=[f"{int(v):,}만" for v in chart1_df["금융부채_가중평균"]],
                       textposition="top center"),
            secondary_y=False)
        # 순자산 평균 (오른쪽)
        fig1.add_trace(
            go.Scatter(x=chart1_df["조사연도"], y=chart1_df["순자산_가중평균"],
                       name="순자산 가중평균 ▲",
                       mode="lines+markers+text",
                       line=dict(color=COLOR_BLUE, width=3),
                       marker=dict(size=9),
                       text=[f"{int(v):,}만" for v in chart1_df["순자산_가중평균"]],
                       textposition="top center"),
            secondary_y=True)
        # 순자산 중앙값 (오른쪽, 강조 - 빨간 점선)
        fig1.add_trace(
            go.Scatter(x=chart1_df["조사연도"], y=chart1_df["순자산_중앙값"],
                       name="순자산 중앙값 ▼ (하락)",
                       mode="lines+markers+text",
                       line=dict(color=COLOR_RED, width=3, dash="dash"),
                       marker=dict(size=9, symbol="diamond"),
                       text=[f"{int(v):,}만" for v in chart1_df["순자산_중앙값"]],
                       textposition="bottom center"),
            secondary_y=True)
        fig1.update_layout(
            title_text="청년 금융부채·순자산 평균 및 중앙값 추이",
            hovermode="x unified", plot_bgcolor="white",
            legend=dict(orientation="h", y=1.08, x=0),
            xaxis=dict(tickvals=[2018, 2021, 2023], gridcolor="#E8EFF6"),
            margin=dict(t=80, b=40, l=60, r=60))
        fig1.update_yaxes(title_text="금융부채 가중평균 (만원)",
                          gridcolor="#E8EFF6", secondary_y=False,
                          titlefont=dict(color=COLOR_RED))
        fig1.update_yaxes(title_text="순자산 (만원)", showgrid=False,
                          secondary_y=True, titlefont=dict(color=COLOR_BLUE))
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.markdown("#### 📊 2018 → 2023 변화율")
        st.markdown(
            f"""<div class="kpi-box red">
            <div class="kpi-label">금융부채 평균 변화율</div>
            <div class="kpi-value">{rate_debt:+.1f}%</div></div>""",
            unsafe_allow_html=True)
        st.markdown(
            f"""<div class="kpi-box">
            <div class="kpi-label">순자산 평균 변화율</div>
            <div class="kpi-value" style="color:#3B82F6">{rate_avg:+.1f}%</div></div>""",
            unsafe_allow_html=True)
        st.markdown(
            f"""<div class="kpi-box red">
            <div class="kpi-label">순자산 중앙값 변화율</div>
            <div class="kpi-value">{rate_med:+.1f}%</div>
            <div class="kpi-sub">⚠️ 평균은 올라도 중앙값은 하락</div></div>""",
            unsafe_allow_html=True)

    insight_q = query_avg
    st.markdown(
        '<div class="insight-box">💡 <b>평균과 중앙값의 역행 — 양극화의 증거</b><br><br>'
        '청년 순자산 <b>평균은 2018년 2.1천만원 → 2021년 2.6천만원으로 상승</b>했습니다. '
        '그러나 <b>중앙값은 2021년 1.55천만원에서 2023년 1.36천만원으로 다시 하락</b>했습니다. '
        '이는 소수 고소득 청년의 자산 급등이 평균을 끌어올린 것일 뿐, '
        '<b>대다수 청년에게는 부채 증가와 자산 증가가 동시에 일어나지 않았음</b>을 의미합니다. '
        '빚은 전체가 함께 늘었지만, 그 결과는 균등하지 않았습니다.</div>',
        unsafe_allow_html=True)
    with st.expander("🗄️ SQL 쿼리 보기"):
        st.code(query_avg, language="sql")

    st.markdown("---")

    # ── 차트 2 : 소득분위별 순자산 추이 ─────────────────────────
    st.markdown('<p class="section-title">② 소득분위별 순자산 격차 — 영끌 시기 기울기의 차이</p>', unsafe_allow_html=True)

    query_q = """
SELECT
    조사연도,
    소득분위_숫자,
    SUM(순자산 * 가중값) / SUM(가중값) AS 순자산_가중평균,
    SUM(부채_금융부채 * 가중값) / SUM(가중값) AS 금융부채_가중평균
FROM hfws_youth
GROUP BY 조사연도, 소득분위_숫자
ORDER BY 조사연도 ASC, 소득분위_숫자 ASC
"""
    qt_df = load_table(query_q)

    c2a, c2b = st.columns([2.5, 1])
    with c2a:
        fig2 = go.Figure()
        for qi in [1, 2, 3, 4, 5]:
            sub = qt_df[qt_df["소득분위_숫자"] == qi]
            is_focus = qi in [1, 5]
            width  = 4 if is_focus else 1.5
            opacity = 1.0 if is_focus else 0.45
            color = Q_COLORS.get(qi, COLOR_GREY)
            dash = "solid"
            label = f"{'★ ' if is_focus else ''}{qi}분위{'(최상위)' if qi==5 else '(최하위)' if qi==1 else ''}"
            fig2.add_trace(go.Scatter(
                x=sub["조사연도"], y=sub["순자산_가중평균"],
                name=label,
                mode="lines+markers",
                line=dict(color=color, width=width, dash=dash),
                marker=dict(size=8 if is_focus else 5, opacity=opacity),
                opacity=opacity,
            ))

        # 2018→2021 기울기 강조 화살표 (5분위)
        sub5 = qt_df[qt_df["소득분위_숫자"] == 5]
        sub1 = qt_df[qt_df["소득분위_숫자"] == 1]
        try:
            y5_18 = sub5[sub5["조사연도"]==2018]["순자산_가중평균"].values[0]
            y5_21 = sub5[sub5["조사연도"]==2021]["순자산_가중평균"].values[0]
            y1_18 = sub1[sub1["조사연도"]==2018]["순자산_가중평균"].values[0]
            y1_21 = sub1[sub1["조사연도"]==2021]["순자산_가중평균"].values[0]

            fig2.add_annotation(
                x=2021, y=y5_21,
                ax=2018, ay=y5_18,
                xref="x", yref="y", axref="x", ayref="y",
                text=f"+{int(y5_21-y5_18):,}만원",
                showarrow=True,
                arrowhead=2, arrowcolor=COLOR_Q5, arrowwidth=2,
                font=dict(color=COLOR_Q5, size=12, family="Arial Black"),
                bgcolor="white", bordercolor=COLOR_Q5, borderwidth=1,
                borderpad=3,
            )
            fig2.add_annotation(
                x=2021, y=y1_21,
                ax=2018, ay=y1_18,
                xref="x", yref="y", axref="x", ayref="y",
                text=f"{int(y1_21-y1_18):,}만원",
                showarrow=True,
                arrowhead=2, arrowcolor=COLOR_Q1, arrowwidth=2,
                font=dict(color=COLOR_Q1, size=12, family="Arial Black"),
                bgcolor="white", bordercolor=COLOR_Q1, borderwidth=1,
                borderpad=3,
            )
        except Exception:
            pass

        fig2.update_layout(
            title="소득분위별 청년 순자산 추이 (2018·2021·2023)",
            hovermode="x unified", plot_bgcolor="white",
            legend=dict(orientation="v", x=1.01, y=1),
            xaxis=dict(tickvals=[2018, 2021, 2023], gridcolor="#E8EFF6"),
            yaxis=dict(title="순자산 가중평균 (만원)", gridcolor="#E8EFF6"),
            margin=dict(t=60, b=40, l=60, r=120))
        st.plotly_chart(fig2, use_container_width=True)

    with c2b:
        # 분위별 연도간 변화율 표
        rows = []
        for qi in [1, 2, 3, 4, 5]:
            sub = qt_df[qt_df["소득분위_숫자"] == qi]
            vals = {int(r["조사연도"]): r["순자산_가중평균"] for _, r in sub.iterrows()}
            def chg(a, b):
                if vals.get(a, 0) == 0:
                    return "N/A"
                return f"{((vals.get(b,0)-vals.get(a,0))/abs(vals.get(a,0)))*100:+.1f}%"
            rows.append({
                "분위": f"{'★' if qi in [1,5] else ''}{qi}분위",
                "18→21": chg(2018, 2021),
                "21→23": chg(2021, 2023),
                "18→23": chg(2018, 2023),
            })
        tbl = pd.DataFrame(rows)
        st.markdown("#### 📋 분위별 순자산 변화율")
        st.dataframe(tbl, hide_index=True, use_container_width=True)

    st.markdown(
        '<div class="insight-box">💡 <b>영끌 시기(2018→2021): 정반대의 결과</b><br><br>'
        f'<b style="color:{COLOR_Q5}">5분위(최상위) 청년</b>은 이 구간에서 순자산이 대폭 증가해 '
        f'기울기가 가파르게 치솟습니다. 반면 '
        f'<b style="color:{COLOR_Q1}">1분위(최하위) 청년</b>은 오히려 순자산이 감소했습니다. '
        '같은 시기, 같은 저금리 환경에서 빚을 냈지만 그 결과는 정반대였습니다.<br><br>'
        '이는 고소득 청년이 부채를 레버리지로 활용해 자산을 증식한 반면, '
        '저소득 청년의 부채는 자산 형성이 아닌 생계 유지에 쓰였음을 시사합니다.</div>',
        unsafe_allow_html=True)
    with st.expander("🗄️ SQL 쿼리 보기"):
        st.code(query_q, language="sql")

    st.markdown("---")

    # ── 차트 2-1 : 소득분위별 순자산 박스플롯 (3번→이동) ────────
    st.markdown('<p class="section-title">② -1 소득분위 × 연도별 순자산 분포 (박스플롯)</p>', unsafe_allow_html=True)

    # hfws_youth 로드
    hfws_q = load_table(
        "SELECT year, 소득분위_숫자, 순자산, 부채_금융부채, 고위험_플래그 FROM hfws_youth"
    )
    q99 = hfws_q["순자산"].quantile(0.99)
    hfws_clip = hfws_q[hfws_q["순자산"] <= q99].copy()
    hfws_clip["소득분위"] = hfws_clip["소득분위_숫자"].map(
        {1: "Q1(하위)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5(상위)"}
    )

    selected_q = st.multiselect(
        "소득분위 선택 (기본: 전체)",
        options=["Q1(하위)", "Q2", "Q3", "Q4", "Q5(상위)"],
        default=["Q1(하위)", "Q2", "Q3", "Q4", "Q5(상위)"],
        key="bp_select"
    )
    if not selected_q:
        selected_q = ["Q1(하위)", "Q2", "Q3", "Q4", "Q5(상위)"]

    hfws_f = hfws_clip[hfws_clip["소득분위"].isin(selected_q)]

    fig_bp = go.Figure()
    year_label_map = {2018: "2018년 (코로나 이전)", 2021: "2021년 (영끌 시기)", 2023: "2023년 (회복기)"}
    for yr in [2018, 2021, 2023]:
        sub = hfws_f[hfws_f["year"] == yr]
        fig_bp.add_trace(go.Box(
            x=sub["소득분위"], y=sub["순자산"],
            name=year_label_map[yr],
            marker_color=YEAR_COLORS[yr],
            boxmean=True,
            offsetgroup=str(yr),
            hovertemplate=f"<b>{yr}년</b><br>소득분위: %{{x}}<br>순자산: %{{y:,.0f}}만원<extra></extra>",
        ))

    fig_bp.update_layout(
        height=480,
        title="소득분위 × 연도별 순자산 분포 (상위 1% 이상치 제거)",
        plot_bgcolor="white",
        xaxis=dict(title="소득분위"),
        yaxis=dict(title="순자산 (만원)", gridcolor="#E8EFF6"),
        boxmode="group",
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=70, b=40, l=60, r=40))
    st.plotly_chart(fig_bp, use_container_width=True)

    bp_query = """
SELECT year, 소득분위_숫자, 순자산
FROM hfws_youth
WHERE 순자산 <= (SELECT PERCENTILE_APPROX(순자산, 0.99) FROM hfws_youth)
-- (Python에서 .quantile(0.99) 로 이상치 제거 후 박스플롯 구성)
"""
    st.markdown(
        '<div class="insight-box">💡 <b>분포 전체가 말해주는 구조적 불균형</b><br><br>'
        'Q5(상위)는 2021년 박스 전체가 위로 크게 이동한 반면, '
        'Q1(하위)은 3개년 내내 같은 자리에 머물러 있습니다. '
        '<b>격차는 수치 차이를 넘어 분포 자체의 구조적 불균형</b>입니다.</div>',
        unsafe_allow_html=True)
    with st.expander("🗄️ SQL 쿼리 보기"):
        st.code("SELECT year, 소득분위_숫자, 순자산 FROM hfws_youth\n-- Python에서 .quantile(0.99)로 이상치 제거 후 박스플롯 시각화", language="sql")

    st.markdown("---")

    # ── 차트 3 : 금융·실물자산 비중 (원래 대시보드2 차트2) ───────
    st.markdown('<p class="section-title">③ 청년 자산 포트폴리오 구성 변화</p>', unsafe_allow_html=True)

    query_asset = """
SELECT
    조사연도,
    SUM(자산_금융자산 * 가중값) AS 금융자산_합,
    SUM(자산_실물자산 * 가중값) AS 실물자산_합
FROM hfws_youth
GROUP BY 조사연도
"""
    asset_df = load_table(query_asset)
    asset_df["총자산_합"] = asset_df["금융자산_합"] + asset_df["실물자산_합"]
    asset_df["금융자산 비중"] = (asset_df["금융자산_합"] / asset_df["총자산_합"]) * 100
    asset_df["실물자산 비중"] = (asset_df["실물자산_합"] / asset_df["총자산_합"]) * 100

    melted = pd.melt(asset_df, id_vars=["조사연도"],
                     value_vars=["금융자산 비중", "실물자산 비중"],
                     var_name="자산종류", value_name="비중")

    fig3 = px.bar(melted, x="조사연도", y="비중", color="자산종류",
                  title="연도별 자산 포트폴리오 비중 변화 (100% 누적 막대)",
                  labels={"비중": "비중 (%)", "조사연도": "조사연도"},
                  color_discrete_map={"금융자산 비중": "#60A5FA", "실물자산 비중": "#93C5FD"})
    fig3.update_layout(xaxis=dict(tickvals=[2018, 2021, 2023]),
                       yaxis=dict(ticksuffix="%"),
                       plot_bgcolor="white",
                       margin=dict(t=60, b=40))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        '<div class="insight-box">💡 <b>실물자산 집중 → 금리 충격에 취약</b><br><br>'
        '청년 가구 자산의 대부분이 부동산 등 실물자산에 집중되어 있습니다. '
        '부동산 폭등기에 실물자산 비중이 급증했으나, '
        '금리 인상·전세사기 이후 금융자산 비중이 소폭 확대되는 모습입니다. '
        '실물 위주의 포트폴리오는 금리 상승 시 청년 신용위험으로 직결됩니다.</div>',
        unsafe_allow_html=True)
    with st.expander("🗄️ SQL 쿼리 보기"):
        st.code(query_asset, language="sql")

    # ── 결론 ────────────────────────────────────────────────────
    st.markdown(
        '<div class="conclusion-box">📢 <b>대시보드 2 종합 결론</b> — '
        '청년 순자산 평균은 상승했지만 중앙값은 하락했습니다. '
        '소득 5분위는 영끌 시기 순자산이 크게 증가한 반면, 1분위는 오히려 감소했습니다. '
        '<b>이 격차가 통계적으로 유의한지, 어떤 요인이 설명하는지 → 대시보드 3에서 검증합니다.</b></div>',
        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  대시보드 3 — 같은 부채, 소득분위별 다른 결과
# ═══════════════════════════════════════════════════════════════
elif page == 3:

    st.markdown('<p class="page-title">🔬 대시보드 3: 같은 부채, 소득분위별 다른 결과</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">격차의 통계적 검증 — t-test · 회귀분석 · 주관적 재정인식까지</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

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
    def load_hfws():
        return pd.read_sql(
            "SELECT year, 소득분위_숫자, 순자산, 고위험_플래그, 부채_금융부채 FROM hfws_youth",
            conn)
    @st.cache_data
    def load_median_kgss():
        hf   = pd.read_sql("SELECT year, 순자산 FROM hfws_youth", conn)
        kgss = pd.read_sql("SELECT * FROM kgss_summary", conn)
        med  = (hf.groupby("year")["순자산"].median()
                .reset_index().rename(columns={"순자산": "순자산_중앙값"}))
        return med.merge(kgss, on="year", how="left")

    ttest  = load_ttest()
    ols    = load_ols()
    merged = load_median_kgss()

    # ── KPI 카드 ────────────────────────────────────────────────
    st.markdown('<p class="section-title">① 핵심 수치 요약 — 영끌 시기의 승자와 패자</p>', unsafe_allow_html=True)

    rh = ttest[(ttest["group"]=="상위(Q4+Q5)") & (ttest["year_from"]==2018) & (ttest["year_to"]==2021)].iloc[0]
    rl = ttest[(ttest["group"]=="하위(Q1+Q2)") & (ttest["year_from"]==2018) & (ttest["year_to"]==2021)].iloc[0]
    coef_age_low  = ols[(ols["group"]=="하위(Q1+Q2)") & (ols["var"]=="연령")].iloc[0]["coef"]
    coef_age_high = ols[(ols["group"]=="상위(Q4+Q5)") & (ols["var"]=="연령")].iloc[0]["coef"]

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f"""<div class="kpi-box">
            <div class="kpi-label">영끌 시기 상위 집단 순자산 변화 (2018→2021)</div>
            <div class="kpi-value" style="color:{COLOR_BLUE}">+{int(rh['diff']):,}만원</div>
            <div class="kpi-sub">통계적으로 유의 <span class="sig-badge sig-yes">p&lt;0.001 ★★★</span></div>
            </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(
            f"""<div class="kpi-box red">
            <div class="kpi-label">영끌 시기 하위 집단 순자산 변화 (2018→2021)</div>
            <div class="kpi-value">{int(rl['diff']):,}만원</div>
            <div class="kpi-sub">통계적으로 유의하지 않음 <span class="sig-badge sig-no">n.s.</span></div>
            </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(
            f"""<div class="kpi-box orange">
            <div class="kpi-label">연령 1세 증가 시 순자산 증가 (회귀계수)</div>
            <div class="kpi-value" style="font-size:1.1rem;">하위 +{int(coef_age_low):,}만 vs 상위 +{int(coef_age_high):,}만</div>
            <div class="kpi-sub">나이 들수록 격차는 구조적으로 더 벌어집니다</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="insight-box">💡 같은 영끌 시기, 빚을 낸 건 모두 같았지만 '
        '<b>자산이 늘어난 것은 상위 집단뿐</b>이었습니다(+7,994만원, p&lt;0.001). '
        '하위 집단은 -40만원으로 사실상 제자리입니다. '
        '연령 계수 격차는 나이가 들수록 두 집단의 격차가 구조적으로 확대됨을 의미합니다.</div>',
        unsafe_allow_html=True)

    st.markdown("---")

    # ── 차트 2 : t-test 결과 히트맵 스타일 ──────────────────────
    st.markdown('<p class="section-title">② t-test 검증 — 순자산 격차는 통계적으로 유의한가?</p>', unsafe_allow_html=True)

    # 표 형태로 직관적으로 표현 + 산점도(dot plot) 스타일
    ttest_disp = ttest[["group","year_from","year_to","mean_from","mean_to","diff","t_stat","p_val","sig"]].copy()

    # --- Dot-plot: 비교 구간별 두 집단 순자산 변화 ---
    fig_t = go.Figure()

    periods = [(2018,2021,"영끌 시기"), (2021,2023,"회복 시기"), (2018,2023,"전체 구간")]
    group_map = {"상위(Q4+Q5)": (COLOR_BLUE, "상위(Q4+Q5)"), "하위(Q1+Q2)": (COLOR_RED, "하위(Q1+Q2)")}

    for i, (yf, yt, label) in enumerate(periods):
        sub = ttest[(ttest["year_from"]==yf) & (ttest["year_to"]==yt)]
        for _, row in sub.iterrows():
            col, name = group_map.get(row["group"], (COLOR_GREY, row["group"]))
            sig_mark = " ★" if row["sig"] != "n.s." else ""
            fig_t.add_trace(go.Scatter(
                x=[row["mean_from"], row["mean_to"]],
                y=[f"{label}\n{name}", f"{label}\n{name}"],
                mode="lines+markers+text",
                line=dict(color=col, width=3),
                marker=dict(size=[12, 12], color=[col, col],
                            symbol=["circle", "circle"]),
                text=[f"{int(row['mean_from']):,}만", f"{int(row['mean_to']):,}만{sig_mark}"],
                textposition=["middle left", "middle right"],
                textfont=dict(size=11),
                name=name,
                showlegend=(i == 0),
                hovertemplate=(
                    f"<b>{label} — {name}</b><br>"
                    f"변화량: {int(row['diff']):+,}만원<br>"
                    f"t={row['t_stat']:.3f}, p={row['p_val']:.4f}<br>"
                    f"유의성: {row['sig']}<extra></extra>"
                ),
            ))

    fig_t.update_layout(
        height=500,
        title="소득 집단별 순자산 변화 비교 (점 = 기간 전후 평균, 선 = 변화 방향)",
        plot_bgcolor="white",
        xaxis=dict(title="순자산 평균 (만원)", gridcolor="#E8EFF6"),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=1.06),
        margin=dict(t=80, b=40, l=200, r=60))
    st.plotly_chart(fig_t, use_container_width=True)

    ttest_show = ttest_disp.copy()
    ttest_show.columns = ["집단","비교시작","비교종료","기간전평균","기간후평균","변화량(만원)","t통계량","p값","유의성"]
    ttest_show["변화량(만원)"] = ttest_show["변화량(만원)"].apply(lambda x: f"{int(x):+,}")
    ttest_show["p값"] = ttest_show["p값"].apply(lambda x: f"{x:.4f}")
    st.dataframe(ttest_show, use_container_width=True, hide_index=True)

    ttest_query = "SELECT * FROM ttest_results"
    st.markdown(
        '<div class="insight-box">💡 <b>통계 검증 결과 요약</b><br><br>'
        '영끌 시기(2018→2021) 상위 집단 순자산은 <b>+7,994만원으로 통계적으로 유의하게 증가(p&lt;0.001, ★★★)</b>한 반면, '
        '하위 집단의 변화량(-40만원)은 p=0.9475로 <b>통계적으로 무의미</b>합니다. '
        '2021→2023 회복 시기에는 양 집단 모두 유의한 변화가 없어, '
        '영끌 시기에 벌어진 격차가 고착화되었음을 시사합니다.</div>',
        unsafe_allow_html=True)
    with st.expander("🗄️ SQL 쿼리 보기"):
        st.code(ttest_query, language="sql")

    st.markdown("---")

    # ── 차트 5 : 회귀분석 — 산점도 + 오차막대 ──────────────────
    st.markdown('<p class="section-title">③ 회귀분석 — 순자산 형성에 영향을 미치는 요인은?</p>', unsafe_allow_html=True)

    ols_data = load_ols()
    ols_data["var_label"] = ols_data["var"].map(
        {"금융부채_만": "금융부채(만원)", "연령": "연령(세)", "수도권": "수도권 거주"}
    )
    ols_data["유의"] = ols_data["sig"].apply(lambda x: x != "n.s.")

    # 오차막대 포함 점-선 차트 (두 집단 비교)
    fig_ols = go.Figure()

    for group, color in [("하위(Q1+Q2)", COLOR_RED), ("상위(Q4+Q5)", COLOR_BLUE)]:
        sub = ols_data[ols_data["group"] == group]
        # 유의한 것만 불투명, 비유의는 회색 점선
        for _, row in sub.iterrows():
            is_sig = row["유의"]
            fig_ols.add_trace(go.Scatter(
                x=[row["var_label"]],
                y=[row["coef"]],
                error_y=dict(
                    type="data",
                    array=[row["se"] * 1.96],
                    color=color if is_sig else COLOR_GREY,
                    thickness=2, width=8),
                mode="markers+text",
                marker=dict(
                    size=16,
                    color=color if is_sig else COLOR_GREY,
                    symbol="circle",
                    line=dict(width=2, color="white")),
                text=[f"{row['coef']:,.0f}<br>{row['sig']}"],
                textposition="top center",
                textfont=dict(size=10, color=color if is_sig else COLOR_GREY),
                name=group,
                showlegend=(row["var_label"] == "연령(세)"),
                hovertemplate=(
                    f"<b>{group} — {row['var_label']}</b><br>"
                    f"회귀계수: {row['coef']:,.1f}만원<br>"
                    f"표준오차: {row['se']:,.1f}<br>"
                    f"유의성: {row['sig']}<extra></extra>"
                ),
                legendgroup=group,
            ))

    fig_ols.add_hline(y=0, line_color=COLOR_GREY, line_width=1, line_dash="dash")

    r2_low  = ols_data[ols_data["group"]=="하위(Q1+Q2)"]["r2"].iloc[0]
    r2_high = ols_data[ols_data["group"]=="상위(Q4+Q5)"]["r2"].iloc[0]

    fig_ols.update_layout(
        height=480,
        title=f"집단별 순자산 회귀계수 (95% 신뢰구간 포함) | 하위 R²={r2_low:.4f} / 상위 R²={r2_high:.4f}",
        plot_bgcolor="white",
        xaxis=dict(title="설명변수", showgrid=False),
        yaxis=dict(title="회귀계수 (만원)", gridcolor="#E8EFF6"),
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=80, b=40, l=60, r=40))
    st.plotly_chart(fig_ols, use_container_width=True)

    # 요약 표
    ols_show = ols_data[["group","var_label","coef","se","t_val","p_val","sig"]].copy()
    ols_show.columns = ["집단","변수","회귀계수(만원)","표준오차","t값","p값","유의성"]
    ols_show["회귀계수(만원)"] = ols_show["회귀계수(만원)"].apply(lambda x: f"{x:,.1f}")
    ols_show["p값"] = ols_show["p값"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    st.dataframe(ols_show, use_container_width=True, hide_index=True)
    st.caption(f"📋 하위 집단 R² = {r2_low:.4f} | 상위 집단 R² = {r2_high:.4f}")

    ols_query = "SELECT * FROM ols_results"
    st.markdown(
        '<div class="insight-box">💡 <b>세 가지 격차 확인</b><br><br>'
        f'① <b>연령 효과</b>: 나이가 1살 늘 때 하위 집단은 순자산이 +535만원 증가하지만, 상위 집단은 +2,003만원 증가합니다. '
        '<b>나이가 들수록 격차는 구조적으로 벌어집니다.</b><br><br>'
        f'② <b>수도권 프리미엄</b>: 수도권 거주 효과가 상위 집단에서만 통계적으로 유의(+5,060만원, ★★★)합니다. '
        '하위 집단에게 수도권 거주는 순자산 형성에 유의미한 도움이 되지 않습니다.<br><br>'
        f'③ <b>금융부채 효과</b>: 두 집단 모두 부채와 순자산이 정(+)의 관계이나, '
        '하위 집단의 계수(0.56)가 상위 집단(0.24)보다 높습니다. '
        '하위 집단은 부채가 늘어날수록 순자산 역시 증가하는 구조이나, '
        '이는 부채로 버티는 생계형 구조일 가능성이 높습니다.</div>',
        unsafe_allow_html=True)
    with st.expander("🗄️ SQL 쿼리 보기"):
        st.code(ols_query, language="sql")

    st.markdown("---")

    # ── 차트 6 : 순자산 중앙값 × 가계만족도 ─────────────────────
    st.markdown('<p class="section-title">④ 순자산 중앙값 vs 가계만족도 (KGSS SATFIN)</p>', unsafe_allow_html=True)

    satfin_query = """
SELECT h.year, MEDIAN(h.순자산) AS 순자산_중앙값, k.avg_satfin
FROM hfws_youth h
LEFT JOIN kgss_summary k ON h.year = k.year
GROUP BY h.year
-- (Python에서 median() 계산 후 LEFT JOIN)
"""

    fig6 = make_subplots(specs=[[{"secondary_y": True}]])
    fig6.add_trace(
        go.Scatter(x=merged["year"], y=merged["순자산_중앙값"],
                   name="순자산 중앙값 (만원)", mode="lines+markers+text",
                   line=dict(color=COLOR_BLUE, width=3),
                   marker=dict(size=9),
                   text=[f"{int(v):,}만" for v in merged["순자산_중앙값"]],
                   textposition="top center",
                   hovertemplate="<b>%{x}년</b><br>순자산 중앙값: %{y:,.0f}만원<extra></extra>"),
        secondary_y=False)
    fig6.add_trace(
        go.Scatter(x=merged["year"], y=merged["avg_satfin"],
                   name="가계만족도 평균 (↑ 높을수록 불만족)",
                   mode="lines+markers+text",
                   line=dict(color=COLOR_ORANGE, width=3, dash="dot"),
                   marker=dict(size=9),
                   text=[f"{v:.3f}" for v in merged["avg_satfin"]],
                   textposition="bottom center",
                   hovertemplate="<b>%{x}년</b><br>가계만족도: %{y:.3f}<extra></extra>"),
        secondary_y=True)
    fig6.update_layout(
        height=400, plot_bgcolor="white",
        title="순자산 중앙값과 가계만족도 추이 (2018·2021·2023)",
        legend=dict(orientation="h", y=1.1),
        xaxis=dict(tickvals=[2018, 2021, 2023], gridcolor="#E8EFF6"),
        margin=dict(t=70, b=40, l=60, r=80))
    fig6.update_yaxes(title_text="순자산 중앙값 (만원)", gridcolor="#E8EFF6", secondary_y=False)
    fig6.update_yaxes(title_text="SATFIN 평균 (↑높을수록 불만족)",
                      showgrid=False, range=[2.5, 3.5], secondary_y=True)
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown(
        '<div class="insight-box">💡 <b>객관적 자산과 주관적 체감이 모두 악화</b><br><br>'
        '순자산 중앙값이 <b>13,660만원(2018) → 15,513만원(2021) → 13,560만원(2023)</b>으로 '
        '영끌 시기 반짝 상승 후 다시 하락했습니다. '
        '가계만족도 지수도 2.87(2018) → 3.05(2021)로 악화되었습니다. '
        '자산 통계상 평균은 올랐지만, <b>실제 청년이 느끼는 체감은 계속 나빠지고 있습니다.</b></div>',
        unsafe_allow_html=True)
    with st.expander("🗄️ SQL 쿼리 보기"):
        st.code("SELECT year, AVG(avg_satfin) FROM kgss_summary GROUP BY year\n-- Python에서 hfws_youth median과 LEFT JOIN 후 시각화", language="sql")

    st.markdown("---")

    # ── 차트 7 : 순자산 중앙값 × 미래 전망 ─────────────────────
    st.markdown('<p class="section-title">⑤ 순자산 중앙값 vs 미래 경제 전망 (KGSS FINPROS)</p>', unsafe_allow_html=True)

    merged_fin = merged[merged["avg_finpros"].notna()].copy()

    fig7 = make_subplots(specs=[[{"secondary_y": True}]])
    fig7.add_trace(
        go.Scatter(x=merged_fin["year"], y=merged_fin["순자산_중앙값"],
                   name="순자산 중앙값 (만원)", mode="lines+markers+text",
                   line=dict(color=COLOR_BLUE, width=3),
                   marker=dict(size=9),
                   text=[f"{int(v):,}만" for v in merged_fin["순자산_중앙값"]],
                   textposition="top center"),
        secondary_y=False)
    fig7.add_trace(
        go.Scatter(x=merged_fin["year"], y=merged_fin["avg_finpros"],
                   name="미래전망 평균 (↑ 높을수록 비관적)",
                   mode="lines+markers+text",
                   line=dict(color="#9333EA", width=3, dash="dot"),
                   marker=dict(size=9),
                   text=[f"{v:.3f}" for v in merged_fin["avg_finpros"]],
                   textposition="bottom center"),
        secondary_y=True)
    fig7.update_layout(
        height=400, plot_bgcolor="white",
        title="순자산 중앙값과 미래 경제 전망 (2021·2023)",
        legend=dict(orientation="h", y=1.1),
        xaxis=dict(tickvals=[2021, 2023], gridcolor="#E8EFF6"),
        margin=dict(t=70, b=40, l=60, r=80))
    fig7.update_yaxes(title_text="순자산 중앙값 (만원)", gridcolor="#E8EFF6", secondary_y=False)
    fig7.update_yaxes(title_text="FINPROS 평균 (↑높을수록 비관적)",
                      showgrid=False, range=[2.0, 3.5], secondary_y=True)
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown(
        '<div class="insight-box">💡 <b>"현재도 나쁘고, 미래도 밝지 않다"</b><br><br>'
        '2023년 순자산 중앙값 하락과 함께 미래 전망 지수도 개선되지 않았습니다(2.534 → 2.503). '
        '⑥번 차트와 함께, 청년의 재정 상황에 대한 <b>객관적 악화와 주관적 비관이 동시에 진행</b>되고 있습니다.</div>',
        unsafe_allow_html=True)
    with st.expander("🗄️ SQL 쿼리 보기"):
        st.code("SELECT year, avg_finpros FROM kgss_summary\n-- Python에서 hfws_youth median과 LEFT JOIN 후 시각화", language="sql")

    # ── 최종 결론 ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="conclusion-box">📢 <b>최종 결론 — 정책적 시사점</b><br><br>'
        '청년 부채의 증가와 그 결과는 소득 수준에 따라 구조적으로 달랐습니다. '
        '상위 소득 청년에게 부채는 자산 증식의 레버리지였지만, '
        '하위 소득 청년에게는 생계 유지 도구였으며 자산 형성으로 이어지지 않았습니다. '
        '<b>청년 금융 정책은 "청년"이라는 단일 범주에서 벗어나, '
        '소득 분위별 차등적 접근이 필요합니다.</b></div>',
        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#9CA3AF; font-size:0.78rem; margin-top:1rem;'>"
        "데이터 출처: 통계청 가계금융복지조사(2018·2021·2023) × KGSS 한국종합사회조사<br>"
        "청년 = 가구주 만 39세 이하 | 경영정보처리론 8조"
        "</div>",
        unsafe_allow_html=True)
