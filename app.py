import os
import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 1. 페이지 기본 설정
# =========================================================
st.set_page_config(
    page_title="Japan Tourism Demand Forecast",
    page_icon="🗾",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 2. 최소 CSS
# - 제목 잘림 방지
# - 카드 / 버튼 / 입력 UI 정리
# - HTML 레이아웃은 사용하지 않음
# =========================================================
st.markdown(
    """
    <style>

    /* 전체 화면 폭과 상단 여백 */
    .block-container {
        max-width: 1180px;
        padding-top: 3.5rem !important;
        padding-bottom: 4rem !important;
    }

    /* 제목이 위아래로 잘려 보이는 현상 방지 */
    h1 {
        line-height: 1.35 !important;
        padding-top: 0.20rem !important;
        padding-bottom: 0.35rem !important;
        margin-bottom: 0.35rem !important;
    }

    h2 {
        line-height: 1.35 !important;
        padding-top: 0.15rem !important;
        padding-bottom: 0.20rem !important;
    }

    h3 {
        line-height: 1.35 !important;
    }

    /* 전체 배경 */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Streamlit border container를 카드처럼 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white;
        border-radius: 16px;
    }

    /* 버튼 */
    div.stButton > button {
        border-radius: 10px;
        min-height: 48px;
        font-weight: 700;
        font-size: 0.98rem;
    }

    /* selectbox */
    div[data-baseweb="select"] > div {
        border-radius: 10px;
    }

    /* number input */
    div[data-testid="stNumberInput"] input {
        border-radius: 10px;
    }

    /* metric 숫자 */
    div[data-testid="stMetricValue"] {
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1.2;
    }

    /* metric label */
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }

    /* radio 간격 */
    div[data-testid="stRadio"] > div {
        gap: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. 파일 경로
# app.py와 아래 3개 파일이 같은 폴더에 있어야 함
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "random_forest_lag_final.pkl"
)

COLUMNS_FILE = os.path.join(
    BASE_DIR,
    "model_columns_final.pkl"
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "japan_inbound_final_v2.csv"
)


# =========================================================
# 4. 모델 / 데이터 로드
# =========================================================
@st.cache_resource
def load_model():
    # Random Forest 모델 로드
    model = joblib.load(MODEL_FILE)

    # 학습 당시 컬럼 목록 로드
    columns = joblib.load(COLUMNS_FILE)

    return model, columns


@st.cache_data
def load_data():
    # General Mode 자동 조회용 데이터
    data = pd.read_csv(DATA_FILE)

    # 연도와 월 자료형 통일
    data["year"] = pd.to_numeric(
        data["year"],
        errors="coerce"
    )

    data["month"] = pd.to_numeric(
        data["month"],
        errors="coerce"
    )

    return data


# 파일 로드 오류를 화면에서 확인할 수 있도록 처리
try:
    loaded_model, loaded_columns = load_model()
    df = load_data()

except Exception as e:
    st.error(
        "모델 또는 데이터 파일을 불러오지 못했습니다."
    )
    st.exception(e)
    st.stop()


# =========================================================
# 5. 모델에서 실제 지원 지역 추출
# =========================================================
prefecture_list = sorted([
    col.replace("prefecture_", "")
    for col in loaded_columns
    if col.startswith("prefecture_")
])


# =========================================================
# 6. 지역명 다국어 매핑
# 일본어 / 영어 / 한국어
# =========================================================
prefecture_name_map = {
    "北海道": "北海道 / Hokkaido / 홋카이도",
    "青森県": "青森県 / Aomori / 아오모리",
    "岩手県": "岩手県 / Iwate / 이와테",
    "宮城県": "宮城県 / Miyagi / 미야기",
    "秋田県": "秋田県 / Akita / 아키타",
    "山形県": "山形県 / Yamagata / 야마가타",
    "福島県": "福島県 / Fukushima / 후쿠시마",
    "茨城県": "茨城県 / Ibaraki / 이바라키",
    "栃木県": "栃木県 / Tochigi / 도치기",
    "群馬県": "群馬県 / Gunma / 군마",
    "埼玉県": "埼玉県 / Saitama / 사이타마",
    "千葉県": "千葉県 / Chiba / 지바",
    "東京都": "東京都 / Tokyo / 도쿄",
    "神奈川県": "神奈川県 / Kanagawa / 가나가와",
    "新潟県": "新潟県 / Niigata / 니가타",
    "富山県": "富山県 / Toyama / 도야마",
    "石川県": "石川県 / Ishikawa / 이시카와",
    "福井県": "福井県 / Fukui / 후쿠이",
    "山梨県": "山梨県 / Yamanashi / 야마나시",
    "長野県": "長野県 / Nagano / 나가노",
    "岐阜県": "岐阜県 / Gifu / 기후",
    "静岡県": "静岡県 / Shizuoka / 시즈오카",
    "愛知県": "愛知県 / Aichi / 아이치",
    "三重県": "三重県 / Mie / 미에",
    "滋賀県": "滋賀県 / Shiga / 시가",
    "京都府": "京都府 / Kyoto / 교토",
    "大阪府": "大阪府 / Osaka / 오사카",
    "兵庫県": "兵庫県 / Hyogo / 효고",
    "奈良県": "奈良県 / Nara / 나라",
    "和歌山県": "和歌山県 / Wakayama / 와카야마",
    "鳥取県": "鳥取県 / Tottori / 돗토리",
    "島根県": "島根県 / Shimane / 시마네",
    "岡山県": "岡山県 / Okayama / 오카야마",
    "広島県": "広島県 / Hiroshima / 히로시마",
    "山口県": "山口県 / Yamaguchi / 야마구치",
    "徳島県": "徳島県 / Tokushima / 도쿠시마",
    "香川県": "香川県 / Kagawa / 가가와",
    "愛媛県": "愛媛県 / Ehime / 에히메",
    "高知県": "高知県 / Kochi / 고치",
    "福岡県": "福岡県 / Fukuoka / 후쿠오카",
    "佐賀県": "佐賀県 / Saga / 사가",
    "長崎県": "長崎県 / Nagasaki / 나가사키",
    "熊本県": "熊本県 / Kumamoto / 구마모토",
    "大分県": "大分県 / Oita / 오이타",
    "宮崎県": "宮崎県 / Miyazaki / 미야자키",
    "鹿児島県": "鹿児島県 / Kagoshima / 가고시마",
    "沖縄県": "沖縄県 / Okinawa / 오키나와"
}


# 모델에서 실제 지원하는 지역만 출력
prefecture_display_list = [
    prefecture_name_map.get(
        prefecture,
        prefecture
    )
    for prefecture in prefecture_list
]


# =========================================================
# 7. 화면 지역명 → 모델용 일본어 지역명
# =========================================================
def extract_prefecture_jp(display_name):
    # ex:
    # 福岡県 / Fukuoka / 후쿠오카
    # → 福岡県
    return display_name.split(" / ")[0]


# =========================================================
# 8. 실제 머신러닝 예측 함수
# =========================================================
def predict_demand(
    prefecture,
    year,
    month,
    foreign_total_lag1,
    korea_visitors_lag1,
    china_visitors_lag1,
    taiwan_visitors_lag1,
    usa_visitors_lag1,
    hongkong_visitors_lag1,
    thailand_visitors_lag1,
    singapore_visitors_lag1,
    australia_visitors_lag1,
    avg_temp_lag1,
    rainfall_lag1
):
    # 모델 입력용 1행 데이터 생성
    input_df = pd.DataFrame([{
        "year": int(year),
        "month": int(month),

        "foreign_total_lag1":
            float(foreign_total_lag1),

        "korea_visitors_lag1":
            float(korea_visitors_lag1),

        "china_visitors_lag1":
            float(china_visitors_lag1),

        "taiwan_visitors_lag1":
            float(taiwan_visitors_lag1),

        "usa_visitors_lag1":
            float(usa_visitors_lag1),

        "hongkong_visitors_lag1":
            float(hongkong_visitors_lag1),

        "thailand_visitors_lag1":
            float(thailand_visitors_lag1),

        "singapore_visitors_lag1":
            float(singapore_visitors_lag1),

        "australia_visitors_lag1":
            float(australia_visitors_lag1),

        "avg_temp_lag1":
            float(avg_temp_lag1),

        "rainfall_lag1":
            float(rainfall_lag1)
    }])

    # 지역 One-Hot 컬럼
    prefecture_col = f"prefecture_{prefecture}"

    if prefecture_col not in loaded_columns:
        raise ValueError(
            f"현재 모델에서 지원하지 않는 지역입니다: {prefecture}"
        )

    input_df[prefecture_col] = 1

    # 학습 당시 컬럼 순서와 완전히 동일하게 맞춤
    input_df = input_df.reindex(
        columns=loaded_columns,
        fill_value=0
    )

    # 예측
    prediction = loaded_model.predict(
        input_df
    )[0]

    return prediction


# =========================================================
# 9. 상단 Hero 카드
# HTML 없이 Streamlit 기본 카드 사용
# =========================================================
with st.container(border=True):

    st.caption(
        "🤖 MACHINE LEARNING FORECAST DASHBOARD"
    )

    st.title(
        "🗾 일본 외국인 숙박 수요 예측"
    )

    st.markdown(
        "### 日本の外国人宿泊需要予測"
    )

    st.caption(
        "Japan Foreign Visitor Accommodation Demand Forecast"
    )

    st.write("")

    st.write(
        "최근 관광 수요와 방문객 흐름, 기후 정보를 종합하여 "
        "일본 지역별 외국인 숙박 수요를 예측합니다."
    )

    st.caption(
        "最近の観光需要、訪問者動向、気象情報を総合し、"
        "日本の地域別外国人宿泊需要を予測します。"
    )

    st.caption(
        "The model combines recent tourism demand, visitor trends, "
        "and weather information to forecast regional accommodation demand."
    )


st.write("")


# =========================================================
# 10. 사용 모드 카드
# =========================================================
with st.container(border=True):

    st.subheader(
        "사용 모드 / 利用モード / User Mode"
    )

    mode = st.radio(
        "사용 목적에 맞는 모드를 선택하세요.",
        [
            "👤 일반 사용자 / 一般ユーザー / General",
            "📊 관광청·실무자 / 観光実務者 / Professional"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )


st.write("")


# =========================================================
# 11. GENERAL MODE
# =========================================================
if "General" in mode:

    # -----------------------------------------------------
    # General 설명 카드
    # -----------------------------------------------------
    with st.container(border=True):

        st.subheader(
            "🔍 간편 예측 / 簡単予測 / Quick Forecast"
        )

        st.write(
            "지역과 예측 연/월을 선택해 주세요. "
            "최근 관광 수요와 방문객 흐름, 기후 정보를 종합하여 "
            "예측 결과를 보여드립니다."
        )

        st.caption(
            "地域と予測する年・月を選択してください。"
            "最近の観光需要、訪問者の動向、気象情報を総合して"
            "予測結果を表示します。"
        )

        st.caption(
            "Please select the region and forecast year/month. "
            "The app combines recent tourism demand, visitor trends, "
            "and weather information to generate the prediction."
        )


    st.write("")


    # -----------------------------------------------------
    # General 입력 카드
    # -----------------------------------------------------
    with st.container(border=True):

        st.markdown(
            "#### 예측 조건 / 予測条件 / Forecast Settings"
        )

        col1, col2, col3 = st.columns(
            [2.2, 1, 1]
        )

        with col1:
            prefecture_display = st.selectbox(
                "지역 / 地域 / Region",                
                prefecture_display_list
            )

        with col2:
            year = st.selectbox(
                "예측 연도 / 予測年 / Forecast Year",
                [2025]
            )

        with col3:
            month = st.selectbox(
                "예측 월 / 予測月 / Forecast Month",
                list(range(1, 13)),
                index=10
            )

        st.write("")

        predict_clicked = st.button(
            "예측 결과 보기 / 予測結果 / Predict",
            type="primary",
            use_container_width=True
        )


    # -----------------------------------------------------
    # General 예측 실행
    # -----------------------------------------------------
    if predict_clicked:

        prefecture = extract_prefecture_jp(
            prefecture_display
        )

        # 예측 대상 월의 이전 월 계산
        if month == 1:
            prev_year = year - 1
            prev_month = 12

        else:
            prev_year = year
            prev_month = month - 1


        # 내부 데이터에서 자동 조회
        prev_row = df[
            (df["prefecture"] == prefecture) &
            (df["year"] == prev_year) &
            (df["month"] == prev_month)
        ]


        if prev_row.empty:

            st.error(
                "예측에 필요한 기준 데이터가 없습니다. "
                "/ 予測に必要な基準データがありません。 "
                "/ Required reference data is unavailable."
            )

        else:

            row = prev_row.iloc[0]

            try:

                prediction = predict_demand(
                    prefecture,
                    year,
                    month,

                    row["foreign_total"],

                    row["korea_visitors"],
                    row["china_visitors"],
                    row["taiwan_visitors"],
                    row["usa_visitors"],

                    row["hongkong_visitors"],
                    row["thailand_visitors"],
                    row["singapore_visitors"],
                    row["australia_visitors"],

                    row["avg_temp"],
                    row["rainfall"]
                )


                st.write("")

                # -----------------------------------------
                # 결과 카드
                # -----------------------------------------
                with st.container(border=True):

                    st.subheader(
                        "📈 예측 결과 / 予測結果 / Forecast Result"
                    )

                    st.caption(
                        f"{prefecture_display} · "
                        f"{year}년 {month}월"
                    )

                    col1, col2 = st.columns(
                        [2, 1]
                    )

                    with col1:

                        st.metric(
                            label=(
                                "예상 외국인 숙박객 수 / "
                                "予測外国人宿泊者数"
                            ),
                            value=f"{prediction:,.0f}"
                        )

                    with col2:

                        st.metric(
                            label="예측 시점 / Forecast Period",
                            value=f"{year}.{month:02d}"
                        )

                    st.caption(
                        "Predicted Foreign Guest Nights"
                    )


            except Exception as e:

                st.error(
                    "예측 과정에서 오류가 발생했습니다."
                )

                st.exception(e)


    st.write("")


    # -----------------------------------------------------
    # 모델 정보 카드
    # -----------------------------------------------------
    with st.container(border=True):

        st.subheader(
            "ℹ️ 모델 정보 / モデル情報 / Model Information"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "데이터 기간",
                "2023–2025"
            )

            st.caption(
                "Data Period"
            )

        with col2:

            st.metric(
                "예측 모델",
                "Random Forest"
            )

            st.caption(
                "Machine Learning Model"
            )

        with col3:

            st.metric(
                "검증 예측 연도",
                "2025"
            )

            st.caption(
                "Validated Forecast Year"
            )

        st.info(
            "현재 버전은 2023~2025년 관광·숙박·기상 데이터를 "
            "기반으로 구축되었으며, 검증된 예측 대상은 2025년입니다."
            "향후 최신 데이터 자동 수집 기능으로 확장 예정입니다."
        )


# =========================================================
# 12. PROFESSIONAL MODE
# =========================================================
else:

    # -----------------------------------------------------
    # Professional 안내 카드
    # -----------------------------------------------------
    with st.container(border=True):

        st.subheader(
            "📊 실무자 직접 입력 예측 / "
            "実務者向け予測 / "
            "Professional Forecast"
        )

        st.write(
            "최근 관광 수요, 국가별 방문객 흐름, 기후 정보를 "
            "직접 입력하여 숙박 수요를 예측합니다."
        )

        st.caption(
            "観光需要、国別訪問者動向、気象情報を直接入力して"
            "宿泊需要を予測します。"
        )

        st.caption(
            "Enter recent tourism demand, visitor trends by country, "
            "and weather information to generate the forecast."
        )


    st.write("")


    # -----------------------------------------------------
    # ① 기본 정보 카드
    # -----------------------------------------------------
    with st.container(border=True):

        st.subheader(
            "① 기본 정보 / 基本情報 / Basic Information"
        )

        col1, col2, col3 = st.columns(
            [2.2, 1, 1]
        )

        with col1:

            prefecture_display = st.selectbox(
                "지역 / 地域 / Region",
                prefecture_display_list,
                key="pro_prefecture"
            )

        with col2:

            year = st.selectbox(
                "예측 연도 / 予測年 / Forecast Year",
                [2025],
                key="pro_year"
            )

        with col3:

            month = st.selectbox(
                "예측 월 / 予測月 / Forecast Month",
                list(range(1, 13)),
                index=10,
                key="pro_month"
            )


    st.write("")


    # -----------------------------------------------------
    # ② 최근 관광 수요 카드
    # -----------------------------------------------------
    with st.container(border=True):

        st.subheader(
            "② 최근 관광 수요 / "
            "最近の観光需要 / "
            "Recent Tourism Demand"
        )

        foreign_total_lag1 = st.number_input(
            "최근 외국인 연숙박객 수 / "
            "直近の外国人延べ宿泊者数 / "
            "Recent Foreign Guest Nights",
            min_value=0.0,
            value=717500.0,
            step=1000.0
        )


    st.write("")


    # -----------------------------------------------------
    # ③ 국가별 방문객 카드
    # -----------------------------------------------------
    with st.container(border=True):

        st.subheader(
            "③ 국가별 방문객 흐름 / "
            "国別訪問者動向 / "
            "Visitor Trends by Country"
        )

        col1, col2 = st.columns(2)


        with col1:

            korea_visitors_lag1 = st.number_input(
                "한국 / 韓国 / Korea",
                min_value=0.0,
                value=867261.0,
                step=1000.0
            )

            taiwan_visitors_lag1 = st.number_input(
                "대만 / 台湾 / Taiwan",
                min_value=0.0,
                value=538428.0,
                step=1000.0
            )

            hongkong_visitors_lag1 = st.number_input(
                "홍콩 / 香港 / Hong Kong",
                min_value=0.0,
                value=200000.0,
                step=1000.0
            )

            singapore_visitors_lag1 = st.number_input(
                "싱가포르 / シンガポール / Singapore",
                min_value=0.0,
                value=70000.0,
                step=1000.0
            )


        with col2:

            china_visitors_lag1 = st.number_input(
                "중국 / 中国 / China",
                min_value=0.0,
                value=790089.0,
                step=1000.0
            )

            usa_visitors_lag1 = st.number_input(
                "미국 / 米国 / USA",
                min_value=0.0,
                value=311933.0,
                step=1000.0
            )

            thailand_visitors_lag1 = st.number_input(
                "태국 / タイ / Thailand",
                min_value=0.0,
                value=120000.0,
                step=1000.0
            )

            australia_visitors_lag1 = st.number_input(
                "호주 / オーストラリア / Australia",
                min_value=0.0,
                value=96158.0,
                step=1000.0
            )


    st.write("")


    # -----------------------------------------------------
    # ④ 기후 정보 카드
    # -----------------------------------------------------
    with st.container(border=True):

        st.subheader(
            "④ 기후 정보 / 気象情報 / Weather Information"
        )

        col1, col2 = st.columns(2)


        with col1:

            avg_temp_lag1 = st.number_input(
                "최근 평균기온 / "
                "直近の平均気温 / "
                "Recent Average Temperature (℃)",
                value=22.4,
                step=0.1
            )


        with col2:

            rainfall_lag1 = st.number_input(
                "최근 강수량 / "
                "直近の降水量 / "
                "Recent Rainfall (mm)",
                min_value=0.0,
                value=184.0,
                step=1.0
            )


    st.write("")


    # -----------------------------------------------------
    # 예측 버튼 카드
    # -----------------------------------------------------
    with st.container(border=True):

        st.subheader(
            "⑤ 예측 실행 / 予測実行 / Run Forecast"
        )

        predict_clicked = st.button(
            "예측 결과 보기 / 予測する / Predict",
            type="primary",
            use_container_width=True,
            key="professional_predict"
        )


    # -----------------------------------------------------
    # Professional 예측 실행
    # -----------------------------------------------------
    if predict_clicked:

        prefecture = extract_prefecture_jp(
            prefecture_display
        )

        try:

            prediction = predict_demand(
                prefecture,
                year,
                month,

                foreign_total_lag1,

                korea_visitors_lag1,
                china_visitors_lag1,
                taiwan_visitors_lag1,
                usa_visitors_lag1,

                hongkong_visitors_lag1,
                thailand_visitors_lag1,
                singapore_visitors_lag1,
                australia_visitors_lag1,

                avg_temp_lag1,
                rainfall_lag1
            )


            st.write("")

            # ---------------------------------------------
            # Professional 결과 카드
            # ---------------------------------------------
            with st.container(border=True):

                st.subheader(
                    "📈 예측 결과 / 予測結果 / Forecast Result"
                )

                st.caption(
                    f"{prefecture_display} · "
                    f"{year}년 {month}월"
                )

                col1, col2 = st.columns(
                    [2, 1]
                )

                with col1:

                    st.metric(
                        label=(
                            "예상 외국인 연숙박객 수 / "
                            "予測外国人延べ宿泊者数"
                        ),
                        value=f"{prediction:,.0f}"
                    )

                with col2:

                    st.metric(
                        label="예측 시점 / Forecast Period",
                        value=f"{year}.{month:02d}"
                    )

                st.caption(
                    "Predicted Foreign Guest Nights"
                )


        except Exception as e:

            st.error(
                "예측 과정에서 오류가 발생했습니다."
            )

            st.exception(e)


    st.write("")


    # -----------------------------------------------------
    # Professional 설명 카드
    # -----------------------------------------------------
    with st.container(border=True):

        st.subheader(
            "ℹ️ Professional Mode 안내"
        )

        st.write(
            "실무자가 보유한 최근 관광 및 기상 데이터를 "
            "직접 입력하여 모델 예측 결과를 확인하는 기능입니다."
        )

        st.info(
            "현재 모델은 2025년 검증 버전입니다. "
            "향후 최신 데이터 자동 수집 및 재학습 기능으로 "
            "확장할 수 있습니다."
        )