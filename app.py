# =========================================================
# 🇮🇳 INDIA TRAFFIC ACCIDENT ANALYSIS SYSTEM
# =========================================================

import os
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import folium
from streamlit_folium import st_folium

import pyttsx3

from google import genai

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# cd D:\myproject\traffic_accident_project
# streamlit run app.py

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="India Traffic Accident Analysis System",
    page_icon="🚦",
    layout="wide"
)


# =========================================================
# APPLICATION DIRECTORY
# =========================================================

APP_DIR = Path(__file__).resolve().parent


# =========================================================
# REQUIRED DATABASE COLUMNS
# =========================================================

REQUIRED_COLUMNS = [
    "city",
    "state",
    "latitude",
    "longitude",
    "date",
    "time",
    "hour",
    "day_of_week",
    "is_weekend",
    "visibility",
    "temperature",
    "traffic_density",
    "cause",
    "accident_severity",
    "vehicles_involved",
    "casualties",
    "is_peak_hour",
    "festival",
    "risk_score"
]


# =========================================================
# DATABASE FILE NAMES
# =========================================================

SUPPORTED_DATABASES = [
    "taraffic_accidents_india.db",
    "traffic_accidents_india.db",
    "traffic_accidents_original.db",
    "traffic_accidents.db"
]


# =========================================================
# CHECK DATABASE
# =========================================================

def database_has_required_columns(db_path):

    try:

        connection = sqlite3.connect(
            str(db_path),
            check_same_thread=False
        )

        tables = pd.read_sql(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            """,
            connection
        )

        for table_name in tables["name"].tolist():

            safe_table = str(table_name).replace(
                '"',
                '""'
            )

            structure = pd.read_sql(
                f'PRAGMA table_info("{safe_table}")',
                connection
            )

            columns = set(
                structure["name"].tolist()
            )

            if set(REQUIRED_COLUMNS).issubset(columns):

                connection.close()

                return True, table_name

        connection.close()

    except Exception:

        pass

    return False, None


# =========================================================
# FIND DATABASE
# =========================================================

def find_database():

    for db_name in SUPPORTED_DATABASES:

        db_path = APP_DIR / db_name

        if db_path.exists():

            valid, table = database_has_required_columns(
                db_path
            )

            if valid:

                return db_path, table

    for db_path in APP_DIR.glob("*.db"):

        valid, table = database_has_required_columns(
            db_path
        )

        if valid:

            return db_path, table

    return None, None


DB_PATH, ACCIDENT_TABLE = find_database()


# =========================================================
# DATABASE NOT FOUND
# =========================================================

if DB_PATH is None:

    st.error(
        "❌ No compatible accident database found."
    )

    st.info(
        f"""
Put your SQLite database inside:

{APP_DIR}

Database should contain an 'accidents' table
with the required accident columns.
"""
    )

    st.stop()


# =========================================================
# DATABASE CONNECTION
# =========================================================

conn = sqlite3.connect(
    str(DB_PATH),
    check_same_thread=False
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data(
    database_path,
    table_name
):

    safe_table = str(table_name).replace(
        '"',
        '""'
    )

    local_conn = sqlite3.connect(
        str(database_path),
        check_same_thread=False
    )

    try:

        data = pd.read_sql(
            f"""
            SELECT *
            FROM "{safe_table}"
            """,
            local_conn
        )

        return data

    finally:

        local_conn.close()


df = load_data(
    DB_PATH,
    ACCIDENT_TABLE
)


# =========================================================
# CHECK COLUMNS
# =========================================================

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]


if missing_columns:

    st.error(
        "❌ Required database columns are missing."
    )

    st.write(
        missing_columns
    )

    st.stop()


# =========================================================
# DATA CLEANING
# =========================================================

numeric_columns = [
    "latitude",
    "longitude",
    "hour",
    "is_weekend",
    "temperature",
    "vehicles_involved",
    "casualties",
    "is_peak_hour",
    "risk_score"
]


for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df["risk_score"] = df["risk_score"].fillna(0)

df["hour"] = df["hour"].fillna(0)

temperature_median = df["temperature"].median()

if pd.isna(temperature_median):
    temperature_median = 30.0

df["temperature"] = df["temperature"].fillna(
    temperature_median
)

df["vehicles_involved"] = df[
    "vehicles_involved"
].fillna(0)

df["casualties"] = df[
    "casualties"
].fillna(0)

df["is_weekend"] = df[
    "is_weekend"
].fillna(0)

df["is_peak_hour"] = df[
    "is_peak_hour"
].fillna(0)


# =========================================================
# HELPER
# =========================================================

def safe_table_name():

    return ACCIDENT_TABLE.replace(
        '"',
        '""'
    )


def clear_data_cache():

    load_data.clear()


# =========================================================
# GET RECORD
# =========================================================

def get_record(record_id):

    safe_table = safe_table_name()

    query = f"""
        SELECT
            rowid AS record_id,
            *
        FROM "{safe_table}"
        WHERE rowid = ?
        LIMIT 1
    """

    return pd.read_sql(
        query,
        conn,
        params=[record_id]
    )


# =========================================================
# INSERT
# =========================================================

def insert_accident_record(values):

    safe_table = safe_table_name()

    columns = [
        "city",
        "state",
        "latitude",
        "longitude",
        "date",
        "time",
        "hour",
        "day_of_week",
        "is_weekend",
        "visibility",
        "temperature",
        "traffic_density",
        "cause",
        "accident_severity",
        "vehicles_involved",
        "casualties",
        "is_peak_hour",
        "festival",
        "risk_score"
    ]

    placeholders = ", ".join(
        ["?"] * len(columns)
    )

    query = f"""
        INSERT INTO "{safe_table}"
        ({", ".join(columns)})
        VALUES ({placeholders})
    """

    cursor = conn.cursor()

    cursor.execute(
        query,
        values
    )

    conn.commit()

    return cursor.lastrowid


# =========================================================
# UPDATE
# =========================================================

def update_accident_record(
    record_id,
    values
):

    safe_table = safe_table_name()

    columns = [
        "city",
        "state",
        "latitude",
        "longitude",
        "date",
        "time",
        "hour",
        "day_of_week",
        "is_weekend",
        "visibility",
        "temperature",
        "traffic_density",
        "cause",
        "accident_severity",
        "vehicles_involved",
        "casualties",
        "is_peak_hour",
        "festival",
        "risk_score"
    ]

    set_clause = ", ".join(
        [
            f"{column} = ?"
            for column in columns
        ]
    )

    query = f"""
        UPDATE "{safe_table}"
        SET {set_clause}
        WHERE rowid = ?
    """

    cursor = conn.cursor()

    cursor.execute(
        query,
        list(values) + [record_id]
    )

    conn.commit()

    return cursor.rowcount


# =========================================================
# DELETE
# =========================================================

def delete_accident_record(record_id):

    safe_table = safe_table_name()

    query = f"""
        DELETE FROM "{safe_table}"
        WHERE rowid = ?
    """

    cursor = conn.cursor()

    cursor.execute(
        query,
        [record_id]
    )

    conn.commit()

    return cursor.rowcount


# =========================================================
# GEMINI API
# =========================================================

GEMINI_MODEL = "gemini-2.5-flash"


def get_gemini_api_key():

    # First try Streamlit secrets
    try:

        key = str(
            st.secrets.get(
                "GEMINI_API_KEY",
                ""
            )
        ).strip()

        if key:

            return key

    except Exception:

        pass

    # Then environment variable
    key = os.environ.get(
        "GEMINI_API_KEY",
        ""
    ).strip()

    if key:

        return key

    # Session state
    return st.session_state.get(
        "gemini_api_key",
        ""
    ).strip()


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_gemini_client():

    key = get_gemini_api_key()

    if not key:

        return None

    try:

        return genai.Client(
            api_key=key
        )

    except Exception:

        return None


# =========================================================
# VOICE
# =========================================================

def speak_text(text):

    try:

        engine = pyttsx3.init()

        engine.setProperty(
            "rate",
            150
        )

        engine.say(
            str(text)
        )

        engine.runAndWait()

        engine.stop()

        return True, ""

    except Exception as e:

        return False, str(e)


# =========================================================
# CLEAN SQL
# =========================================================

def clean_sql(sql):

    sql = str(sql).strip()

    sql = re.sub(
        r"^```sql\s*",
        "",
        sql,
        flags=re.IGNORECASE
    )

    sql = re.sub(
        r"^```\s*",
        "",
        sql
    )

    sql = re.sub(
        r"\s*```$",
        "",
        sql
    )

    sql = sql.strip()

    # Remove accidental trailing semicolon
    sql = sql.rstrip(";").strip()

    return sql


# =========================================================
# SAFE SELECT
# =========================================================

def is_safe_select(sql):

    sql = clean_sql(sql)

    if not re.match(
        r"^(SELECT|WITH)\b",
        sql,
        re.IGNORECASE
    ):

        return False

    blocked = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "REPLACE",
        "TRUNCATE",
        "ATTACH",
        "DETACH",
        "PRAGMA"
    ]

    upper_sql = sql.upper()

    for word in blocked:

        if re.search(
            rf"\b{word}\b",
            upper_sql
        ):

            return False

    # Multiple statements are not allowed
    if ";" in sql:

        return False

    return True


# =========================================================
# GEMINI SQL
# =========================================================

def generate_sql_from_question(
    question
):

    client = get_gemini_client()

    if client is None:

        return None

    schema = """
TABLE: accidents

city TEXT
state TEXT
latitude REAL
longitude REAL
date TEXT
time TEXT
hour INTEGER
day_of_week TEXT
is_weekend INTEGER
visibility TEXT
temperature REAL
traffic_density TEXT
cause TEXT
accident_severity TEXT
vehicles_involved INTEGER
casualties INTEGER
is_peak_hour INTEGER
festival TEXT
risk_score REAL
"""

    prompt = f"""
You are a SQL expert.

Database is SQLite.

Schema:

{schema}

User question:

{question}

Generate ONE read-only SQLite SELECT query.

Rules:
- Use only accidents table.
- Use only available columns.
- Do not modify database.
- Return only SQL.
- No markdown.
- Use SQLite syntax.
- For highest/lowest questions, use ORDER BY and LIMIT.
"""

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        if not response:

            return None

        sql = clean_sql(
            response.text
        )

        if not is_safe_select(sql):

            return None

        return sql

    except Exception:

        return None


# =========================================================
# GEMINI ANSWER
# =========================================================

def generate_ai_answer(
    question,
    sql,
    result
):

    client = get_gemini_client()

    if client is None:

        return None

    if result.empty:

        result_text = "No records found."

    else:

        result_text = result.head(
            50
        ).to_string(
            index=False
        )

    prompt = f"""
You are an AI assistant for an India Traffic Accident Analysis System.

Question:
{question}

SQL:
{sql}

Result:
{result_text}

Give a short, clear answer based only on the result.

Do not invent information.
Do not mention API errors.
Do not explain the SQL unless necessary.
"""

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        if response and response.text:

            return response.text.strip()

    except Exception:

        pass

    return None


# =========================================================
# LOCAL FALLBACK SQL
# =========================================================

def local_fallback_query(question):

    q = question.lower().strip()

    # -----------------------------------------------------
    # TOTAL ACCIDENTS
    # -----------------------------------------------------

    if (
        "total accident" in q
        or "how many accident" in q
        or "number of accident" in q
    ):

        return """
        SELECT COUNT(*) AS total_accidents
        FROM accidents
        """, "Total accidents"


    # -----------------------------------------------------
    # TOTAL CITIES
    # -----------------------------------------------------

    if (
        "total cit" in q
        or "how many cit" in q
        or "number of cit" in q
    ):

        return """
        SELECT COUNT(DISTINCT city) AS total_cities
        FROM accidents
        """, "Total cities"


    # -----------------------------------------------------
    # TOTAL STATES
    # -----------------------------------------------------

    if (
        "total state" in q
        or "how many state" in q
        or "number of state" in q
    ):

        return """
        SELECT COUNT(DISTINCT state) AS total_states
        FROM accidents
        """, "Total states"


    # -----------------------------------------------------
    # TOTAL CASUALTIES
    # -----------------------------------------------------

    if (
        "total casualties" in q
        or "total casualty" in q
        or "how many casualties" in q
    ):

        return """
        SELECT SUM(casualties) AS total_casualties
        FROM accidents
        """, "Total casualties"


    # -----------------------------------------------------
    # HIGH RISK
    # -----------------------------------------------------

    if (
        "high risk" in q
        or "high-risk" in q
    ):

        return """
        SELECT COUNT(*) AS high_risk_accidents
        FROM accidents
        WHERE risk_score >= 0.70
        """, "High risk accidents"


    # -----------------------------------------------------
    # AVERAGE RISK
    # -----------------------------------------------------

    if (
        "average risk" in q
        or "avg risk" in q
        or "mean risk" in q
    ):

        return """
        SELECT ROUND(AVG(risk_score), 2) AS average_risk
        FROM accidents
        """, "Average risk score"


    # -----------------------------------------------------
    # HIGHEST RISK LOCATION
    # -----------------------------------------------------

    if (
        ("highest" in q or "top" in q)
        and (
            "risk location" in q
            or "risky location" in q
            or "risky city" in q
        )
    ):

        return """
        SELECT
            city,
            state,
            COUNT(*) AS accident_count,
            ROUND(AVG(risk_score), 2) AS average_risk
        FROM accidents
        GROUP BY city, state
        ORDER BY average_risk DESC
        LIMIT 1
        """, "Highest risk location"


    # -----------------------------------------------------
    # STATE WITH HIGHEST ACCIDENTS
    # -----------------------------------------------------

    if (
        (
            "highest number of accidents" in q
            or "most accidents" in q
            or "maximum accidents" in q
            or "highest accidents" in q
        )
        and "state" in q
    ):

        return """
        SELECT
            state,
            COUNT(*) AS accident_count
        FROM accidents
        GROUP BY state
        ORDER BY accident_count DESC
        LIMIT 1
        """, "State with highest accidents"


    # -----------------------------------------------------
    # CITY WITH HIGHEST ACCIDENTS
    # -----------------------------------------------------

    if (
        (
            "highest number of accidents" in q
            or "most accidents" in q
            or "maximum accidents" in q
            or "highest accidents" in q
        )
        and "city" in q
    ):

        return """
        SELECT
            city,
            state,
            COUNT(*) AS accident_count
        FROM accidents
        GROUP BY city, state
        ORDER BY accident_count DESC
        LIMIT 1
        """, "City with highest accidents"


    # -----------------------------------------------------
    # MOST CASUALTIES STATE
    # -----------------------------------------------------

    if (
        (
            "most casualties" in q
            or "highest casualties" in q
            or "maximum casualties" in q
        )
        and "state" in q
    ):

        return """
        SELECT
            state,
            SUM(casualties) AS total_casualties
        FROM accidents
        GROUP BY state
        ORDER BY total_casualties DESC
        LIMIT 1
        """, "State with highest casualties"


    # -----------------------------------------------------
    # MOST COMMON CAUSE
    # -----------------------------------------------------

    if (
        "most common cause" in q
        or "common cause" in q
        or "main cause" in q
        or "most frequent cause" in q
    ):

        return """
        SELECT
            cause,
            COUNT(*) AS accident_count
        FROM accidents
        GROUP BY cause
        ORDER BY accident_count DESC
        LIMIT 1
        """, "Most common accident cause"


    # -----------------------------------------------------
    # MOST COMMON SEVERITY
    # -----------------------------------------------------

    if (
        "most common severity" in q
        or "common severity" in q
        or "most frequent severity" in q
    ):

        return """
        SELECT
            accident_severity,
            COUNT(*) AS accident_count
        FROM accidents
        GROUP BY accident_severity
        ORDER BY accident_count DESC
        LIMIT 1
        """, "Most common severity"


    # -----------------------------------------------------
    # PEAK HOUR
    # -----------------------------------------------------

    if (
        "peak hour" in q
        or "busiest hour" in q
        or "most accidents hour" in q
    ):

        return """
        SELECT
            hour,
            COUNT(*) AS accident_count
        FROM accidents
        GROUP BY hour
        ORDER BY accident_count DESC
        LIMIT 1
        """, "Busiest accident hour"


    # -----------------------------------------------------
    # TRAFFIC DENSITY
    # -----------------------------------------------------

    if (
        "traffic density" in q
        or "most common traffic" in q
    ):

        return """
        SELECT
            traffic_density,
            COUNT(*) AS accident_count
        FROM accidents
        GROUP BY traffic_density
        ORDER BY accident_count DESC
        LIMIT 1
        """, "Most common traffic density"


    # -----------------------------------------------------
    # SEVERE ACCIDENTS
    # -----------------------------------------------------

    if (
        "severe accident" in q
        or "fatal accident" in q
        or "major accident" in q
    ):

        return """
        SELECT COUNT(*) AS severe_accidents
        FROM accidents
        WHERE LOWER(accident_severity)
        IN ('fatal', 'major')
        """, "Severe accidents"


    # -----------------------------------------------------
    # TOP 10 STATES
    # -----------------------------------------------------

    if (
        "top 10 state" in q
        or "top ten state" in q
        or "top states" in q
    ):

        return """
        SELECT
            state,
            COUNT(*) AS accident_count
        FROM accidents
        GROUP BY state
        ORDER BY accident_count DESC
        LIMIT 10
        """, "Top 10 states"


    # -----------------------------------------------------
    # TOP 10 CITIES
    # -----------------------------------------------------

    if (
        "top 10 cit" in q
        or "top ten cit" in q
        or "top cities" in q
    ):

        return """
        SELECT
            city,
            state,
            COUNT(*) AS accident_count
        FROM accidents
        GROUP BY city, state
        ORDER BY accident_count DESC
        LIMIT 10
        """, "Top 10 cities"


    return None, None


# =========================================================
# LOCAL ANSWER GENERATOR
# =========================================================

def local_answer(
    question,
    result,
    title
):

    if result.empty:

        return "No accident records were found."

    try:

        row = result.iloc[0]

        # Single-value results
        if len(result.columns) == 1:

            column = result.columns[0]

            value = row[column]

            if pd.isna(value):

                value = 0

            return f"{title}: {value}"

        # Highest risk location
        if (
            "city" in result.columns
            and "state" in result.columns
            and "average_risk" in result.columns
        ):

            return (
                f"The highest-risk location is "
                f"{row['city']}, {row['state']} "
                f"with an average risk score of "
                f"{float(row['average_risk']):.2f}."
            )

        # State result
        if (
            "state" in result.columns
            and "accident_count" in result.columns
        ):

            return (
                f"{row['state']} has the highest number "
                f"of accidents with "
                f"{int(row['accident_count']):,} accidents."
            )

        # City result
        if (
            "city" in result.columns
            and "accident_count" in result.columns
        ):

            return (
                f"{row['city']}, {row['state']} has "
                f"{int(row['accident_count']):,} accidents."
            )

        # Cause
        if (
            "cause" in result.columns
            and "accident_count" in result.columns
        ):

            return (
                f"The most common accident cause is "
                f"{row['cause']} with "
                f"{int(row['accident_count']):,} accidents."
            )

        # Generic multi-column result
        values = []

        for column in result.columns:

            value = row[column]

            if pd.isna(value):
                value = "Unknown"

            values.append(
                f"{column}: {value}"
            )

        return " | ".join(values)

    except Exception:

        return (
            f"{title}: "
            f"{result.iloc[0].to_dict()}"
        )


# =========================================================
# RUN LOCAL FALLBACK
# =========================================================

def run_local_fallback(question):

    sql, title = local_fallback_query(
        question
    )

    if sql is None:

        return None, None, None

    try:

        result = pd.read_sql(
            sql,
            conn
        )

        answer = local_answer(
            question,
            result,
            title
        )

        return sql, result, answer

    except Exception:

        return None, None, None


# =========================================================
# RUN GEMINI + FALLBACK
# =========================================================

def answer_question(question):

    # First try Gemini
    generated_sql = generate_sql_from_question(
        question
    )

    if generated_sql:

        try:

            result = pd.read_sql(
                generated_sql,
                conn
            )

            ai_answer = generate_ai_answer(
                question,
                generated_sql,
                result
            )

            if ai_answer:

                return (
                    generated_sql,
                    result,
                    ai_answer
                )

            # Gemini answer failed but SQL worked
            fallback_answer = local_answer(
                question,
                result,
                "Query result"
            )

            return (
                generated_sql,
                result,
                fallback_answer
            )

        except Exception:

            pass

    # Gemini unavailable → local SQLite fallback
    return run_local_fallback(
        question
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🚦 India Traffic Accident"
)



menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Data Analytics",
        "📄 Database Overview",
        "🧠 Intelligent Accident Insights",
        "📍 Future Risk Forecasting",
        "🗺️ Accident Map",
        "🤖 ML Prediction"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if menu == "🏠 Dashboard":

    st.header(
        "🇮🇳 India Traffic Accident Dashboard"
    )

    st.write(
        "Traffic accident analytics, risk analysis and machine learning system."
    )

    st.divider()

    total_accidents = len(df)

    total_cities = df["city"].nunique()

    total_states = df["state"].nunique()

    total_casualties = int(
        df["casualties"].sum()
    )

    high_risk = int(
        (
            df["risk_score"] >= 0.70
        ).sum()
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(
            "🚗 Total Accidents",
            f"{total_accidents:,}"
        )

    with c2:

        st.metric(
            "🏙️ Cities",
            f"{total_cities:,}"
        )

    with c3:

        st.metric(
            "🗺️ States",
            f"{total_states:,}"
        )

    with c4:

        st.metric(
            "🚑 Casualties",
            f"{total_casualties:,}"
        )

    with c5:

        st.metric(
            "⚠️ High Risk",
            f"{high_risk:,}"
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        state_counts = (
            df["state"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        state_counts.columns = [
            "state",
            "accidents"
        ]

        fig = px.bar(
            state_counts,
            x="state",
            y="accidents",
            title="Top 10 States by Accidents"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        severity_counts = (
            df["accident_severity"]
            .value_counts()
            .reset_index()
        )

        severity_counts.columns = [
            "severity",
            "accidents"
        ]

        fig = px.pie(
            severity_counts,
            names="severity",
            values="accidents",
            title="Accident Severity"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    hourly = (
        df.groupby("hour")
        .size()
        .reset_index(
            name="accidents"
        )
    )

    fig = px.line(
        hourly,
        x="hour",
        y="accidents",
        markers=True,
        title="Accidents by Hour"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# DATA ANALYTICS
# =========================================================

elif menu == "📊 Data Analytics":

    st.header(
        "📊 Traffic Accident Data Analytics"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        states = sorted(
            df["state"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_state = st.selectbox(
            "Select State",
            ["All"] + states
        )

    with col2:

        severity_list = sorted(
            df["accident_severity"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_severity = st.selectbox(
            "Select Severity",
            ["All"] + severity_list
        )

    with col3:

        density_list = sorted(
            df["traffic_density"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_density = st.selectbox(
            "Traffic Density",
            ["All"] + density_list
        )

    filtered_df = df.copy()

    if selected_state != "All":

        filtered_df = filtered_df[
            filtered_df["state"].astype(str)
            == selected_state
        ]

    if selected_severity != "All":

        filtered_df = filtered_df[
            filtered_df["accident_severity"].astype(str)
            == selected_severity
        ]

    if selected_density != "All":

        filtered_df = filtered_df[
            filtered_df["traffic_density"].astype(str)
            == selected_density
        ]

    st.divider()

    a1, a2, a3, a4 = st.columns(4)

    with a1:

        st.metric(
            "Filtered Accidents",
            f"{len(filtered_df):,}"
        )

    with a2:

        avg_risk = filtered_df["risk_score"].mean()

        if pd.isna(avg_risk):
            avg_risk = 0

        st.metric(
            "Average Risk",
            f"{avg_risk:.2f}"
        )

    with a3:

        avg_casualties = filtered_df["casualties"].mean()

        if pd.isna(avg_casualties):
            avg_casualties = 0

        st.metric(
            "Average Casualties",
            f"{avg_casualties:.2f}"
        )

    with a4:

        avg_vehicles = filtered_df["vehicles_involved"].mean()

        if pd.isna(avg_vehicles):
            avg_vehicles = 0

        st.metric(
            "Average Vehicles",
            f"{avg_vehicles:.2f}"
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        cause_data = (
            filtered_df["cause"]
            .value_counts()
            .reset_index()
        )

        cause_data.columns = [
            "cause",
            "accidents"
        ]

        fig = px.bar(
            cause_data,
            x="cause",
            y="accidents",
            title="Accidents by Cause"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        density_data = (
            filtered_df["traffic_density"]
            .value_counts()
            .reset_index()
        )

        density_data.columns = [
            "density",
            "accidents"
        ]

        fig = px.pie(
            density_data,
            names="density",
            values="accidents",
            title="Traffic Density"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    day_data = (
        filtered_df.groupby(
            "day_of_week"
        )
        .size()
        .reindex(
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ],
            fill_value=0
        )
        .reset_index(
            name="accidents"
        )
    )

    day_data.columns = [
        "day",
        "accidents"
    ]

    fig = px.bar(
        day_data,
        x="day",
        y="accidents",
        title="Accidents by Day of Week"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# DATABASE OVERVIEW
# =========================================================

elif menu == "📄 Database Overview":

    st.header(
        "📄 Accident Database Overview"
    )

    st.write(
        "View accident records and manage database data."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🚗 Total Accidents",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "🏙️ Total Cities",
            f"{df['city'].nunique():,}"
        )

    with col3:

        st.metric(
            "🗺️ Total States",
            f"{df['state'].nunique():,}"
        )

    st.divider()

    st.subheader(
        "📄 Accident Data"
    )

    show_records = st.slider(
        "Number of records",
        10,
        min(1000, len(df)),
        min(100, len(df)),
        10
    )

    safe_table = safe_table_name()

    sample = pd.read_sql(
        f"""
        SELECT
            rowid AS record_id,
            *
        FROM "{safe_table}"
        LIMIT {show_records}
        """,
        conn
    )

    st.dataframe(
        sample,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "🛠️ Database Management"
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "➕ Add Data",
            "✏️ Edit Data",
            "🗑️ Delete Data"
        ]
    )

    # =====================================================
    # ADD
    # =====================================================

    with tab1:

        st.subheader(
            "➕ Add New Accident"
        )

        with st.form(
            "add_accident_form"
        ):

            col1, col2, col3 = st.columns(3)

            with col1:

                city = st.text_input(
                    "City"
                )

                state = st.text_input(
                    "State"
                )

                latitude = st.number_input(
                    "Latitude",
                    value=23.0225,
                    format="%.6f"
                )

                longitude = st.number_input(
                    "Longitude",
                    value=72.5714,
                    format="%.6f"
                )

                date = st.text_input(
                    "Date",
                    value="2026-09-11"
                )

                time = st.text_input(
                    "Time",
                    value="12:00"
                )

                hour = st.number_input(
                    "Hour",
                    0,
                    23,
                    12
                )

            with col2:

                days = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday"
                ]

                day_of_week = st.selectbox(
                    "Day of Week",
                    days
                )

                is_weekend = st.selectbox(
                    "Is Weekend",
                    [0, 1]
                )

                visibility = st.text_input(
                    "Visibility"
                )

                temperature = st.number_input(
                    "Temperature",
                    value=30.0
                )

                traffic_density = st.selectbox(
                    "Traffic Density",
                    [
                        "low",
                        "medium",
                        "high"
                    ]
                )

                cause = st.selectbox(
                    "Cause",
                    [
                        "weather",
                        "distraction",
                        "poor road",
                        "overspeeding"
                    ]
                )

            with col3:

                accident_severity = st.selectbox(
                    "Accident Severity",
                    [
                        "fatal",
                        "major",
                        "minor"
                    ]
                )

                vehicles_involved = st.number_input(
                    "Vehicles Involved",
                    min_value=1,
                    value=2
                )

                casualties = st.number_input(
                    "Casualties",
                    min_value=0,
                    value=1
                )

                is_peak_hour = st.selectbox(
                    "Is Peak Hour",
                    [0, 1]
                )

                festival = st.text_input(
                    "Festival"
                )

                risk_score = st.number_input(
                    "Risk Score",
                    0.0,
                    1.0,
                    0.50,
                    0.01
                )

            add_button = st.form_submit_button(
                "➕ Add Accident"
            )

        if add_button:

            if not city.strip():

                st.error(
                    "❌ Please enter City."
                )

            elif not state.strip():

                st.error(
                    "❌ Please enter State."
                )

            else:

                try:

                    values = [
                        city,
                        state,
                        latitude,
                        longitude,
                        date,
                        time,
                        hour,
                        day_of_week,
                        is_weekend,
                        visibility,
                        temperature,
                        traffic_density,
                        cause,
                        accident_severity,
                        vehicles_involved,
                        casualties,
                        is_peak_hour,
                        festival,
                        risk_score
                    ]

                    new_id = insert_accident_record(
                        values
                    )

                    clear_data_cache()

                    st.success(
                        f"✅ Accident added successfully! Record ID: {new_id}"
                    )

                    st.rerun()

                except Exception as e:

                    st.error(
                        f"❌ Error: {e}"
                    )


    # =====================================================
    # EDIT
    # =====================================================

    with tab2:

        st.subheader(
            "✏️ Edit Accident"
        )

        edit_id = st.number_input(
            "Enter Record ID",
            min_value=1,
            step=1,
            value=1,
            key="edit_record_id"
        )

        if st.button(
            "🔍 Find Accident",
            key="find_edit"
        ):

            edit_data = get_record(
                edit_id
            )

            if edit_data.empty:

                st.warning(
                    "⚠️ Record ID not found."
                )

                st.session_state[
                    "edit_found"
                ] = False

            else:

                st.session_state[
                    "edit_found"
                ] = True

                st.session_state[
                    "edit_record_id_saved"
                ] = int(
                    edit_data.iloc[0]["record_id"]
                )

                st.success(
                    "✅ Accident found!"
                )

        if st.session_state.get(
            "edit_found",
            False
        ):

            rowid = st.session_state[
                "edit_record_id_saved"
            ]

            edit_data = get_record(
                rowid
            )

            if not edit_data.empty:

                row = edit_data.iloc[0]

                def text_value(value):

                    if pd.isna(value):

                        return ""

                    return str(value)

                with st.form(
                    "edit_accident_form"
                ):

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        new_city = st.text_input(
                            "City",
                            text_value(
                                row["city"]
                            )
                        )

                        new_state = st.text_input(
                            "State",
                            text_value(
                                row["state"]
                            )
                        )

                        new_latitude = st.number_input(
                            "Latitude",
                            value=float(
                                row["latitude"]
                            )
                        )

                        new_longitude = st.number_input(
                            "Longitude",
                            value=float(
                                row["longitude"]
                            )
                        )

                        new_date = st.text_input(
                            "Date",
                            text_value(
                                row["date"]
                            )
                        )

                        new_time = st.text_input(
                            "Time",
                            text_value(
                                row["time"]
                            )
                        )

                    with col2:

                        new_hour = st.number_input(
                            "Hour",
                            0,
                            23,
                            int(
                                row["hour"]
                            )
                        )

                        days = [
                            "Monday",
                            "Tuesday",
                            "Wednesday",
                            "Thursday",
                            "Friday",
                            "Saturday",
                            "Sunday"
                        ]

                        current_day = text_value(
                            row["day_of_week"]
                        )

                        day_index = (
                            days.index(
                                current_day
                            )
                            if current_day in days
                            else 0
                        )

                        new_day = st.selectbox(
                            "Day of Week",
                            days,
                            index=day_index
                        )

                        new_weekend = st.selectbox(
                            "Is Weekend",
                            [0, 1],
                            index=int(
                                row["is_weekend"]
                            )
                        )

                        new_visibility = st.text_input(
                            "Visibility",
                            text_value(
                                row["visibility"]
                            )
                        )

                        new_temperature = st.number_input(
                            "Temperature",
                            value=float(
                                row["temperature"]
                            )
                        )

                        densities = [
                            "low",
                            "medium",
                            "high"
                        ]

                        current_density = text_value(
                            row["traffic_density"]
                        )

                        density_index = (
                            densities.index(
                                current_density
                            )
                            if current_density in densities
                            else 0
                        )

                        new_density = st.selectbox(
                            "Traffic Density",
                            densities,
                            index=density_index
                        )

                    with col3:

                        causes = [
                            "weather",
                            "distraction",
                            "poor road",
                            "overspeeding"
                        ]

                        current_cause = text_value(
                            row["cause"]
                        )

                        cause_index = (
                            causes.index(
                                current_cause
                            )
                            if current_cause in causes
                            else 0
                        )

                        new_cause = st.selectbox(
                            "Cause",
                            causes,
                            index=cause_index
                        )

                        severities = [
                            "fatal",
                            "major",
                            "minor"
                        ]

                        current_severity = text_value(
                            row["accident_severity"]
                        )

                        severity_index = (
                            severities.index(
                                current_severity
                            )
                            if current_severity in severities
                            else 0
                        )

                        new_severity = st.selectbox(
                            "Accident Severity",
                            severities,
                            index=severity_index
                        )

                        new_vehicles = st.number_input(
                            "Vehicles Involved",
                            min_value=1,
                            value=int(
                                row["vehicles_involved"]
                            )
                        )

                        new_casualties = st.number_input(
                            "Casualties",
                            min_value=0,
                            value=int(
                                row["casualties"]
                            )
                        )

                        new_peak = st.selectbox(
                            "Is Peak Hour",
                            [0, 1],
                            index=int(
                                row["is_peak_hour"]
                            )
                        )

                        new_festival = st.text_input(
                            "Festival",
                            text_value(
                                row["festival"]
                            )
                        )

                        new_risk = st.number_input(
                            "Risk Score",
                            0.0,
                            1.0,
                            float(
                                row["risk_score"]
                            ),
                            0.01
                        )

                    update_button = st.form_submit_button(
                        "💾 Update Accident"
                    )

                if update_button:

                    try:

                        values = [
                            new_city,
                            new_state,
                            new_latitude,
                            new_longitude,
                            new_date,
                            new_time,
                            new_hour,
                            new_day,
                            new_weekend,
                            new_visibility,
                            new_temperature,
                            new_density,
                            new_cause,
                            new_severity,
                            new_vehicles,
                            new_casualties,
                            new_peak,
                            new_festival,
                            new_risk
                        ]

                        affected = update_accident_record(
                            rowid,
                            values
                        )

                        if affected > 0:

                            clear_data_cache()

                            st.session_state[
                                "edit_found"
                            ] = False

                            st.success(
                                "✅ Accident updated successfully!"
                            )

                            st.rerun()

                        else:

                            st.error(
                                "❌ Record could not be updated."
                            )

                    except Exception as e:

                        st.error(
                            f"❌ Update Error: {e}"
                        )


    # =====================================================
    # DELETE
    # =====================================================

    with tab3:

        st.subheader(
            "🗑️ Delete Accident"
        )

        st.warning(
            "⚠️ Delete operation is permanent. "
            "Once a record is deleted, it cannot be recovered."
        )

        delete_id = st.number_input(
            "Enter Record ID",
            min_value=1,
            step=1,
            value=1,
            key="delete_record_id"
        )

        confirm_delete = st.checkbox(
            "I understand that this deletion is permanent."
        )

        if st.button(
            "🗑️ Delete Accident",
            key="delete_button"
        ):

            if not confirm_delete:

                st.error(
                    "❌ Please confirm permanent deletion first."
                )

            else:

                try:

                    affected = delete_accident_record(
                        int(delete_id)
                    )

                    if affected > 0:

                        clear_data_cache()

                        st.success(
                            "✅ Accident deleted successfully!"
                        )

                        st.rerun()

                    else:

                        st.warning(
                            "⚠️ Record ID not found."
                        )

                except Exception as e:

                    st.error(
                        f"❌ Delete Error: {e}"
                    )


    st.divider()

    st.subheader(
        "🔎 Accidents Table Structure"
    )

    schema = pd.read_sql(
        f"""
        PRAGMA table_info("{safe_table}")
        """,
        conn
    )

    st.dataframe(
        schema[
            [
                "cid",
                "name",
                "type"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# INTELLIGENT ACCIDENT INSIGHTS
# =========================================================

elif menu == "🧠 Intelligent Accident Insights":

    st.header(
        "🧠 Intelligent Accident Insights"
    )

    st.write(
        "Explore accident data using SQL queries, voice answers and Gemini AI."
    )

    st.divider()

    # =====================================================
    # KEY ACCIDENT INSIGHTS
    # =====================================================

    st.subheader(
        "📌 Key Accident Insights"
    )

    q1, q2, q3 = st.columns(3)

    q4, q5 = st.columns(2)


    def save_quick_result(
        title,
        query,
        voice_function
    ):

        try:

            result = pd.read_sql(
                query,
                conn
            )

            st.session_state[
                "quick_query_result"
            ] = result

            st.session_state[
                "quick_query_title"
            ] = title

            st.session_state[
                "quick_query_voice"
            ] = voice_function(
                result
            )

        except Exception as e:

            st.error(
                f"❌ SQL Error: {e}"
            )


    with q1:

        if st.button(
            "🚗 Total Accidents",
            use_container_width=True
        ):

            save_quick_result(
                "🚗 Total Accidents",
                """
                SELECT COUNT(*) AS total_accidents
                FROM accidents
                """,
                lambda r:
                    f"Total accidents are {r.iloc[0, 0]:,}"
            )


    with q2:

        if st.button(
            "🏙️ Total Cities",
            use_container_width=True
        ):

            save_quick_result(
                "🏙️ Total Cities",
                """
                SELECT COUNT(DISTINCT city) AS total_cities
                FROM accidents
                """,
                lambda r:
                    f"Total cities are {r.iloc[0, 0]:,}"
            )


    with q3:

        if st.button(
            "🗺️ Total States",
            use_container_width=True
        ):

            save_quick_result(
                "🗺️ Total States",
                """
                SELECT COUNT(DISTINCT state) AS total_states
                FROM accidents
                """,
                lambda r:
                    f"Total states are {r.iloc[0, 0]:,}"
            )


    with q4:

        if st.button(
            "⚠️ High Risk Accidents",
            use_container_width=True
        ):

            save_quick_result(
                "⚠️ High Risk Accidents",
                """
                SELECT COUNT(*) AS high_risk_accidents
                FROM accidents
                WHERE risk_score >= 0.70
                """,
                lambda r:
                    f"High risk accidents are {r.iloc[0, 0]:,}"
            )


    with q5:

        if st.button(
            "🚑 Total Casualties",
            use_container_width=True
        ):

            save_quick_result(
                "🚑 Total Casualties",
                """
                SELECT SUM(casualties) AS total_casualties
                FROM accidents
                """,
                lambda r:
                    f"Total casualties are {r.iloc[0, 0]:,}"
            )


    # =====================================================
    # QUICK RESULT
    # =====================================================

    if "quick_query_result" in st.session_state:

        st.divider()

        st.subheader(
            st.session_state[
                "quick_query_title"
            ]
        )

        result = st.session_state[
            "quick_query_result"
        ]

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

        voice_text = st.session_state.get(
            "quick_query_voice",
            ""
        )

        if voice_text:

            st.info(
                voice_text
            )

            if st.button(
                "🔊 Speak Result",
                key="speak_quick_result"
            ):

                ok, error = speak_text(
                    voice_text
                )

                if ok:

                    st.success(
                        "🔊 Result spoken."
                    )

                else:

                    st.error(
                        f"Voice Error: {error}"
                    )


    st.divider()

    # =====================================================
    # NATURAL LANGUAGE ASSISTANT
    # =====================================================

    st.subheader(
        "💬 Natural-Language Data Assistant"
    )

    question = st.text_area(
        "Ask a question about accident data",
        placeholder=(
            "Example: Which state has the highest number of accidents?"
        ),
        key="natural_language_question"
    )

    if st.button(
        "🤖 Ask Gemini",
        key="ask_gemini_button"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            generated_sql, result, answer = answer_question(
                question
            )

            if generated_sql is not None:

                st.subheader(
                    "🧾 Generated SQL"
                )

                st.code(
                    generated_sql,
                    language="sql"
                )

                st.subheader(
                    "📊 Query Result"
                )

                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True
                )

                if answer:

                    st.subheader(
                        "🧠 AI Answer"
                    )

                    st.success(
                        answer
                    )

                    if st.button(
                        "🔊 Speak AI Answer",
                        key="speak_ai_answer"
                    ):

                        ok, error = speak_text(
                            answer
                        )

                        if not ok:

                            st.error(
                                f"Voice Error: {error}"
                            )

            else:

                st.info(
                    "I could not understand that question. "
                    "Try asking about accidents, cities, states, "
                    "casualties, causes, risk or severity."
                )


# =========================================================
# FUTURE RISK FORECASTING
# =========================================================

elif menu == "📍 Future Risk Forecasting":

    st.header(
        "📍 Future Risk Forecasting"
    )

    location_summary = (
        df.groupby(
            ["city", "state"],
            dropna=False
        )
        .agg(
            accident_count=(
                "risk_score",
                "count"
            ),
            average_risk=(
                "risk_score",
                "mean"
            ),
            total_casualties=(
                "casualties",
                "sum"
            ),
            severe_accidents=(
                "accident_severity",
                lambda x:
                    x.astype(str)
                    .str.lower()
                    .isin(
                        [
                            "fatal",
                            "major"
                        ]
                    )
                    .sum()
            )
        )
        .reset_index()
    )

    location_summary["severe_rate"] = (
        location_summary["severe_accidents"]
        /
        location_summary["accident_count"]
    )

    average_casualties = (
        location_summary[
            "total_casualties"
        ]
        /
        location_summary[
            "accident_count"
        ].replace(
            0,
            1
        )
    )

    location_summary["forecast_score"] = (
        location_summary["average_risk"] * 0.45
        +
        location_summary["severe_rate"] * 0.35
        +
        (
            average_casualties.clip(
                0,
                5
            )
            / 5
            * 0.20
        )
    )

    location_summary = location_summary.sort_values(
        "forecast_score",
        ascending=False
    )

    st.subheader(
        "🔥 Highest-Risk Locations"
    )

    st.dataframe(
        location_summary.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    chart_data = location_summary.head(
        10
    ).copy()

    chart_data["location"] = (
        chart_data["city"].astype(str)
        +
        ", "
        +
        chart_data["state"].astype(str)
    )

    fig = px.bar(
        chart_data,
        x="forecast_score",
        y="location",
        orientation="h",
        title="Top 10 Risk Locations"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    


# =========================================================
# ACCIDENT MAP
# =========================================================

elif menu == "🗺️ Accident Map":

    st.header(
        "🗺️ India Traffic Accident Map"
    )

    map_df = df.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    ).copy()

    map_df = map_df[
        map_df["latitude"].between(
            6,
            38
        )
        &
        map_df["longitude"].between(
            68,
            98
        )
    ]

    st.write(
        f"Showing {len(map_df):,} valid accident locations."
    )

    if map_df.empty:

        st.warning(
            "No valid India coordinates available."
        )

    else:

        india_map = folium.Map(
            location=[
                22.9734,
                78.6569
            ],
            zoom_start=5
        )

        marker_df = map_df.head(
            1000
        )

        for _, row in marker_df.iterrows():

            risk = float(
                row["risk_score"]
            )

            if risk >= 0.70:

                marker_color = "red"

            elif risk >= 0.40:

                marker_color = "orange"

            else:

                marker_color = "green"

            popup_text = f"""
            <b>City:</b> {row['city']}<br>
            <b>State:</b> {row['state']}<br>
            <b>Cause:</b> {row['cause']}<br>
            <b>Severity:</b> {row['accident_severity']}<br>
            <b>Vehicles:</b> {row['vehicles_involved']}<br>
            <b>Casualties:</b> {row['casualties']}<br>
            <b>Risk Score:</b> {risk:.2f}
            """

            folium.CircleMarker(
                location=[
                    row["latitude"],
                    row["longitude"]
                ],
                radius=5,
                popup=folium.Popup(
                    popup_text,
                    max_width=300
                ),
                color=marker_color,
                fill=True
            ).add_to(
                india_map
            )

        st_folium(
            india_map,
            width=1200,
            height=600
        )


# =========================================================
# MACHINE LEARNING
# =========================================================

elif menu == "🤖 ML Prediction":

    st.header(
        "🤖 Machine Learning Accident Risk Prediction"
    )

    st.write(
        "Random Forest model for accident risk score prediction."
    )

    st.divider()

    feature_columns = [
        "hour",
        "temperature",
        "vehicles_involved",
        "casualties",
        "is_weekend",
        "is_peak_hour"
    ]

    target_column = "risk_score"

    ml_df = df[
        feature_columns
        +
        [target_column]
    ].copy()

    ml_df = ml_df.dropna()

    if len(ml_df) < 20:

        st.error(
            "Not enough data for ML training."
        )

        st.stop()

    X = ml_df[
        feature_columns
    ]

    y = ml_df[
        target_column
    ]

    # =====================================================
    # TRAIN TEST
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # =====================================================
    # MODEL
    # =====================================================

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    with st.spinner(
        "Training Random Forest model..."
    ):

        model.fit(
            X_train,
            y_train
        )

    # =====================================================
    # PREDICTIONS
    # =====================================================

    train_prediction = model.predict(
        X_train
    )

    test_prediction = model.predict(
        X_test
    )

    # =====================================================
    # METRICS
    # =====================================================

    train_r2 = r2_score(
        y_train,
        train_prediction
    )

    test_r2 = r2_score(
        y_test,
        test_prediction
    )

    test_mae = mean_absolute_error(
        y_test,
        test_prediction
    )

    st.subheader(
        "📈 Model Performance"
    )

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Training R² Score",
            f"{train_r2:.4f}"
        )

    with m2:

        st.metric(
            "Testing R² Score",
            f"{test_r2:.4f}"
        )

    with m3:

        st.metric(
            "Testing MAE",
            f"{test_mae:.4f}"
        )

    st.divider()




    # =====================================================
    # ACTUAL VS PREDICTED
    # =====================================================

    st.subheader(
        "🎯 Actual vs Predicted Risk"
    )

    comparison_df = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Predicted": test_prediction
        }
    )



    #
    # =====================================================
    # REAL-TIME PREDICTION
    # =====================================================

    st.subheader(
        "🔮 Real-Time Accident Risk Prediction"
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        input_hour = st.number_input(
            "Hour",
            0,
            23,
            12,
            key="input_hour"
        )

        input_temperature = st.number_input(
            "Temperature",
            value=30.0,
            key="input_temperature"
        )

    with p2:

        input_vehicles = st.number_input(
            "Vehicles Involved",
            min_value=1,
            value=2,
            key="input_vehicles"
        )

        input_casualties = st.number_input(
            "Casualties",
            min_value=0,
            value=1,
            key="input_casualties"
        )

    with p3:

        input_weekend = st.selectbox(
            "Is Weekend",
            [0, 1],
            key="input_weekend"
        )

        input_peak = st.selectbox(
            "Is Peak Hour",
            [0, 1],
            key="input_peak"
        )

    if st.button(
        "🔮 Predict Accident Risk",
        key="predict_accident_risk"
    ):

        input_data = pd.DataFrame(
            [
                [
                    input_hour,
                    input_temperature,
                    input_vehicles,
                    input_casualties,
                    input_weekend,
                    input_peak
                ]
            ],
            columns=feature_columns
        )

        prediction = float(
            model.predict(
                input_data
            )[0]
        )

        prediction = max(
            0.0,
            min(
                1.0,
                prediction
            )
        )

        if prediction >= 0.70:

            risk_label = "HIGH"

        elif prediction >= 0.40:

            risk_label = "MEDIUM"

        else:

            risk_label = "LOW"

        st.divider()

        r1, r2 = st.columns(2)

        with r1:

            st.metric(
                "Predicted Risk Score",
                f"{prediction:.2f}"
            )

        with r2:

            st.metric(
                "Risk Level",
                risk_label
            )

        if risk_label == "HIGH":

            st.error(
                "🔴 High accident risk"
            )

        elif risk_label == "MEDIUM":

            st.warning(
                "🟠 Medium accident risk"
            )

        else:

            st.success(
                "🟢 Low accident risk"
            )


# =========================================================
# END
# =========================================================