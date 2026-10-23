

import streamlit as st
import pandas as pd
import requests
from datetime import date

API_URL = "http://127.0.0.1:8010"

st.set_page_config(
    page_title="Daily Demand Forecasting",
    page_icon="📈",
    layout="wide",
)

st.title("Daily Demand Forecasting Orders")
st.write(
    "Enter one set of order inputs to generate predictions for "
    "the next 7 working days."
)

try:
    response = requests.get(f"{API_URL}/", timeout=5)
    response.raise_for_status()
except requests.RequestException:
    st.error(
        "API server is not running. Start FastAPI on port 8010, "
        "then refresh this page."
    )
    st.stop()


with st.form("forecast_form"):
    st.subheader("Starting Date")
    start_date = st.date_input(
        "Select the first date to forecast from",
        value=date.today(),
    )

    st.subheader("Enter order inputs once")
    st.caption(
        "These same order-related values will be used for all 7 dates. "
        "Only the date-based week/day features will change."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        non_urgent = st.number_input(
            "Non-urgent orders",
            min_value=0.0,
            value=316.307,
        )
        urgent = st.number_input(
            "Urgent orders",
            min_value=0.0,
            value=223.27,
        )
        fiscal = st.number_input(
            "Fiscal sector orders",
            min_value=0.0,
            value=0.0,
        )

    with col2:
        traffic = st.number_input(
            "Traffic controller sector orders",
            min_value=0.0,
            value=65556.0,
        )
        banking1 = st.number_input(
            "Banking orders (1)",
            min_value=0.0,
            value=44914.0,
        )

    with col3:
        banking2 = st.number_input(
            "Banking orders (2)",
            min_value=0.0,
            value=188411.0,
        )
        banking3 = st.number_input(
            "Banking orders (3)",
            min_value=0.0,
            value=14793.0,
        )

    submit = st.form_submit_button("Predict Next 7 Working Days")


if submit:
    payload = {
        "start_date": start_date.isoformat(),
        "inputs": {
            "non_urgent_order": non_urgent,
            "urgent_order": urgent,
            "fiscal_sector_orders": fiscal,
            "traffic_controller_orders": traffic,
            "banking_orders_1": banking1,
            "banking_orders_2": banking2,
            "banking_orders_3": banking3,
        },
    }

    try:
        response = requests.post(
            f"{API_URL}/forecast/7-days",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        df = pd.DataFrame(result["forecast"])

        st.subheader("Next 7 Working Days — Predictions")
        st.dataframe(df, use_container_width=True, hide_index=True)

        chart_df = df.set_index("date")[["predicted_total_orders"]]
        st.line_chart(chart_df)

        if any(
            row["predicted_total_orders"] < 0
            for row in result["forecast"]
        ):
            st.warning(
                "Some estimates are negative. Review the inputs and "
                "model suitability; the estimates have not been forced to zero."
            )

        st.info(result["message"])

    except requests.HTTPError as exc:
        try:
            detail = exc.response.json()
        except ValueError:
            detail = exc.response.text
        st.error(f"API error: {detail}")

    except requests.RequestException as exc:
        st.error(f"Could not contact the API: {exc}")

    except (ValueError, KeyError) as exc:
        st.error(f"Could not process the forecast response: {exc}")
