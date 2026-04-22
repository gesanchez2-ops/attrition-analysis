import pandas as pd
import pytest
from metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    """Four employees: Sales all leave, HR none leave."""
    return pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "department": ["Sales", "Sales", "HR", "HR"],
        "overtime": ["Yes", "Yes", "No", "No"],
        "job_satisfaction": [1, 2, 3, 4],
        "monthly_income": [3000, 5000, 7000, 9000],
        "attrition": ["Yes", "Yes", "No", "No"],
    })


# --- attrition_rate ---

def test_attrition_rate_fifty_percent(sample_df):
    assert attrition_rate(sample_df) == 50.0


def test_attrition_rate_all_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


def test_attrition_rate_no_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


# --- attrition_by_department ---

def test_attrition_by_department_columns(sample_df):
    result = attrition_by_department(sample_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_rates(sample_df):
    result = attrition_by_department(sample_df)
    sales = result[result["department"] == "Sales"].iloc[0]
    hr = result[result["department"] == "HR"].iloc[0]
    assert sales["attrition_rate"] == 100.0
    assert hr["attrition_rate"] == 0.0


def test_attrition_by_department_sorted_descending(sample_df):
    result = attrition_by_department(sample_df)
    rates = result["attrition_rate"].tolist()
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_rates(sample_df):
    result = attrition_by_overtime(sample_df)
    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    no_row = result[result["overtime"] == "No"].iloc[0]
    assert yes_row["attrition_rate"] == 100.0
    assert no_row["attrition_rate"] == 0.0


# --- average_income_by_attrition ---

def test_average_income_by_attrition_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values(sample_df):
    result = average_income_by_attrition(sample_df)
    leavers = result[result["attrition"] == "Yes"].iloc[0]
    stayers = result[result["attrition"] == "No"].iloc[0]
    assert leavers["avg_monthly_income"] == 4000.0   # (3000 + 5000) / 2
    assert stayers["avg_monthly_income"] == 8000.0   # (7000 + 9000) / 2


# --- satisfaction_summary ---

def test_satisfaction_summary_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_rates(sample_df):
    result = satisfaction_summary(sample_df).reset_index(drop=True)
    assert result.loc[0, "attrition_rate"] == 100.0  # satisfaction 1: 1 leaver / 1 employee
    assert result.loc[1, "attrition_rate"] == 100.0  # satisfaction 2: 1 leaver / 1 employee
    assert result.loc[2, "attrition_rate"] == 0.0    # satisfaction 3: 0 leavers / 1 employee
    assert result.loc[3, "attrition_rate"] == 0.0    # satisfaction 4: 0 leavers / 1 employee


def test_satisfaction_summary_sorted_ascending(sample_df):
    result = satisfaction_summary(sample_df)
    levels = result["job_satisfaction"].tolist()
    assert levels == sorted(levels)
