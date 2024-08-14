import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Data Analysis",
    page_icon="🚚",
)

@st.cache_data  # 👈 Add the caching decorator
def load_data(url):
    df = pd.read_csv(url, low_memory=False)
    return df


st.markdown("## 🚚 OLIMP Data Analysis & Warehousing Price Prediction")


col1, col2 = st.columns([10, 3])

with col2:
    st.image('olimp.png')

df = load_data('OLIMP_clean_dataset.csv')

median_prices_by_year = df.groupby("Year")["Price"].median()
df["Price_Year"] = df["Year"].map(median_prices_by_year)

median_prices_by_location = df.groupby("Location")["Price"].median()
df["Price_Location"] = df["Location"].map(median_prices_by_location)

median_prices_by_month = df.groupby("Month")["Price"].median()
df["Price_Month"] = df["Month"].map(median_prices_by_month)

median_prices_by_hour = df.groupby("Hour")["Price"].median()
df["Price_Hour"] = df["Hour"].map(median_prices_by_hour)


# List of columns for selection
color_columns = ['Select', 'DateTime', 'Location', 'Price', 'Hour', 'Time_of_Day', 'Day', 'Day_W', 'Month', 'Year', 'Season', 'High_Qual']

left_column, right_column = st.columns(2)

with left_column:
    option = st.selectbox('Features1', df.columns, key="selectbox_1")

color_columns = ['Select'] + [col for col in df.columns if col != option]

with right_column:
    option2 = st.selectbox('Features2', color_columns, key="selectbox_2")


if option2 == 'Select':
    fig = px.histogram(df, x=option).update_xaxes(categoryorder="total descending")
else:
    fig = px.histogram(df, x=option, color=option2, barmode='group').update_xaxes(categoryorder="total descending")

fig.update_layout(width=700, height=500)
st.plotly_chart(fig)

left_column, right_column = st.columns(2)

with left_column:
    option = st.selectbox('Features1', df.columns, key="selectbox_3")

color_columns = df.columns.tolist()
if option in color_columns:
    color_columns.remove(option)


with right_column:
    option2 = st.selectbox('Features2', ['Select'] + color_columns, key="selectbox_4")


if option2 == 'Select':
    fig = px.histogram(df, x=option).update_xaxes(categoryorder="total descending")
else:
    df_sorted = df[[option, option2]].dropna().sort_values(by=option)

    histogram = go.Histogram(
        x=df_sorted[option],
        name=f'{option} Distribution',
        opacity=0.6,
        marker=dict(color='lightblue'),
        yaxis='y'
    )

    line_chart = go.Scatter(
        x=df_sorted[option],
        y=df_sorted[option2],
        mode='lines+markers',
        name=f'{option2} Trend',
        line=dict(color='coral', width=2),
        marker=dict(color='coral', size=8),
        yaxis='y2'
    )

    fig = go.Figure(data=[histogram, line_chart])

    # Layout updates
    fig.update_layout(
        title=f'{option} and {option2} Combined',
        title_x=0.5,
        xaxis=dict(title=option),
        yaxis=dict(title='Count', overlaying='y2', side='left'),
        yaxis2=dict(title=option2, side='right'),
        width=900,
        height=600
    )

st.plotly_chart(fig)

###############################################################
#Modelling
###############################################################
st.markdown("# 📦 Warehousing Price Prediction")

df_model = load_data('OLIMP_model_dataset.csv')
df_model = df_model.drop("Price", axis=1)
df_model.head()
########################

def label_encoding(df, target_column=None):
    target_mapping = None
    encode_mapping = {}

    if target_column and target_column in df.columns:
        target_mapping = {label: idx for idx, label in enumerate(df[target_column].unique())}
        df[target_column] = df[target_column].map(target_mapping)

    for col in df.columns:
        if col != target_column:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encode_mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    return df, target_mapping, encode_mapping


left_column, right_column = st.columns(2)
with left_column:
    Location = st.selectbox("Location", df_model['Location'].unique())

with right_column:
    Hour = st.slider("Hour", min_value=0, max_value=23, step=1)

with left_column:
    Time_of_Day = st.selectbox("Time_of_Day", df_model['Time_of_Day'].unique())

with right_column:
    Day = st.slider("Day", min_value=1, max_value=31, step=1)

with left_column:
    Day_W = st.selectbox("Day_W", df_model['Day_W'].unique())

with right_column:
    Month = st.slider("Month", min_value=1, max_value=12, step=1)

with left_column:
    Year = st.selectbox("Year", df_model["Year"].unique())
df_model["Year"].unique()
with right_column:
    Season = st.selectbox("Season", df_model['Season'].unique())

with left_column:
    High_Qual = st.selectbox("High_Qual", df_model['High_Qual'].unique())

with right_column:
    Position = st.selectbox("Position", df_model['Position'].unique())

with left_column:
    Seafront = st.selectbox("Seafront", df_model["Seafront"].unique())


df_model_50 = df_model.head(200)
df_model_50, target_mapping, encode = label_encoding(df_model_50)

input_list = [tuple([encode['Location'][Location],
                    encode['Hour'][Hour],
                    encode['Time_of_Day'][Time_of_Day],
                    encode['Day'][Day],
                    encode['Day_W'][Day_W],
                    encode['Month'][Month],
                    encode['Year'][Year],
                    encode['Season'][Season],
                    encode['High_Qual'][High_Qual],
                    encode['Position'][Position],
                    encode['Seafront'][Seafront]
                     ])]


with open('pricing_model.sav', 'rb') as f:
    model = pickle.load(f)


predict_df = pd.DataFrame(input_list, columns= model.feature_names_in_)

prediction = model.predict(predict_df)
st.success(f"###  👉 Price : {int(prediction)}")
