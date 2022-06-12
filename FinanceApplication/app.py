import functools
from pathlib import Path
from data import News, Stock,ProfitableSort, Options
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.shared import JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
import plotly.express as px
from PIL import Image


uploaded_data = open("stocks.csv", "r")
df = pd.read_csv(uploaded_data)
stockList = list(df['Symbol'].unique())
chart = functools.partial(st.plotly_chart, use_container_width=True)
COMMON_ARGS = {
    "color": "symbol",
    "color_discrete_sequence": px.colors.sequential.Greens,
    "hover_data": [
        "account_name",
        "percent_of_account",
        "quantity",
        "total_gain_loss_dollar",
        "total_gain_loss_percent",
    ],
}


@st.experimental_memo
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take Raw Fidelity Dataframe and return usable dataframe.
    - snake_case headers
    - Include 401k by filling na type
    - Drop Cash accounts and misc text
    - Clean $ and % signs from values and convert to floats

    Args:
        df (pd.DataFrame): Raw fidelity csv data

    Returns:
        pd.DataFrame: cleaned dataframe with features above
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_", regex=False).str.replace("/", "_", regex=False)

    df.type = df.type.fillna("unknown")
    df = df.dropna()

    price_index = df.columns.get_loc("last_price")
    cost_basis_index = df.columns.get_loc("cost_basis_per_share")
    df[df.columns[price_index : cost_basis_index + 1]] = df[
        df.columns[price_index : cost_basis_index + 1]
    ].transform(lambda s: s.str.replace("$", "", regex=False).str.replace("%", "", regex=False).astype(float))

    quantity_index = df.columns.get_loc("quantity")
    most_relevant_columns = df.columns[quantity_index : cost_basis_index + 1]
    first_columns = df.columns[0:quantity_index]
    last_columns = df.columns[cost_basis_index + 1 :]
    df = df[[*most_relevant_columns, *first_columns, *last_columns]]
    return df


@st.experimental_memo
def filter_data(
    df: pd.DataFrame, account_selections: list[str], symbol_selections: list[str]
) -> pd.DataFrame:
    """
    Returns Dataframe with only accounts and symbols selected

    Args:
        df (pd.DataFrame): clean fidelity csv data, including account_name and symbol columns
        account_selections (list[str]): list of account names to include
        symbol_selections (list[str]): list of symbols to include

    Returns:
        pd.DataFrame: data only for the given accounts and symbols
    """
    df = df.copy()
    df = df[
        df['Sector'].isin(account_selections) & df['Symbol'].isin(symbol_selections) 
    ]

    return df


def main() -> None:
    st.title("The Shyam-Helper-Tool")

    with st.expander("OVERVIEW"):
        st.write(Path("README.md").read_text())

#    st.subheader("Upload your CSV from Fidelity")
#    uploaded_data = st.file_uploader(
#        "Drag and Drop or Click to Upload", type=".csv", accept_multiple_files=False
#    )

#    if uploaded_data is None:
#        st.info("Using example data. Upload a file above to use your own data!")
#        uploaded_data = open("example.csv", "r")
#    else:
#        st.success("Uploaded your file!")
    uploaded_data = open("Top50.csv", "r")
    df = pd.read_csv(uploaded_data)
    #with st.expander("Raw Dataframe"):
    #    st.write(df)

    #df = clean_data(df)
    #with st.expander("Cleaned Data"):
    #    st.write(df)
    symbols = list(df['Symbol'].unique())
    if st.sidebar.button("Click this button if you want the most recent stock market data."):
        st.sidebar.write("It takes approximately 5 minutes to get all of the most recent data.")
        st.sidebar.write("Thank you for your patience.")
        stockList = []
        uploaded_data = open("stocks.csv", "r")
        df = pd.read_csv(uploaded_data)
        for x in range(len(df['Symbol'])):
            try:
                stockList.append(Stock(df['Symbol'][x]))
            except:
                pass
        df = ProfitableSort(stockList,df).returnDf()    


    st.sidebar.subheader("Filter Displayed Accounts")

    sectors = list(df['Sector'].unique())
    sector_selections = st.sidebar.multiselect(
        "Select Sectors to View", options=sectors, default=sectors
    )
    st.sidebar.subheader("Filter Displayed Tickers")

    
    name_selections = list(df['Name'].unique())
    symbols = list(df.loc[df['Sector'].isin(sector_selections), "Symbol"].unique())
    symbol_selections = st.sidebar.multiselect(
        "Select Ticker Symbols to View", options=symbols, default=symbols
    )

   

    array = df['Symbol']
    df = filter_data(df, sector_selections, symbol_selections)
    st.subheader("Ticker Data")
    cellsytle_jscode = JsCode(
        """
    function(params) {
        if (params.value > 0) {
            return {
                'color': 'white',
                'backgroundColor': 'forestgreen'
            }
        } else if (params.value < 0) {
            return {
                'color': 'white',
                'backgroundColor': 'crimson'
            }
        } else {
            return {
                'color': 'white',
                'backgroundColor': 'slategray'
            }
        }
    };
    """
    )

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_columns(
        (
            "1 Day %Change",
            "5 Day %Change",
            "1 Month %Change",
            "6 Month %Change",
            "1 Year %Change",
            "5 Year %Change",
        ),
        cellStyle=cellsytle_jscode,
    )
    gb.configure_pagination()
    gb.configure_columns(("Sector", "Symbol"), pinned=True)
    gridOptions = gb.build()

    AgGrid(df, gridOptions=gridOptions, allow_unsafe_jscode=True)
    st.header('Black-Scholes Model Formula')
    image = Image.open('blackScholes.png')
    st.image(image)
    st.write("S0 = Stock Price")
    st.write("K = Strike Price")
    st.write("r = Risk-Free Rate")
    st.write("T = Time to Expiration")
    st.write("σ = Volatility")
    image2 = Image.open('normcdf.png')
    st.image(image2,caption = 'Normal Cumulative Distribution Function')
    
    try:
        st.header('Company Info')
        tick = st.text_input("Please Enter the Ticker of the Company you are seeking more information about. (ALL CAPS)")
        specific = Stock(tick)
        stockNews = News(tick)
        st.markdown(
                f"""
                * Ticker Symbol : {specific.getTicker()}
                * {specific.isSP500()}    #Using Linear Search 
                * Current Price : ${specific.getCurrPrice()}
                * Company Info :  {specific.getCompanyInfo()}
                Per Yahoo Finance

                """
            )    
        st.subheader("Most Recent Articles")
        for i in range(3):
            st.markdown(
                f"""
                Article {i+1}
                * Title : {stockNews.getTitle(i)}
                * Publisher : {stockNews.getPublisher(i)}
                * Link : {stockNews.getLink(i)}
                """
            )
    except:
        pass
    try:
        #Black-Scholes Model for Options Pricing
        st.header('Black-Scholes Options Pricing Model')
        ticker = st.text_input("Enter the Ticker Symbol. (ALL CAPS)")
        strikePrice = float(st.number_input("Enter the Strike Price"))
        time = float(st.number_input("Enter the Time to Expiration in Years"))
        riskFreeRate = float(st.number_input("Enter the Risk Free Rate"))
        if(ticker != None and strikePrice != None and time != None and riskFreeRate != None):
            st.markdown(
                f"""
                * Ticker Symbol: {ticker}
                * Strike Price: {strikePrice}
                * Time : {time}
                * Risk Free Rate : {riskFreeRate}
                * Call Option Price : {Options(ticker, strikePrice, time, riskFreeRate).calculateIdealCall()}
                * Put Option Price : {Options(ticker, strikePrice, time, riskFreeRate).calculateIdealPut()}
                """
            )
    except:
        pass
    try:
        st.subheader('1st Degree Taylor Polynomial Call vs. Black-Scholes Model')
        st.pyplot(Options(ticker, strikePrice, time, riskFreeRate).makeTaylorApproximationTimeGreeks1st())
        st.write("X Axis: Time to Expiration(Years)")
        st.write("Y Axis: Call Option Price$")
        st.subheader('1st Degree Taylor Polynomial Put vs. Black-Scholes Model')
        st.pyplot(Options(ticker, strikePrice, time, riskFreeRate).taylorPolynomialPut1st())
        st.write("X Axis: Time to Expiration(Years)")
        st.write("Y Axis: Put Option Price$")
        st.subheader('2nd Degree Taylor Polynomial  Call vs. Black-Scholes Model')
        st.pyplot(Options(ticker, strikePrice, time, riskFreeRate).makeTaylorApproximationTimeGreeks2nd())
        st.write("X Axis: Time to Expiration(Years)")
        st.write("Y Axis: Call Option Price$")
        st.subheader('2nd Degree Taylor Polynomial Put vs. Black-Scholes Model')
        st.pyplot(Options(ticker, strikePrice, time, riskFreeRate).taylorPolynomialPut2nd())
        st.write("X Axis: Time to Expiration(Years)")
        st.write("Y Axis: Put Option Price$")
        st.subheader('4th Degree Taylor Polynomial Call vs. Black-Scholes Model')
        st.pyplot(Options(ticker, strikePrice, time, riskFreeRate).makeTaylorApproximationTimeGreeks4th())
        st.write("X Axis: Time to Expiration(Years)")
        st.write("Y Axis: Call Option Price$")
        st.subheader('4th Degree Taylor Polynomial Put vs. Black-Scholes Model')
        st.pyplot(Options(ticker, strikePrice, time, riskFreeRate).taylorPolynomialPut4th())
        st.write("X Axis: Time to Expiration(Years)")
        st.write("Y Axis: Put Option Price$")
        st.subheader('8th Degree Taylor Polynomial Call vs. Black-Scholes Model')
        st.pyplot(Options(ticker, strikePrice, time, riskFreeRate).makeTaylorApproximationTimeGreeks8th())
        st.write("X Axis: Time to Expiration(Years)")
        st.write("Y Axis: Call Option Price$")
        st.subheader('8th Degree Taylor Polynomial Put vs. Black-Scholes Model')
        st.pyplot(Options(ticker, strikePrice, time, riskFreeRate).taylorPolynomialPut8th())
        st.write("X Axis: Time to Expiration(Years)")
        st.write("Y Axis: Put Option Price$")
    except:
        pass
#    def draw_bar(y_val: str) -> None:
#        fig = px.bar(df, y=y_val, x="symbol", **COMMON_ARGS)
#        fig.update_layout(barmode="stack", xaxis={"categoryorder": "total descending"})
#        chart(fig)

#    account_plural = "s" if len(account_selections) > 1 else ""
#    st.subheader(f"Option Pricing Model{account_plural}")
#    totals = df.groupby("Sector", as_index=False).sum()
#    if len(account_selections) > 1:
#        st.metric(
#            "Call Option Price",
#            f"${totals.current_value.sum():.2f}",
#            f"{totals.total_gain_loss_dollar.sum():.2f}",
#        )
#    for column, row in zip(st.columns(len(totals)), totals.itertuples()):
#        column.metric(
#            row['Sector'],
#            f"${row.current_value:.2f}",
#            f"{row.total_gain_loss_dollar:.2f}",
#        )
#
#    fig = px.bar(
#       totals,
#        y="account_name",
#        x="current_value",
#        color="account_name",
#        color_discrete_sequence=px.colors.sequential.Greens,
#    )
#    fig.update_layout(barmode="stack", xaxis={"categoryorder": "total descending"})
#    chart(fig)

#    st.subheader("Value of each Symbol")
#    draw_bar("current_value")

#    st.subheader("Value of each Symbol per Account")
#    fig = px.sunburst(
#        df, path=["account_name", "symbol"], values="current_value", **COMMON_ARGS
#    )
#    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
#    chart(fig)

#    st.subheader("Value of each Symbol")
#    fig = px.pie(df, values="current_value", names="symbol", **COMMON_ARGS)
#    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
#    chart(fig)

#    st.subheader("Total Value gained each Symbol")
#    draw_bar("total_gain_loss_dollar")
#    st.subheader("Total Percent Value gained each Symbol")
#    draw_bar("total_gain_loss_percent")


if __name__ == "__main__":
    st.set_page_config(
        "Shyam's Finance Application",
        "📊",
        initial_sidebar_state="expanded",
        layout="wide",
    )
    main()
