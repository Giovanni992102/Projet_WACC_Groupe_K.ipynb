import os
import pandas as pd
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from matplotlib import ticker
import statsmodels.api as sm
import seaborn as sns
import numpy as np

# definition des variables
path_local_data = 'combined_data.csv'

# Define the start and end dates
start_date = '2017-10-18'
end_date = '2023-10-18'

# Définition de la liste des indices
tickers = {
    'STOXX': '^STOXX',
    'NASDAQ': '^IXIC',
    'SP500': '^GSPC',
    'FTSE': '^FTSE',
    'CAC40': '^FCHI',
    'FTSEMIB': 'FTSEMIB.MI',
    'MSCI': 'MSCI',
    'MSCI ACWI': 'ACWI',
}

# Titre à étudier: amplifon
titre_etudie = {
    'AMPLIFON': 'AMP.MI',
}

# Indice de référence
indice_ref = 'MSCI'

# Taux sans risque
rf_rate_date = '01/12/2024'

tickers_rates_us = {'1m': 'tmubmusd01m',
                    '3m': 'tmubmusd03m',
                    '6m': 'tmubmusd06m',
                    '1y': 'tmubmusd01y',
                    '2y': 'tmubmusd02y',
                    '3y': 'tmubmusd03y',
                    '5y': 'tmubmusd05y',
                    '7y': 'tmubmusd07y',
                    '10y': 'tmubmusd10y',
                    '30y': 'tmubmusd30y'
                    }

tickers_rates_fr = {'1m': 'tmbmbfr-01m',
                    '3m': 'tmbmbfr-03m',
                    '6m': 'tmbmbfr-06m',
                    '1y': 'tmbmbfr-01y',
                    '2y': 'tmbmkfr-02y',
                    '3y': 'tmbmkfr-03y',
                    '4y': 'tmbmkfr-04y',
                    '5y': 'tmbmkfr-05y',
                    '6y': 'tmbmkfr-06y',
                    '7y': 'tmbmkfr-07y',
                    '10y': 'tmbmkfr-10y',
                    '15y': 'tmbmkfr-15y',
                    '20y': 'tmbmkfr-20y',
                    '25y': 'tmbmkfr-25y',
                    '30y': 'tmbmkfr-30y'
                    }
# telechargement des donnees
# Information sur le bond du titre à étudier
face_value = 100
coupon_rate = 0.01125
current_market_price = 92.125
remaining_years_to_maturity = 3


def download_yf_data(tickers, start_date, end_date):
    data = {}
    for index, ticker in tickers.items():
        data[index] = yf.download(ticker, start=start_date, end=end_date)['Close']

    return data


def save_df_csv(df, file_name):
    # Define the file path and name
    file_path = os.path.join(os.getcwd(), file_name)

    # Save the DataFrame as a CSV file
    df.to_csv(file_path, index=True)


raw_data = download_yf_data(tickers, start_date, end_date)

raw_titre_etudie = download_yf_data(titre_etudie, start_date, end_date)

ref_index_df = pd.DataFrame(raw_data)
ref_index_df.info()

ref_index_df.head()

# Nettoyage des données

combined_data = pd.concat([ref_index_df, pd.DataFrame(raw_titre_etudie)], axis=1)
combined_data.info()
combined_data.head()

# Clean the data
clean_combined_data = combined_data.dropna()
clean_combined_data.info()

save_df_csv(clean_combined_data, path_local_data)


# Téléchargement des données locales

def load_data(file_name):
    try:
        # Define the file path and name
        file_path = os.path.join(os.getcwd(), file_name)

        # Load the DataFrame from a CSV file
        df = pd.read_csv(file_path, index_col=0)

        return df
    except FileNotFoundError:
        print('The file does not exist')
        data = download_yf_data(tickers, start_date, end_date)
        ref_index_df = pd.DataFrame(data)
        ref_index_df_cleaned = ref_index_df.dropna()
        save_df_csv(ref_index_df_cleaned, file_name)

        return ref_index_df_cleaned


data = load_data(path_local_data)
data

# Etude de distribution des données


# Set the figure size
plt.figure(figsize=(20, 10))

# Create the line plot
sns.lineplot(data=data)

# Set labels and title
plt.xticks([])
plt.xlabel('Date', fontsize=16)

plt.ylabel('Value')

plt.title(f'Time Series of the different index from {start_date} to {end_date}', fontsize=20)

# Show the plot
plt.show()

from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

# Normalize each column of the DataFrame
normalized_df = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)

# Plot the normalized values
normalized_df.plot(figsize=(10, 6))

# Set labels and title
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.title('Normalized Values of Columns')

# Show the plot
plt.show()

sns.violinplot(data=data.pct_change(), inner='quartile', orient='h', scale='width', palette='Set3')
plt.xlabel('Log Return distribution')
plt.ylabel('Index')
plt.title('Violin Plot of Data')
plt.xscale('log')
plt.show()

sns.pairplot(data=data.pct_change(), diag_kind='kde', plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, height=4)

# To fill in with the stock you want to analyze
studied_company = 'AMPLIFON'
relevant_market = 'MSCI'


def build_title_info(titre, window=30):
    try:
        data_titre = pd.DataFrame(data[titre])
    except:
        raise KeyError(f"The titre {titre} is not present in the data")

    data_titre.columns = [f'{titre}_close']
    data_titre[f'{titre}_pct change'] = data_titre.pct_change()
    data_titre[f'{titre}_std'] = data_titre[f'{titre}_close'].rolling(window).std()
    data_titre[f'{titre}_sma'] = data_titre[f'{titre}_close'].rolling(window).mean()

    return data_titre


def create_study(df1, df2):
    df = pd.concat([df1, df2], axis=1)
    df.dropna(inplace=True)
    return df


company_to_study = build_title_info(studied_company)
market_info = build_title_info(relevant_market)

study = create_study(company_to_study, market_info)

study.head()

company_to_study.dropna(inplace=True)
company_to_study.head()

market_info.dropna(inplace=True)
market_info.head()


###Calcul du Cout des capitaux propres par la méthode du CAPM

# Calculation of Beta
def compute_beta(df, titre, index_ref, beta_window=160):
    titre_ret = df[titre].iloc[-beta_window:].pct_change().dropna()
    index_ret = df[index_ref].iloc[-beta_window:].pct_change().dropna()
    expected_market_return = index_ret.mean()
    annualized_expected_market_return = (1 + expected_market_return) ** 365 - 1

    x = index_ret
    y = titre_ret
    covariance = y.cov(x)
    market_variance = x.var()

    beta = covariance / market_variance
    result_tuple = (beta, expected_market_return, annualized_expected_market_return)
    return result_tuple


result = compute_beta(data, 'AMPLIFON', 'MSCI ACWI')
beta_value, expected_return, annualized_expected_return = result

print(
    f"""The {studied_company.lower()}'s beta has a value of {result[0]:.02f}
The {relevant_market.lower()}'s annualized market return is {result[2] * 100:.02f} %
With a daily market return of {result[1] * 10000:.02f} bp
"""
)


# Download of risk free curve
def get_days(bucket):
    if bucket[-1].lower() == 'm':
        base = 30
    elif bucket[-1].lower() == 'y':
        base = 360
    else:
        raise ValueError(f'wrong bucket format| bucket={bucket}')
    return int(bucket[: -1]) * base


def get_marketwatch_curve(date, tickers):
    url = r'https://www.marketwatch.com/investing/bond/{ticker}/downloaddatapartial?startdate={startdate}%2000:00:00&enddate={enddate}%2000:00:00&daterange={daterange}&frequency=p1d&csvdownload=true&downloadpartial=false&newdates=false&countrycode=bx'

    daterange = 'd30'

    dfs = []
    i = 0
    for bucket, ticker in tickers.items():
        i += 1
        url_ = url.format(ticker=ticker, startdate=date, enddate=date, daterange=daterange)
        df = pd.read_csv(url_)
        df['bucket'] = bucket
        df['days'] = get_days(bucket)
        dfs.append(df)

    result = pd.concat(dfs)
    result = result[['Date', 'bucket', 'days', 'Close']]
    result['Close'] = result['Close'].str[: -1].astype(float) / 100

    return result


rates_fr = get_marketwatch_curve(rf_rate_date, tickers_rates_fr)
rates_fr

french_risk_free_rates = pd.DataFrame(rates_fr)
french_rf_10 = french_risk_free_rates[french_risk_free_rates['bucket'] == '10y']
french_rf_10_close_value = french_rf_10['Close'].values[0]
print(f"The french 10 year rate is currently at {french_rf_10_close_value * 100:.02f} %")

# Calculation of cost of equity using CAPM Formula
cost_of_equity = french_rf_10_close_value + beta_value * (annualized_expected_return - french_rf_10_close_value)
print(f"The cost of equity for {studied_company.lower()} is {cost_of_equity * 100:0.2f} %")


### Cost of debt calculation

# Calculation of yield to maturity (YTM)
def calculate_ytm(face_value, coupon_rate, current_market_price, remaining_years_to_maturity):
    try:

        ytm = (coupon_rate * face_value + (face_value - current_market_price) / remaining_years_to_maturity) / (
                (face_value + current_market_price) / 2)

        return ytm
    except ZeroDivisionError:
        raise ZeroDivisionError(
            f"Unable to calculate the Cost of Debt. The remaining_years_to_maturity must not be equal to 0")


cost_of_debt = calculate_ytm(face_value, coupon_rate, current_market_price, remaining_years_to_maturity)

print(f"Estimated Cost of Debt: {cost_of_debt * 100:.2f} %")


###WACC Calculation
# income tax calculation
def compute_tax_is(ticker):
    net_income_data = yf.Ticker(ticker).financials.loc['Net Income', :]
    pretax_income_data = yf.Ticker(ticker).financials.loc['Pretax Income', :]

    # Extract Net Income and Pretax Income values for the most recent year
    net_income = net_income_data.iloc[0]
    pretax_income = pretax_income_data.iloc[0]

    t_is = 1 - (net_income / pretax_income)

    return t_is


for company, ticker in titre_etudie.items():
    tax = compute_tax_is(ticker)
    print(f"{company} income tax: {tax * 100:0.2f} %")


def get_financials_data(ticker):
    stock = yf.Ticker(ticker)

    info = stock.info

    balance_sheet = stock.balance_sheet.T

    # Extract market cap and total debt values
    market_cap = info.get("marketCap", "Not Available")
    total_debt = balance_sheet.get("Total Debt", None)

    financials_tuple = (market_cap, total_debt)
    return financials_tuple


for company, ticker in titre_etudie.items():
    market_cap, total_debt = get_financials_data(ticker)

    print(f"{company} Market Cap: {market_cap / 1000000:0.2f} M €")
    print(f"{company} Total Debt: {total_debt.values[0] / 1000000:0.2f} M €")

WACC = market_cap / (market_cap + total_debt.iloc[0]) * cost_of_equity + total_debt.iloc[0] / (
        market_cap + total_debt.iloc[0]) * cost_of_debt / 100 * (1 - tax)
print(f"{company} WACC: {WACC * 100:0.2f} %")
