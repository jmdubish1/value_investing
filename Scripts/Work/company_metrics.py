import random
import pandas as pd
import numpy as np
import os
import scipy.stats as sstats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math


class CompanyPortfolio:
    def __init__(self, symbol, data_loc):
        self.symbol = symbol
        self.data_loc = data_loc
        self.balancesheet = dict
        self.cashflow = dict
        self.earnings = dict
        self.income_statement = dict
        self.daily_adjusted = None

    def fill_balancesheet(self, columns=None):
        """Used to fill the balance sheet and created an easily manipulated dictionary
        :param columns: Default is all, or pass a list"""

        df = pd.read_csv(f'{self.data_loc}\\BALANCE_SHEET\\{self.symbol}.csv')
        df.drop([df.columns[0], 'reportedCurrency'], axis=1, inplace=True)
        df_dict = dict()

        if columns is None:
            working_columns = df.columns
        else:
            working_columns = columns + ['shortTermInvestments', 'longTermInvestments', 'investments']

        for c in working_columns:
            if not c == 'fiscalDateEnding':
                df_dict[c] = {'data': list, 'growth': float, 'mean': float, 'std': float, 'pdf_params': []}
                df_dict[c]['data'] = df[c]
                df_dict[c]['data'] = replace_none(df[c])
                df_dict[c]['growth'] = find_growth_rate(df_dict[c]['data'])

        self.balancesheet = df_dict
        self.fix_investments()
        self.fix_debt()

    def fill_cashflow(self, columns=None):
        """Used to fill the cashflow sheet and created an easily manipulated dictionary"""
        df = pd.read_csv(f'{self.data_loc}\\CASHFLOW\\{self.symbol}.csv')
        df.drop([df.columns[0], 'reportedCurrency'], axis=1, inplace=True)
        df_dict = dict()

        if columns is None:
            working_columns = df.columns
        else:
            working_columns = columns

        for c in working_columns:
            if not c == 'fiscalDateEnding':
                df_dict[c] = {'data': list, 'growth': float, 'mean': float, 'std': float, 'pdf_params': []}
                df_dict[c]['data'] = replace_none(df[c])
                df_dict[c]['growth'] = find_growth_rate(df_dict[c]['data'])

        self.cashflow = df_dict

    def fill_earnings(self, columns=None):
        """Used to fill the earnings sheet and created an easily manipulated dictionary"""
        df = pd.read_csv(f'{self.data_loc}\\EARNINGS\\{self.symbol}.csv')
        df.drop([df.columns[0], 'reportedCurrency'], axis=1, inplace=True)
        df_dict = dict()

        if columns is None:
            working_columns = df.columns
        else:
            working_columns = columns

        for c in working_columns:
            if not c == 'fiscalDateEnding':
                df_dict[c] = {'data': list, 'growth': float, 'mean': float, 'std': float, 'pdf_params': []}
                df_dict[c]['data'] = replace_none(df[c])
                df_dict[c]['growth'] = find_growth_rate(df_dict[c]['data'])

        self.earnings = df_dict

    def fill_incomestate(self, columns=None):
        """Used to fill the income statement and created an easily manipulated dictionary"""
        df = pd.read_csv(f'{self.data_loc}\\INCOME_STATEMENT\\{self.symbol}.csv')
        df.drop([df.columns[0], 'reportedCurrency'], axis=1, inplace=True)
        df_dict = dict()

        if columns is None:
            working_columns = df.columns
        else:
            working_columns = columns

        for c in working_columns:
            if not c == 'fiscalDateEnding':
                df_dict[c] = {'data': list, 'growth': float, 'mean': float, 'std': float, 'pdf_params': []}
                df_dict[c]['data'] = replace_none(df[c])
                df_dict[c]['growth'] = find_growth_rate(df_dict[c]['data'])

        self.income_statement = df_dict

    def fill_overview(self, columns=None):
        """Used to fill the company overview and created an easily manipulated dictionary"""
        df = pd.read_csv(f'{self.data_loc}\\OVERVIEW\\{self.symbol}.csv')
        df.drop([df.columns[0]], axis=1, inplace=True)
        df_dict = dict()

        if columns is None:
            working_columns = df.columns
        else:
            working_columns = columns

        for c in working_columns:
            df_dict[c] = df[c]

        self.income_statement = df_dict

    def fill_daily_adj(self):
        """Used to fill the daily adjusted price as a series"""
        df = pd.read_csv(f'{self.data_loc}\\TIME_SERIES_DAILY_ADJUSTED\\{self.symbol}.csv')
        df.drop([df.columns[0]], axis=1, inplace=True)
        df.rename({df.columns[0]: 'date'}, inplace=True)
        self.daily_adjusted = df[['date', '5. adjusted close']]

    def apply_dist_pdf_params(self, metric, columns='All'):
        """Saves pfd params to the class"""

        if columns == 'All':
            working_columns = metric
        else:
            working_columns = columns
        for m in working_columns:
            if not m == 'fiscalDateEnding':
                data = metric[m]['data']
                metric[m]['pdf_params'] = get_gammadist_params(data, 250)

    def fix_investments(self):
        for r in list(range(0, len(self.balancesheet['investments']['data']))):
            inv = self.balancesheet['investments']['data'].iloc[r]
            short_term = self.balancesheet['shortTermInvestments']['data'].iloc[r]
            long_term = self.balancesheet['longTermInvestments']['data'].iloc[r]

            if not inv == short_term + long_term:
                if short_term == 0:
                    self.balancesheet['shortTermInvestments']['data'].iloc[r] = inv - long_term
                    self.balancesheet['totalCurrentAssets']['data'].iloc[r] = \
                        self.balancesheet['shortTermInvestments']['data'].iloc[r]
                elif long_term == 0:
                    self.balancesheet['longTermInvestments']['data'].iloc[r] = inv - short_term
                    self.balancesheet['totalNonCurrentAssets']['data'].iloc[r] = \
                        self.balancesheet['longTermInvestments']['data'].iloc[r]
                elif inv != short_term + long_term:
                    self.balancesheet['investments']['data'].iloc[r] = long_term + short_term

    def fix_debt(self):
        for r in list(range(0, len(self.balancesheet['shortLongTermDebtTotal']['data']))):
            inv = self.balancesheet['shortLongTermDebtTotal']['data'].iloc[r]
            short_term = self.balancesheet['shortTermDebt']['data'].iloc[r]
            long_term = self.balancesheet['longTermDebt']['data'].iloc[r]
            if inv == 0:
                self.balancesheet['shortLongTermDebtTotal']['data'].iloc[r] = long_term + short_term
            elif short_term == 0:
                self.balancesheet['shortTermDebt']['data'].iloc[r] = inv - long_term
                self.balancesheet['totalCurrentLiabilities']['data'].iloc[r] = \
                    self.balancesheet['shortTermDebt']['data'].iloc[r]
            elif long_term == 0:
                self.balancesheet['longTermDebt']['data'].iloc[r] = inv - short_term
                self.balancesheet['totalNonCurrentLiabilities']['data'].iloc[r] = \
                    self.balancesheet['longTermDebt']['data'].iloc[r]


def replace_none(data):
    """Replaces none values with zero.
    :param data: series of floats or int with possible nan values"""

    for d in list(range(0, len(data.index))):
        if data.iloc[d] == 'None':
            data.iloc[d] = 0
    data = data.astype(float)

    return data


def find_percent_diff(data):
    """Find difference between previous period within the dataset
    :param data: the series/list that will be worked on

    :return List of the difference between the points, less on nan element"""

    return data[::-1].pct_change().iloc[1:]


def find_growth_rate(data):
    return np.mean(data[::-1].pct_change())


def estimate_beta(market_prices, stock_prices):
    """Converts daily changes to monthly, then finds the stock beta"""
    '''Maybe introduce some kind of price smoothing?'''
    stock_prices = stock_prices[::21]
    price_change = stock_prices.iloc[:60][::-1].pct_change() * 100
    price_change = price_change.drop(price_change.index[0])

    market_prices = market_prices[::21]
    market_change = market_prices.iloc[:60][::-1].pct_change() * 100
    market_change = market_change.drop(market_change.index[0])

    cov = np.cov([market_change.iloc[:len(price_change)], price_change])[0][1]

    return cov / np.var(market_change)


def earnings_surprise_avg(earnings_df, lookback):
    """Gives the average surprise for the last lookback period
    :param  earnings_df: full earnings df
    :param  lookback: int for number of quarters to average"""

    return np.mean(earnings_df['surprise_prc'].iloc[:lookback])


def bootstrap_mse(data, sample_size):
    """Used to bootstrap the data with a modified MSE that is the result of the sqrt of the MSE multiplied by some
    random value between 0 and 1 as well as a randomly selected positive or negative value.
    -Can be used to create a distribution that is statistically similar to the small original where data can be pulled
    from.
    :param data: the series or list that work will be done on
    :param sample_size: the number of elements you want in your output"""

    data = find_percent_diff(data)
    data = [d for d in data if not (math.isinf(d) or math.isnan(d))]
    reg_model = LinearRegression()
    try:
        x = np.linspace(min(data), max(data), len(data)).reshape(-1, 1)
        reg_model.fit(x, data)
        y_pred = reg_model.predict(x)
        rmse = mean_squared_error(data, y_pred)
    except ValueError as ve:
        print(f'{ve}.. Continuing')
        rmse = 0

    return [d + (random.randint(-1, 1) * random.random() * np.sqrt(rmse)) for d in data * sample_size]


def get_gammadist_params(data, sample_size):
    """get the parameters for a gamma distribution that can be saved in lieu of a dataset in order to save space.
    :param data: the series of list that work will be done on
    :param sample_size: the number of elements you want in your output

    :return List of params in the order 'distribution,' 'scale,' then 'location.'"""

    booted = bootstrap_mse(data, sample_size)
    dist = getattr(sstats, 'gamma')
    try:
        param = dist.fit(booted)
        param = [p for p in param]
    except ValueError as ve:
        print(f'{ve}... Continuing')
        param = [None, None, None]

    return param


def create_distribution_pdf(data, dist_size):
    """Gives paramters of a Gamma Distribution of the dataset that can be used to recreate an accurate sample.
    It's not necessary to use this right now.
    :param data: the series or list that work will be done on
    :param dist_size: how many data points the final dist will have

    :return List of datapoints of the recreated distribution and the wilcox p-value for comparing the data to the
    original"""

    booted = bootstrap_mse(data, dist_size)
    skew = sstats.skew(np.array(booted))
    left_skew = False

    if skew < -0.1:
        booted = [-d for d in booted]
        left_skew = True

    ae, loce, scalee = sstats.skewnorm.fit(booted)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    pdf = sstats.skewnorm.pdf(x, ae, loce, scalee)
    dist = random.choices(x, weights=pdf / sum(pdf), k=dist_size)
    _, wilcox = sstats.mannwhitneyu(data, dist, method='auto')

    if left_skew:
        dist = [-d for d in dist]

    return dist, wilcox
