from datetime import datetime
import pandas as pd
import alphaVantageAPI as alpha
import alphav_tools as a_tools
import twelvedata


def main():
    api_key = '0EPJYHV4DHTGCRRA'
    data_loc = r'C:\Python\Python\Projects\Data\NYSE-NASDAQ'
    outputsize = 'full'
    listing_file = r'C:\Python\Python\Projects\Value_Investing\value_investments.csv'
    listings = pd.read_csv(listing_file)
    av = alpha.AlphaVantage(api_key, output_size=outputsize)

    '''Downloads SPX from TwelveData as it's not available on AlphaVantage'''
    a_tools.get_spx_daily(8, f'{data_loc}\\TIME_SERIES_DAILY_ADJUSTED')

    a_tools.get_data(av.balance, 'BALANCE_SHEET', data_loc, listings, av, 168)
    a_tools.get_data(av.income, 'INCOME_STATEMENT', data_loc, listings, av, 168)
    # a_tools.get_data(av.cashflow, 'CASHFLOW', data_loc, listings, av, 168)
    # a_tools.get_data(av.overview, 'OVERVIEW', data_loc, listings, av, 168)
    # a_tools.get_data(av.earnings, 'EARNINGS', data_loc, listings, av, 168)
    # a_tools.get_data(av.daily_adjusted, 'TIME_SERIES_DAILY_ADJUSTED', data_loc, listings, av, 168)

if __name__ == '__main__':
    main()

