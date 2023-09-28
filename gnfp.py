import io
import pandas as pd
import requests
import re
import streamlit as st


@st.cache_data
def get_betting_data():

    # get conference affiliation from local CSV
    p5_df = pd.read_csv('p5_affliation.csv')

    # Scrape the 'last updated' date from thepredictiontracker.com
    url = "https://www.thepredictiontracker.com/predncaa.html"

    # Send a GET request to the URL to fetch the webpage content
    response = requests.get(url)

    # pattern to find the latest updated date
    pattern = r"Updated: (.*?):"

    # get the returned text that matches
    match = re.search(pattern, response.text)

    # just grab the text that includes the day & date
    last_updated = match.group(1)

    # Remove the final 3 characters
    last_updated = last_updated[:-3]

    session = requests.Session()
    response = session.get(
        'https://www.thepredictiontracker.com/ncaapredictions.csv')

    # Create a file-like object from the bytes object
    csv_file = io.StringIO(response.content.decode())

    # Read the CSV file into a Pandas dataframe
    df = pd.read_csv(csv_file)

    # keep the following columns
    df = df[[
        'home',
        'road',
        'lineopen',
        'line',
        'linemidweek',
        'lineavg',
        'linestd',
        'linemedian',
        'phcover',
        'phwin'
    ]]

    # Replace "St." with "State" in 'home' and 'road' columns
    df['home'] = df['home'].str.replace("St.", "State")
    df['road'] = df['road'].str.replace("St.", "State")

    # Replace "Va." with "Virginia" in 'home' and 'road' columns
    df['home'] = df['home'].str.replace("Va.", "Virginia")
    df['road'] = df['road'].str.replace("Va.", "Virginia")

    # Replace "Mississippi" with "Ole Miss", which requires a bit more work
    def replace_mississippi(value):
        if value == "Mississippi":
            return "Ole Miss"
        else:
            return value

    # Apply the custom function to the 'home' and 'road' columns
    df['home'] = df['home'].apply(replace_mississippi)
    df['road'] = df['road'].apply(replace_mississippi)

    # Looking at the 'home' or 'road' columns
    filtered_df = df[df['home'].isin(
        p5_df['School']) | df['road'].isin(p5_df['School'])]

    # Reset the index if needed
    filtered_df.reset_index(drop=True, inplace=True)

    # Merge 'p5_df' into 'df' based on the "School" and "home" columns
    merged_df = filtered_df.merge(
        p5_df[['School', 'Conference']], left_on='home', right_on='School', how='left')

    # Rename the "Conference" column for home teams
    merged_df = merged_df.rename(columns={'Conference': 'homeTeam_conference'})

    # Merge 'p5_df' into 'df' again for the "road" column
    merged_df = merged_df.merge(
        p5_df[['School', 'Conference']], left_on='road', right_on='School', how='left')

    # Rename the "Conference" column for road teams
    merged_df = merged_df.rename(columns={'Conference': 'roadTeam_conference'})

    # Drop the extra "School" columns
    merged_df = merged_df.drop(columns=['School_x', 'School_y'])

    # fill in the teams that aren't in the Power 5
    merged_df['homeTeam_conference'] = merged_df['homeTeam_conference'].fillna(
        'Non-P5')
    merged_df['roadTeam_conference'] = merged_df['roadTeam_conference'].fillna(
        'Non-P5')

    # if not mid-week line has been released, just fill this in
    merged_df['linemidweek'] = merged_df['linemidweek'].fillna('no line yet')

    # create a column to show movement in the lines
    merged_df['line_movement'] = abs(merged_df['line'] - merged_df['lineopen'])

    # create matchup column
    merged_df['matchup'] = merged_df['road'] + ' @ ' + merged_df['home']

    # function to create string of betting favorite at OPENING line
    def opening_line(row):
        if row['lineopen'] < 0:
            return f"{row['road']} {row['lineopen']}"
        else:
            return f"{row['home']} {-row['lineopen']}"

    # Apply the custom function to create the new column
    merged_df['opening_line'] = merged_df.apply(opening_line, axis=1)

    # function to create string of betting favorite at CURRENT line
    def current_line(row):
        if row['line'] < 0:
            return f"{row['road']} {row['line']}"
        else:
            return f"{row['home']} {-row['line']}"

    # Apply the custom function to create the new column
    merged_df['current_line'] = merged_df.apply(current_line, axis=1)

    # function to show the average prediction outcome
    def average_prediction_outcome(row):
        if row['lineavg'] < 0:
            return f"{row['road']} by {-row['lineavg']:.1f}"
        else:
            return f"{row['home']} by {row['lineavg']:.1f}"

    # Apply the custom function to create the new column
    merged_df['avg_predicted_winner'] = merged_df.apply(
        average_prediction_outcome, axis=1)

    # drop unneeded columns
    merged_df = merged_df.drop([
        'lineopen',
        'line',
        'lineavg',
        'linemedian'
    ], axis=1)

    # rename
    merged_df = merged_df.rename(columns={
        'linestd': 'prediction_st_dev',
        'phcover': 'prob_homeTeam_covers',
        'phwin': 'prob_homeTeam_wins',
        'linemidweek': 'midweek_line'
    })

    merged_df['prob_roadTeam_covers'] = 1 - merged_df['prob_homeTeam_covers']

    # rearrange columns
    merged_df = merged_df[[
        'home',
        'road',
        'matchup',
        'homeTeam_conference',
        'roadTeam_conference',
        'opening_line',
        'current_line',
        'line_movement',
        'midweek_line',
        'prob_homeTeam_covers',
        'prob_roadTeam_covers',
        'avg_predicted_winner',
        'prediction_st_dev',
        'prob_homeTeam_wins'
    ]]

    merged_df['brinks_number'] = merged_df.apply(lambda row: max(
        row['prob_homeTeam_covers'], row['prob_roadTeam_covers']), axis=1)

    # Create a new 'brinks_label' column based on the greater value
    merged_df['brinks_label'] = merged_df.apply(
        lambda row: f"{row['home']} covers" if row['prob_homeTeam_covers'] > row['prob_roadTeam_covers'] else f"{row['road']} covers",
        axis=1
    )

    return merged_df


st.dataframe(get_betting_data(), use_container_width=True)
