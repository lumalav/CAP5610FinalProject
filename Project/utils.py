#drawing the court
from matplotlib.patches import Circle, Rectangle, Arc; import numpy as np; import matplotlib.pyplot as plt; import pandas as pd;

def engineer_features_and_write_file(data):
    offensive_game_scores = []
    for _, group in data.groupby('game_id'):
        points_attempted_2 = 0 
        points_made_2 = 0
        points_made_3 = 0
        points_attempted_3 = 0
        points = 0
        fga = 0
        fg = 0
        offensive_game_score = 0
        for _, group2 in group.groupby('game_event_id'):
            fga += 1
            if group2['shot_type'].iloc[0] == '2PT Field Goal':
                points_attempted_2 += 1
                if group2['shot_made_flag'].iloc[0] == 1:
                    points_made_2 +=1
                    fg += 1
                    points += 2
            else:
                points_attempted_3 += 1
                if group2['shot_made_flag'].iloc[0] == 1:
                    points_made_3 += 1
                    fg += 1
                    points += 3
            offensive_game_score = points + (fg) - (fga)
            if offensive_game_score < 0:
                offensive_game_score = 0
            offensive_game_scores.append(offensive_game_score)
    data['efficiency'] = offensive_game_scores
    data['efficiency_normalized'] = data['efficiency'] / data['efficiency'].max()
    data.game_date = pd.to_datetime(data.game_date)

    data['home_game'] = data.matchup.apply(lambda x: 1 if 'vs.' in x else 0)
    data['shot_zone_area2']=data['shot_zone_area'].astype('category').cat.codes
    # Create features for shot angle and distance from hoop
    data['angle'] = np.degrees(np.arctan(data['loc_y'] / data['loc_x']))
    data['distance'] = np.sqrt(np.power(data['loc_x'], 2) + np.power(data['loc_y'], 2))
    # Note that distance from hoop is based on the position values and so does not correspond to a unit
    data.to_csv('data_engineered.csv', index=False)

def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    """
    Draws the court and returns the ax object
    source: http://savvastjortjoglou.com/nba-shot-sharts.html
    """
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

def generate_correlations(data, test):
    """
    Generates the file with all the correlations between the original data and the test data 
    preserving the date to avoid data leakage
    """
    test.sort_values(by=['game_date'], inplace=True, ascending=True)

    shot_made_corr_df = pd.DataFrame()

    for test_idx in test.index:
        shot_made_corr_df = shot_made_corr_df.append(data[data['game_date'] < test['game_date'][test_idx]].corr()['shot_made_flag'], ignore_index=True)

    shot_made_corr_df.to_csv('shot_made_correlations.csv', index=False)

def draw_efficiency_pdf_cdf_until_date(data, test, date='01/01/2000'):
    """
    Draws the pdf and the cdf of the player's efficiency until that particular date
    """
    min_date = data['game_date'].min()
    max_date = data['game_date'].max()
    date = pd.to_datetime(date)

    if date < min_date:
        print('WARNING: the minimum game recorded is: ' + str(min_date))
        date = min_date
    if pd.to_datetime(date) > max_date:
        print('WARNING: the maximum game recorded is: ' + str(max_date))
        date = max_date

    subset = data.loc[data['game_date'] <= date][['efficiency', 'efficiency_normalized']]
    group = subset.groupby('efficiency')['efficiency'].agg('count').pipe(pd.DataFrame).rename(columns = {'efficiency': 'frequency'})

    # PDF
    group['pdf'] = group['frequency'] / sum(group['frequency'])

    # CDF
    group['cdf'] = group['pdf'].cumsum()
    group = group.reset_index()
    group.plot.bar(x = 'efficiency', y = ['pdf', 'cdf'], grid = True, title='PDF and CDF for Efficiency until ' + str(date.strftime('%Y-%m-%d')))

    #cdf using the normalized efficiency
    subset['cdf'] = subset[['efficiency_normalized']].rank(method = 'average', pct = True)
    # Sort and plot
    subset.sort_values('efficiency_normalized').plot(x = 'efficiency_normalized', y = 'cdf', grid = True, title='CDF for Efficiency until ' + str(date.strftime('%Y-%m-%d')))