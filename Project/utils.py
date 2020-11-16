# drawing the court
from matplotlib.patches import Circle, Rectangle, Arc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


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
                    points_made_2 += 1
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
    data['efficiency_normalized'] = data['efficiency'] / \
        data['efficiency'].max()
    data.game_date = pd.to_datetime(data.game_date)
    data['action_type2'] = data['action_type'].astype('category').cat.codes
    data['combined_shot_type2'] = data['combined_shot_type'].astype(
        'category').cat.codes
    data['opponent2'] = data['opponent'].astype('category').cat.codes

    data['home_game'] = data.matchup.apply(lambda x: 1 if 'vs.' in x else 0)
    data['shot_zone_area2'] = data['shot_zone_area'].astype(
        'category').cat.codes
    # Create features for shot angle and distance from hoop
    data['angle'] = np.degrees(
        np.arctan(np.nan_to_num(data['loc_y'] / data['loc_x'])))
    data['distance'] = np.sqrt(
        np.power(data['loc_x'], 2) + np.power(data['loc_y'], 2))
    # Note that distance from hoop is based on the position values and so does not correspond to a unit

    data['distance_traveled'] = data.apply(
        lambda row: __distance_from_LA(row), axis=1)

    data['right_of_net'] = data.loc_x.apply(lambda x: 1 if x <= 0 else 0)
    data.to_csv('data_engineered.csv', index=False)


def __distance_from_LA(row):
    lat = row['lat']
    lon = row['lon']
    home_lat = 34.00
    home_lon = -118.218

    radius = 6371

    dlat = math.radians(lat - home_lat)
    dlon = math.radians(lon - home_lon)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(home_lat)) * math.cos(math.radians(lat)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


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
        shot_made_corr_df = shot_made_corr_df.append(
            data[data['game_date'] < test['game_date'][test_idx]].corr()['shot_made_flag'], ignore_index=True)

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

    subset = data.loc[data['game_date'] <= date][[
        'efficiency', 'efficiency_normalized']]
    group = subset.groupby('efficiency')['efficiency'].agg('count').pipe(
        pd.DataFrame).rename(columns={'efficiency': 'frequency'})

    # PDF
    group['pdf'] = group['frequency'] / sum(group['frequency'])

    # CDF
    group['cdf'] = group['pdf'].cumsum()
    group = group.reset_index()
    group.plot.bar(x='efficiency', y=['pdf', 'cdf'], grid=True,
                   title='PDF and CDF for Efficiency until ' + str(date.strftime('%Y-%m-%d')))

    # cdf using the normalized efficiency
    subset['cdf'] = subset[['efficiency_normalized']].rank(
        method='average', pct=True)
    # Sort and plot
    subset.sort_values('efficiency_normalized').plot(x='efficiency_normalized', y='cdf',
                                                     grid=True, title='CDF for Efficiency until ' + str(date.strftime('%Y-%m-%d')))


def draw_exploratory_data_charts(data, test, chosen_features):
    # A
    groups = data[['action_type', 'shot_type', 'shot_made_flag']].groupby(
        ['action_type', 'shot_type'], as_index=False).mean()

    parameters = {
        '2PT Field Goal': {
            'figsize': {
                'y': 20
            }
        },
        '3PT Field Goal': {
            'figsize': {
                'y': 8
            }
        }
    }

    # this plot might get merged into a single one
    for n, g in groups.groupby('shot_type'):
        ax = g.sort_values(by='shot_made_flag', ascending=False).plot(kind='barh', x='action_type', y='shot_made_flag', figsize=(
            8, parameters[n]['figsize']['y']), color='#86bf91', zorder=2, width=0.85)
        # Set x-axis label
        ax.set_xlabel('Accuracy of ' + n + 's',
                      labelpad=20, weight='bold', size=12)
        # Set y-axis label
        ax.set_ylabel('Action Type', labelpad=20, weight='bold', size=12)

    # B
    # shots made vs shots failed
    parameters = [{
        'shot_made_flag': 1,
        'color': '#86bf91'
    }, {
        'shot_made_flag': 0,
        'color': '#f44336'
    }]

    for x in range(2):
        plt.figure(figsize=(12, 11))
        shots_made = data.loc[data['shot_made_flag']
                              == parameters[x]['shot_made_flag']]
        plt.scatter(shots_made.loc_x, shots_made.loc_y,
                    color=parameters[x]['color'])

        draw_court()
        # Adjust plot limits to just fit in half court
        plt.xlim(-250, 250)
        # Descending values along th y axis from bottom to top
        # in order to place the hoop by the top of plot
        plt.ylim(422.5, -47.5)
        # get rid of axis tick labels
        # plt.tick_params(labelbottom=False, labelleft=False)
        plt.show()

    # C
    # shot accuracy by home or away games
    from scipy.spatial import ConvexHull, convex_hull_plot_2d
    from matplotlib.lines import Line2D

    colors = {
        'Back Court(BC)': '#8e24aa',  # purple
        'Center(C)': '#3949ab',  # indigo
        'Left Side Center(LC)': '#03a9f4',  # light blue
        'Left Side(L)': '#ffeb3b',  # yellow
        'Right Side Center(RC)': '#ec407a',  # pink
        'Right Side(R)': '#ffa726'  # orange
    }

    home_or_away_str = ['away', 'home']

    for n, g in data.groupby(['home_game'], as_index=False):
        sub_data = g[['shot_zone_area', 'loc_x', 'loc_y', 'shot_made_flag']]
        hulls = []
        legends = []
        for n2, g2 in sub_data.groupby('shot_zone_area', as_index=False):
            points = g2[['loc_x', 'loc_y']].to_numpy()
            hulls.append({
                'name': n2,
                'points': points,
                'hull': None if points.size < 3 else ConvexHull(points)
            })
            legends.append(Line2D([0], [0], color=colors[n2], lw=4, label=n2 +
                                  ' ' + '{:.2%}'.format(g2['shot_made_flag'].mean())))
        plt.figure(figsize=(12, 11))
        ax = draw_court()
        for hull in hulls:
            # plt.text(hull['centroid'][0], hull['centroid'][1], hull['accuracy'], fontsize=12)
            if hull['hull'] is not None:
                for simplex in hull['hull'].simplices:
                    plt.plot(hull['points'][simplex, 0], hull['points']
                             [simplex, 1], 'k-', color=colors[hull['name']], linewidth=4)

        ax.legend(handles=legends, loc='center')

        plt.xlim(-250, 250)
        # # Descending values along th y axis from bottom to top
        # # in order to place the hoop by the top of plot
        plt.ylim(422.5, -47.5)
        # # get rid of axis tick labels
        # # plt.tick_params(labelbottom=False, labelleft=False)
        plt.title('shot accuracy by shot_zone_area on ' +
                  home_or_away_str[n] + ' games')
        plt.show()

    # C
    # use in case correlations need to be regenerated
    # generate_correlations(data, test)
    pd.read_csv('shot_made_correlations.csv')[chosen_features].plot.line()

    plt.title('Correlations with shot made through time', color='black')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.show()

    # D
    draw_efficiency_pdf_cdf_until_date(data, test, '05/05/1997')


def db_scan(data):
    # Perform DBSCAN clustering
    X = data[['loc_x', 'loc_y']]
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=0.07, min_samples=50).fit(X)
    labels = db.labels_
    data["cluster"] = labels
    # chart
    plt.figure(figsize=(12, 11))
    # plt.scatter(data.loc_x, data.loc_y, c=np.where((data['shot_made_flag'] == 1), 'g', 'r'))
    plt.scatter(data.loc_x, data.loc_y, c=data.cluster)
    draw_court()
    # Adjust plot limits to just fit in half court
    plt.xlim(-250, 250)
    # Descending values along th y axis from bottom to top
    # in order to place the hoop by the top of plot
    plt.ylim(422.5, -47.5)
    # get rid of axis tick labels
    # plt.tick_params(labelbottom=False, labelleft=False)
    plt.show()
