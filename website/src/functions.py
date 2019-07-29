# The basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _choose_max_arg(df, item_name, match=False):
    if match:
        mask = df.name.str.match(item_name)
    else:
        mask = df.name.str.contains(item_name)
    max_val = df[mask].groupby('business_id').count().max()
    _ = df[mask].groupby('business_id').count() == max_val
    new_mask = _.iid
    return _[new_mask].index[0]


def build_item_map(df):
    '''
    Takes in a dataframe with fields business_id, name, and i_business_id (the item
    number for surprise), and returns a dictionary with business_id's as keys and
    tuples containing the name and i_business_id as values based on the subset of
    restaurants that were selected for the website demo. These restaurants were
    hard coded specifically for this implementation of the website.
    '''
    business_ids = [_choose_max_arg(df, 'The Stand'),
                    _choose_max_arg(df, 'In-N-Out'),
                    _choose_max_arg(df, 'Shake Shack'),
                    _choose_max_arg(df, 'Chick-fil-A'),
                    _choose_max_arg(df, 'ATL Wings'),
                    _choose_max_arg(df, 'Firehouse'),
                    _choose_max_arg(df, "Hungry Howie"),
                    _choose_max_arg(df, 'Buffalo Wild Wings'),
                    _choose_max_arg(df, 'Arrogant'),
                    _choose_max_arg(df, 'Dressing Room'),
                    _choose_max_arg(df, 'Pizzeria Bianco'),
                    _choose_max_arg(df, 'Angels Trumpet'),
                    _choose_max_arg(df, 'Citizen Public'),
                    _choose_max_arg(df, "Hearth '61"),
                    _choose_max_arg(df, 'Durant'),
                    _choose_max_arg(df, 'Steak 44'),
                    _choose_max_arg(df, 'Buck & Rider'),
                    _choose_max_arg(df, 'Avanti'),
                    _choose_max_arg(df, 'Cafe Monarch'),
                    _choose_max_arg(df, 'Sel', match=True),
                    _choose_max_arg(df, "Binkley's Restaurant"),
                    _choose_max_arg(df, "Mastro's City Hall"),
                    _choose_max_arg(df, "Dominick's Steakhouse"),
                    _choose_max_arg(df, 'Bourbon Steak'),
                    _choose_max_arg(df, "Morton's The Steakhouse")]

    names = df.name[df.business_id.isin(business_ids)].unique()
    iids = df.iid[df.business_id.isin(business_ids)].unique()

    item_map = {}
    for item in business_ids:
        item_map[item] = (df.name[df.business_id.str.match(item)].unique()[0],
                          df.iid[df.business_id.str.match(item)].unique()[0])

    return item_map


def get_similarity_matrix(fitted_model):
    '''
    Compute similarity matrix from a fitted model and convert it to a
    Pandas Dataframe. This won't use any additional memory, but lends more
    functionality.
    '''
    similarity_matrix = fitted_model.compute_similarities()
    return pd.DataFrame(similarity_matrix)


def get_recs(similarity_matrix, data_df, item_map, n, selections=['In-N-Out Burger', 'Chick-fil-A', 'The Stand', 'Whataburger']):
    '''
    Use the similarity matrix to find most similar restaurants to the four
    that the user specified in their selections.

    Params:
    ~~~~~~~~~~~~
    similarity_matrix: type - Pandas Dataframe
    Similarity matrix from our fitted production model

    data_df: type - Pandas Dataframe
    Dataframe that has all of the pertinent data so that we can ensure we don't
    the same chain of restaurants to someone just because they had a different
    location.

    item_map: type - Dict
    Dictionary that maps the business_id, name, and i_business_id (for surprise), with
    business_id as keys, and tupes of name and i_business_id as values.

    n: type - in
    Number of recommendations to provide

    selections: type - list of strings
    Names of restaurants that user selected for recommendations.
    '''
    # Create new dataframe for holding the vectors of similarities for
    # the user's selections
    item_vectors = pd.DataFrame()
    for key, val in item_map.items():
        if val[0] in selections:
            item_vectors[val[1]] = similarity_matrix.iloc[:, val[1]]

    # Take mean similarity across columns and sort by it to
    # raise the best picks to the top
    item_vectors['mean_similarity'] = item_vectors.mean(axis=1)
    item_vectors.sort_values(by='mean_similarity', ascending=False, inplace=True)
    picks = item_vectors.index[:n + 2]

    # Make sure that the picks don't have the same exact name
    copy_df = data_df.copy(deep=True)
    mask = copy_df.name.isin(selections)
    # picks = item_vectors[~name_mask].index.unique()[:n]
    copy_df = copy_df[~mask]

    # Make sure that the picks are open
    mask = copy_df.is_open == 1
    # picks = item_vectors[open_mask].index.unique()[:n]
    copy_df = copy_df[mask]

    # picks = item_vectors[~name_mask].iloc[:n, :].index
    recs = copy_df[['business_id', 'name', 'iid']].drop_duplicates(['iid'])
    recs = recs[recs.iid.isin(picks)]

    return recs


def calculate_region_density(state, businesses, bus_reviews):
    num_businesses = businesses.filter((businesses.categories.like('%Restaurants%')) &
                                       (business_df.state == state)).count()
    num_reviews = bus_reviews.filter((bus_reviews.categories.like('%Restaurants%')) &
                                     (bus_reviews.state == state)).count()
    return num_reviews / num_businesses


def compare_region_densities(regions, businesses, bus_reviews):
    max_density = 0
    density_dict = {}
    for region in regions:
        density = calculate_region_density(region, businesses, bus_reviews)
        density_dict[region] = density
        if density > max_density:
            max_density = density
            max_region = region

    print('Best Region: {}\nBest Density: {}'.format(max_region, max_density))
    return density_dict
