# The basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def build_item_map(df):
    '''
    Takes in a dataframe with fields business_id, name, and i_business_id (the item 
    number for surprise), and returns a dictionary with business_id's as keys and 
    tuples containing the name and i_business_id as values based on the subset of
    restaurants that were selected for the website demo. These restaurants were
    hard coded specifically for this implementation of the website. 
    '''
    business_ids = [df.business_id[df.name.str.match("The Stand")].iloc[0],
                    df.business_id[df.name.str.contains("Chick-fil-A")].iloc[0],
                    df.business_id[df.name.str.contains("In-N-Out")].iloc[0],
                    df.business_id[df.name.str.contains("Shake Shack")].iloc[0],
                    df.business_id[df.name.str.contains("Whataburger")].iloc[0],
                    df.business_id[df.name.str.contains("Firehouse")].iloc[0],
                    df.business_id[df.name.str.contains("Arrogant")].iloc[0],
                    df.business_id[df.name.str.contains("Dressing Room")].iloc[0],
                    df.business_id[df.name.str.contains("Pizzeria Bianco")].iloc[0],
                    df.business_id[df.name.str.contains("Angels Trumpet")].iloc[0],
                    df.business_id[df.name.str.contains("Citizen Public")].iloc[0],
                    df.business_id[df.name.str.contains("Roaring Fork")].iloc[0],
                    df.business_id[df.name.str.contains("Hearth '61")].iloc[0],
                    df.business_id[df.name.str.contains("Durant")].iloc[0],
                    df.business_id[df.name.str.contains("Steak 44")].iloc[0],
                    df.business_id[df.name.str.contains("Buck & Rider")].iloc[0],
                    df.business_id[df.name.str.contains("Avanti")].iloc[0],
                    df.business_id[df.name.str.contains("Cafe Monarch")].iloc[0],
                    df.business_id[df.name.str.match("Sel")].iloc[0],
                    df.business_id[df.name.str.contains("Binkley's Restaurant")].iloc[0],
                    df.business_id[df.name.str.contains("Mastro's City Hall")].iloc[0],
                    df.business_id[df.name.str.contains("Dominick's Steakhouse")].iloc[0],
                    df.business_id[df.name.str.contains("Bourbon Steak")].iloc[0],
                    df.business_id[df.name.str.contains("Morton's The Steakhouse")].iloc[0]]
    names = df.name[df.business_id.isin(business_ids)].unique()
    i_business_ids = df.i_business_id[df.business_id.isin(business_ids)].unique()

    item_map = {}
    for item in business_ids:
        item_map[item] = (df.name[df.business_id.str.match(item)].unique()[0],
                          df.i_business_id[df.business_id.str.match(item)].unique()[0])
    return item_map


def get_similarity_matrix(fitted_model):
    '''
    Compute similarity matrix from a fitted model and convert it to a 
    Pandas Dataframe. This won't use any additional memory, but lends more
    functionality.
    '''
    similarity_matrix = fitted_model.compute_similarities()
    return pd.DataFrame(similarity_matrix)


def recommend(similarity_matrix, data_df, item_map, n, selections=['In-N-Out Burger', 'Chick-fil-A', 'The Stand', 'Whataburger']):
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

    # Make sure that the picks don't have the same exact name
    name_mask = data_df.name.isin(selections)
    picks = item_vectors[~name_mask].index.unique()[:n]
    print(picks)

# #     picks = item_vectors[~name_mask].iloc[:n, :].index
    recs = data_df[['business_id', 'name', 'i_business_id']].drop_duplicates(['i_business_id'])
    recs = recs[recs.i_business_id.isin(picks)]

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
