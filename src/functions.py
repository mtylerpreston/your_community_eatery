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
