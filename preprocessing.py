# extract unique event values from the training data

import numpy as np
import pandas as pd


eventVars = ['event_name', 'name', 'level', 'page', 'text', 'fqid', 'room_fqid', 'text_fqid' ]
eventVars.sort()

numericalVars = ['elapsed_time','room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y',
        'hover_duration']


def readData(fileLocation):
    dtypes={
        'elapsed_time':np.int32,
        'event_name':'category',
        'name':'category',
        'level':'category',
        'page':'category',
        'room_coor_x':np.float32,
        'room_coor_y':np.float32,
        'screen_coor_x':np.float32,
        'screen_coor_y':np.float32,
        'hover_duration':np.float32,
        'text':'category',
        'fqid':'category',
        'room_fqid':'category',
        'text_fqid':'category',
        'fullscreen':'category',
        'hq':'category',
        'music':'category',
        'level_group':'category'}
    data = pd.read_csv(fileLocation, dtype=dtypes)

    for column in eventVars:
        data[column] = data[column].cat.add_categories(['-1'])
        data[column] = data[column].fillna('-1')
    return data


def makeEventLabels(trainData):
    """Make a table containing the labels for any set of event values"""
    eventGrouping = trainData.groupby(eventVars, observed=True)
    eventLabels = pd.DataFrame( eventGrouping.size().index,
                    columns=['event_profile'])

    eventLabels['event_label'] = pd.DataFrame( 
            map(lambda i: 'e_'+ str(i), range(len(eventLabels))),
            dtype='category')
    return eventLabels


def makeEventTable(data, eventLabels):
    """makes a table grouped by event types, session_id, and level_group"""
    eventColumns = ['session_id', 'level_group', *eventVars]
    eventTable = data[ eventColumns ]

    eventTable = eventTable.groupby(eventColumns, observed=True).size().to_frame('counts')
    eventTable = eventTable.reset_index(['session_id', 'level_group'])

    eventDetails = pd.DataFrame( eventTable.index, columns=['event_profile'])
    eventDetails = eventDetails.merge( eventLabels, on='event_profile', how='left' )

    eventTable = eventTable.reset_index().drop(columns=eventVars)
    eventTable['event_label'] = eventDetails['event_label']

    print(eventTable)

    eventCounts = eventTable.pivot(index=['session_id', 'level_group'], columns='event_label', values='counts')


    return eventCounts.fillna(0)




