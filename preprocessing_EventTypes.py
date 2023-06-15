# extract unique event values from the training data

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

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


def readLabels(fileLocation):
    """Read the labels dataset"""
    labels = pd.read_csv(fileLocation)
    labels['session'] = labels.session_id.apply(lambda x: int(x.split('_')[0]))
    labels['q'] = labels.session_id.apply(lambda x: int(x.split('_')[-1][1:]))
    
    return labels


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
    

    eventCounts = eventTable.pivot(index=['session_id', 'level_group'], columns='event_label', values='counts')
    eventCounts = eventCounts.fillna(0)
    return eventCounts


def splitDataset(dataset, labels, train_ratio=0.80):
    """Random split of session_ids in the data.
       The dataset should be indexed by ['session_id', 'level_group']
    """
    
    # `session_id` and `level_group` are the indices of our feature engineered dataset
    if dataset.index.names != ['session_id', 'level_group']:
        raise Exception( 'Data must be indexed by [session_id, level_group]' )
    
    sessionIds = dataset.index.get_level_values('session_id').unique()
    trainIds, valIds = train_test_split(sessionIds, train_size=train_ratio )

    trainData = dataset.loc[trainIds]
    valData =  dataset.loc[valIds]
    
    trainLabels = labels[ labels.session.isin(trainIds) ]
    valLabels = labels[ labels.session.isin(valIds) ]
    
    return trainData, trainLabels, valData, valLabels



def fullProcessing(dataFile, labelFile):
    trainData = readData(dataFile)
    labelData = readLabels(labelFile)


    eventLabels = makeEventLabels(trainData)
    
    eventTable = makeEventTable(trainData, eventLabels)
    
    # split datasets
    trainData, trainLabels, valData, valLabels = splitDataset(eventTable, labelData, train_ratio=0.8)
    
    return trainData, trainLabels, valData, valLabels, eventLabels