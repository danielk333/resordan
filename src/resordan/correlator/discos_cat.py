import requests

URL = 'https://discosweb.esoc.esa.int'


def item_as_tuple(item):
    """converting result item into a tuple"""
    nameo = item['attributes']['name']
    satno = item['attributes']['satno']
    objectClass = item['attributes']['objectClass']
    mission = item['attributes']['mission']
    mass = item['attributes']['mass']
    shape = item['attributes']['shape']
    width = item['attributes']['width']
    height = item['attributes']['height']
    depth = item['attributes']['depth']
    diameter = item['attributes']['diameter']
    span = item['attributes']['span']
    xSectMax = item['attributes']['xSectMax']
    xSectMin = item['attributes']['xSectMin']
    xSectAvg = item['attributes']['xSectAvg']
    return (
        nameo, satno, objectClass, mission, mass, shape, width,
        height, depth, diameter, span, xSectMax, xSectMin, xSectAvg
    )


def get_discos_objects(object_ids, token):
    """
    query discos service for data on a collection of objecs

    Params
    ------
        object_ids: (list)
            list of object ids (str)
        token: (str)
            service token for discos service

    Returns
    -------
        list (tuple)
            list of (object_id, data_tuple)
            data_tuple : (
                nameo, satno, objectClass, mission, mass, shape, width,
                height, depth, diameter, span, xSectMax, xSectMin, xSectAvg)

    """
    object_ids_str = ','.join([str(i) for i in object_ids])
    response = requests.get(
        f'{URL}/api/objects',
        headers={'Authorization': f'Bearer {token}', 'DiscosWeb-Api-Version': '2'},
        params={'filter': f'in(satno,({object_ids_str}))', 'page[size]': 100},
        verify=False)

    doc = response.json()

    data = {item['attributes']['satno']: item_as_tuple(item) for item in doc['data']} 

    return data
