import requests
import urllib3
urllib3.disable_warnings()

URL = 'https://discosweb.esoc.esa.int'
    

def get_discos_cat(catid, token):
    """
    catid (?):
        discos catalogue id
    token (str):
        access token from DISCOS
    """
    response = requests.get(f'{URL}/api/objects',
        headers={'Authorization': f'Bearer {token}','DiscosWeb-Api-Version': '2'},
        #params={'filter': f'eq(reentry.epoch,null)&eq(satno,{catid})', # some objected decayed at the time of this running
        params={'filter': f'eq(satno,{catid})',
                'sort': 'satno','page[size]': 10,'page[number]':1},
                verify=False)
    doc = response.json()
    doc = doc['data']
    
    time.sleep(5) 
    
    for i in doc:
        nameo = i['attributes']['name']
        satno = i['attributes']['satno']
        objectClass = i['attributes']['objectClass']
        mission = i['attributes']['mission']
        mass = i['attributes']['mass']
        shape = i['attributes']['shape']
        width = i['attributes']['width']
        height = i['attributes']['height']
        depth = i['attributes']['depth']
        diameter = i['attributes']['diameter']
        span = i['attributes']['span']
        xSectMax = i['attributes']['xSectMax']
        xSectMin = i['attributes']['xSectMin']
        xSectAvg = i['attributes']['xSectAvg']
    
    return nameo, satno, objectClass, mission, mass, shape, width, height, depth, diameter, span, xSectMax, xSectMin, xSectAvg

# Options: owhd, osp, os1, os2, os3, rbd; o=object, w=width, h=height, d=depth, sp=sphere, s=starlink, 1=grupo1,etc, rbd=rocket bodies + debris

