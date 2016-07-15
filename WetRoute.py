import json
import datetime
import time
import urllib
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import shapefile as sf
import calendar

api_main='dcb895bf062fef57'

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a)) 
    r = 3956.0 # Approximate radius of earth in miles
    return c * r

def calcBearing(lon1, lat1, lon2, lat2):
    """
    calculates the compass direction of the great circle connecting two points 
    """
    lon1, lat1, lon2, lat2 = map(np.radians,[lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) \
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(y, x))

def calcProjectile(lon, lat, bearing, distance):
    """
    calculates the new coordinates at the end of  a straight line with a given
    distance and bearing from the intial point
    """
    r = 3956.0
    Dd = distance/r
    Cc, lona, lata = map(np.radians, [bearing, lon, lat])
    latb = np.arcsin(np.cos(Cc)*np.cos(lata)*np.sin(Dd) + 
                     np.sin(lata)*np.cos(Dd))
    lonb = lona + np.arctan2(np.sin(Cc)*np.sin(Dd)*np.cos(lata), 
                np.cos(Dd)-np.sin(lata)*np.sin(latb))
    return (np.degrees(lonb), np.degrees(latb))

def dumbProximity(route_frame, index_points):
    """
    for each point in route_frame, finds which point given by index_points is 
    nearest by building a distance matrix and finding the min arg of each row
    """
    dmat = np.zeros(shape = [route_frame.shape[0], len(points)])
    for i in range(dmat.shape[0]):
        for j in range(dmat.shape[1]):
            lon1, lat1 = route_frame.loc[i,['lon','lat']]
            lon2, lat2 = route_frame.loc[points[j],['lon','lat']]
            dmat[i,j] =  haversine(lon1, lat1, lon2, lat2)
    return np.apply_along_axis(np.argmin,1,dmat)

def smartProximity(route_frame, points):
    """
    for each point in route_frame, finds which point given by index_points is 
    nearest by iterating through route_frame and picking the nearest index_point
    from the two points its index is in between
    """
    nearest_point = [0 for i in range(points[-1])]
    bound_pts = [points[0], points[0]]
    lonlow, latlow = route_frame.loc[bound_pts[0],['lon','lat']]
    lonhigh, lathigh = route_frame.loc[bound_pts[1],['lon','lat']]
    upper=False
    j=0
    i=0
    while i<points[-1]:
        if i==bound_pts[1]:
            upper=False
            j+=1
            bound_pts = [points[j], points[j+1]]
            lonlow, latlow = route_frame.loc[bound_pts[0],['lon','lat']]
            lonhigh, lathigh = route_frame.loc[bound_pts[1],['lon','lat']]
            nearest_point[i]=j
        elif upper==True:
            nearest_point[i]=j+1
        else:
            lon, lat =  route_frame.loc[i,['lon','lat']]
            d = [haversine(lon, lat, lonlow, latlow),
                 haversine(lon, lat, lonhigh, lathigh)]
            if np.argmin(d):
                nearest_point[i]=j+1
                upper=True
            else:
                nearest_point[i]=j
        i+=1
    return (nearest_point + 
            [j+1 for i in range(route_frame.shape[0]-points[-1])])


def getForecastCodes(lon,lat,api_key=api_main):
    """
    makes a call to weather underground api for forecast codes at a given lat 
    lon and returns the forecast code for the next 32 hours
    """
    api_call = ('http://api.wunderground.com/api/' + api_key + '/hourly/q/' +
               str(lat) +',' + str(lon) + '.json')
    data_file =  urllib.urlopen(api_call)
    data = json.load(data_file)
    codes=[forhour['fctcode'] for forhour in data['hourly_forecast']] 
    return codes

def readLocalCodes(i):
    """
    development function to avoid excessive calls to weather underground api
    reads ith set of forecast codes from 'localcodes' folder
    """
    data_file = open('localcodes/'+str(i)+'.json', 'r')
    data = json.load(data_file)
    codes=[forhour['fctcode'] for forhour in data['hourly_forecast']] 
    return codes
    

def readRoute(routefile):
    """
    parses a gpx route with lat, lon and time values at each point and returns 
    a pandas DataFrame object
    """
    tree = ET.parse(routefile)
    root = tree.getroot()
    latlist = [bit.attrib['lat'] for bit in root[0][0]]
    lonlist = [bit.attrib['lon'] for bit in root[0][0]]
    tstrlist = [bit[1].text for bit in root[0][0]]
    dateform = '%Y-%m-%dT%H:%M:%SZ'
    timelist = [datetime.datetime.strptime(bit,dateform) for bit in tstrlist]
    route_frame = pd.DataFrame(np.array([lonlist,latlist,timelist]).T, 
        columns = ['lon','lat','t'])
    return route_frame

def synthRoute(shape='I90',speed=65.,speedunit='mph'):
    """
    creates a synthetic route based on input shapefile and speed. route starts 
    at the first point and proceeds at 'speed' to the last point, hitting each 
    point in between with points placed every five seconds
    """
    speed = speed/60./60.
    if speedunit=='kph':
        speed = speed * .6213
    elif speedunit != 'mph':
        raise ValueError('speedunit must be either "mph" or "kph"')
    step = speed*5.
    f = sf.Reader(shape)
    pts = f.shape().points
    tfin=0.0
    route_frame = pd.DataFrame(columns=['lon','lat','time'])
    for i in range(len(pts)-1):
        tnow = tfin
        lonnow,latnow = pts[i]
        lonnxt,latnxt = pts[i+1]
        dp = haversine(lonnow, latnow, lonnxt, latnxt)
        bearing = calcBearing(lonnow, latnow, lonnxt, latnxt)
        tfin = tnow + dp/speed
        tnow = tnow - tnow%5.
        while tnow +5. < tfin:
            tnow = tnow + 5.
            lonnow, latnow = calcProjectile(lonnow, latnow, bearing, step)
            route_frame = route_frame.append( {'lon': lonnow, 'lat':latnow,
                               'time':tnow},ignore_index=True)
    return route_frame

def OSRM_Route(lon1, lat1, lon2, lat2, mode='driving'):
    api_call = ("http://localhost:5000/route/v1/" + mode + "/" + 
                str(lon1) + ',' + str(lat1) + ';' +
                str(lon2) + ',' + str(lat2) + 
                "?steps=true&geometries=geojson")
    data_file =  urllib.urlopen(api_call)
    data = json.load(data_file)
    t = 0
    route_frame = pd.DataFrame(columns=['lon','lat','time','speed'])
    for step in data['routes'][0]['legs'][0]['steps']:
        Dt = step['duration']
        Dd = step['distance']
        if Dt!=0:
            sd = Dd*1./Dt
        else:
            sd = 1
        points = step['geometry']['coordinates']
        dt = Dt*1./len(points)
        for point in points:
            t = t+dt
            route_frame = route_frame.append( {'lon': point[0], 'lat': point[1],
                               'time': t, 'speed':sd},ignore_index=True)
    return route_frame

def forecastMatrix(forecast_frame ,dev=True):
    """
    generates a forecast matrix with rows corresponding to the size of forecast 
    frame and 36 columns corresonding to hours
    """
    Npt = forecast_frame.shape[0]
    forecast_frame = forecast_frame.set_index([range(Npt)])
    Ncast = 36
    fmat = np.zeros([Npt,Ncast],int)
    for i in range(Npt):
        if dev:
            fmat[i,:] = readLocalCodes(i)
        else:
            lon,lat = forecast_frame.loc[i,['lon','lat']]
            fmat[i,:] = getForecastCodes(lon,lat,api_key=api_main)
    return fmat

def forecastRoute(route_frame, Npoints = 10, dev=True):
    """
    takes a route and finds Npoints number of points along the route, calls the 
    weather underground API and pairs the nearest forecast to each point on the 
    route. Returns the route with forecast description number as a column
    """
    Nroute = route_frame.shape[0]
    forecast_points = [int((i+.5)*Nroute/Npoints) for i in range(Npoints)]
    nearest_point = smartProximity(route_frame, forecast_points)
    past_hour = (time.time()/60./60.)%1
    forecast_hour = (past_hour + route_frame['time']/60./60.+.5).apply(int)
    fmat = forecastMatrix(route_frame.loc[forecast_points], dev)
    forecastcodes=np.zeros(Nroute,int)
    for i in range(Nroute):
        forecastcodes[i] = fmat[nearest_point[i],forecast_hour[i]]
    route_frame['wt']=pd.Series(forecastcodes)
    return route_frame



lon1=-76.138793
lat1=43.074061
lon2=-73.985173
lat2=44.289662

route_frame = OSRM_Route(lon1, lat1, lon2, lat2, mode='driving')

route_frame = forecastRoute(route_frame, Npoints=10, dev=False)

route_frame['wt'].unique()








