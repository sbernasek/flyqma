import requests


class API:
    """
    Object for querying an API.
    """

    def __init__(self, URL, API_KEY, timeout=30):
        """ Initialize with a <URL> and an <API_KEY>. """
        self.URL = URL
        self.H = {'X-Mashape-Key': API_KEY, "Accept": "application/json"}
        self.timeout = timeout

    def query(self, query):
        """ Sends <query> to API and returns response. """
        try:
            response = requests.get(query,headers=self.H, timeout=self.timeout)
        except requests.exceptions.Timeout:
            response = None
        return response.json()

    def build_query(self, query):
        """ Returns formatted <query>. """
        return '{:s}/{:s}'.format(self.URL, query)




class NUShuttleAPI(API):
    """
    Object for building an API query.
    """

    URL = 'https://transloc-api-1-2.p.mashape.com'
    API_KEY = 'hixX24yb5YmshQO9NoxYaxwlYibwp1HBO1kjsn854hpcCbIG3a'

    LOCATIONS = {
        'chicago': '41.881832,-87.623177|{:0.2f}'.format(30*1000),
        'home': '42.003810,-87.663420|{:0.2f}'.format(1*1000)}

    def __init__(self, geotag=None):
        """
        Instantiate TransLoc API.
        """
        super().__init__(self.URL, self.API_KEY)

        # store agency ID
        self.agency_id = 665

        # store geotag
        if type(geotag) == str:
            geotag = self.LOCATIONS[geotag]
        self.geotag = geotag

    def query(self, query):
        """ Sends <query> to API and returns response. """
        return super().query(self.build_query(query))

    @property
    def routes(self):
        """ Query all routes by agency. """
        return self.query((self._route(self.agency_id)))

    @property
    def stops(self):
        """ Query stops served by agency. """
        return self.query((self._stops(self.agency_id, self.geotag)))

    @staticmethod
    def _route(agency_id, geotag=None):
        """ Query routes offered by <agency_id>, optionally within <geotag>. """
        query = 'routes.json?&callback=call'
        query += '&agencies={:d}'.format(agency_id)
        if geotag is not None:
            query += '&geo_area={:s}'.format(geotag)
        return query

    @staticmethod
    def _stops(agency_id, geotag=None):
        """ Query stops served by <agency_id>, optionally within <geotag>. """
        query = 'stops.json?&callback=call'
        query += '&agencies={:d}'.format(agency_id)
        if geotag is not None:
            query += '&geo_area={:s}'.format(geotag)
        return query


class ShuttleRouteAPI(NUShuttleAPI):

    def __init__(self, route_id=None, geotag='chicago'):
        """
        Instantiate Shuttle Route API.
        """
        super().__init__(geotag)
        self.route_id = route_id

    @property
    def stops(self):
        """ Stops visited by this route. """
        stops = self.query((self._stops(self.agency_id)))
        included = lambda s: str(self.route_id) in s['routes']
        return {s['name']: s['stop_id'] for s in stops['data'] if included(s)}

    def query_arrival_time(self, stop_id):
        """ Query arrival times for <stop_id> """
        args = (self.agency_id, self.route_id, stop_id)
        return self.query((self._arrival(*args)))

    @staticmethod
    def _arrival(agency_id, route_id, stop_id):
        """ Query arrival times. """
        query = 'arrival-estimates.json?&callback=call'
        query += '&agencies={:d}'.format(agency_id)
        query += '&routes={:d}'.format(route_id)
        query += '&stops={:d}'.format(stop_id)
        return query


class IntercampusAPI(ShuttleRouteAPI):

    STOPS ={
        'to_evanston': 8204724,
        'to_chicago': 8204726}

    def __init__(self):
        super().__init__(route_id=8005040, geotag='home')

    @property
    def northbound_arrival(self):
        """ Query arrival time for Northbound shuttle. """
        return self.build_query(self._arrival(self.agency_id, self.route_id, self.STOPS['to_evanston']))

    @property
    def southbound_arrival(self):
        """ Query arrival time for Southbound shuttle. """
        return self.build_query(self._arrival(self.agency_id, self.route_id, self.STOPS['to_chicago']))
