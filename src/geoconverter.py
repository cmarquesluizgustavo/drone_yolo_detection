import math
from pyproj import proj

class GeoConverter():
    
    @staticmethod
    def polar_to_xy(ref_x, ref_y, theta, r):
        x = math.sin(math.radians(theta)) * r
        y = math.cos(math.radians(theta)) * r
        if ref_x is not None and ref_y is not None:
            x = ref_x + x
            y = ref_y + y
        return x, y

    @staticmethod
    def polar_to_geo(ref_lat, ref_lon, theta, r):
        ref_x, ref_y = GeoConverter.geo_to_xy(ref_lat, ref_lon)
        x, y = GeoConverter.polar_to_xy(ref_x, ref_y, theta, r)
        return GeoConverter.xy_to_geo(x, y)

    @staticmethod
    def geo_to_polar(ref_lat, ref_lon, lat, lon):
        ref_x, ref_y = GeoConverter.geo_to_xy(ref_lat, ref_lon)
        x, y = GeoConverter.geo_to_xy(lat, lon)
        return GeoConverter.xy_to_polar(ref_x, ref_y, x, y)

    @staticmethod
    def xy_to_polar(ref_x, ref_y, x, y):
        if ref_x is None or ref_y is None:
            dx = x
            dy = y
        else:
            dx = x - ref_x
            dy = y - ref_y
        r = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
        theta = math.degrees(math.atan2(dx, dy)) 
        theta = (theta + 360) % 360 
        return theta, r

    @staticmethod
    def geo_to_xy(lat, lon):
        if lat is None or lon is None:
            return None, None
        p = proj.Proj(proj='utm', zone=23, ellps='WGS84', preserve_units=False)
        x, y = p(lon, lat)
        return x, y

    @staticmethod
    def xy_to_geo(x, y):
        if x is None or y is None:
            return None, None
        p = proj.Proj(proj='utm', zone=23, ellps='WGS84',
                      preserve_units=False)  # assuming you're using WGS84 geographic
        lon, lat = p(x, y, inverse=True)
        return lat, lon