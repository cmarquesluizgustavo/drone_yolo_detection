import math
from functools import lru_cache

from pyproj import proj

class GeoConverter:

    @staticmethod
    def _utm_zone_from_lon(lon: float) -> int:
        # UTM zones are 1..60 spanning -180..180.
        # See: https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system
        zone = int((lon + 180.0) / 6.0) + 1
        return max(1, min(60, zone))

    @staticmethod
    @lru_cache(maxsize=64)
    def _utm_proj(zone: int, south: bool) -> proj.Proj:
        return proj.Proj(proj='utm', zone=zone, ellps='WGS84', south=south)

    @staticmethod
    def _proj_for_ref(ref_lat: float, ref_lon: float) -> proj.Proj:
        zone = GeoConverter._utm_zone_from_lon(float(ref_lon))
        south = float(ref_lat) < 0.0
        return GeoConverter._utm_proj(zone, south)

    @staticmethod
    def polar_to_xy(ref_x, ref_y, bearing_deg, distance_m):
        x = math.sin(math.radians(bearing_deg)) * distance_m
        y = math.cos(math.radians(bearing_deg)) * distance_m
        return ref_x + x, ref_y + y

    @staticmethod
    def geo_to_xy(lat, lon):
        p = GeoConverter._proj_for_ref(float(lat), float(lon))
        return p(float(lon), float(lat))

    @staticmethod
    def xy_to_geo(x, y, *, ref_lat=None, ref_lon=None, zone=None, south=None):
        if ref_lat is not None and ref_lon is not None:
            p = GeoConverter._proj_for_ref(float(ref_lat), float(ref_lon))
        elif zone is not None and south is not None:
            p = GeoConverter._utm_proj(int(zone), bool(south))
        else:
            raise ValueError("xy_to_geo requires ref_lat/ref_lon or zone/south")

        lon, lat = p(float(x), float(y), inverse=True)
        return lat, lon

    @staticmethod
    def polar_to_geo(ref_lat, ref_lon, bearing_deg, distance_m):
        ref_x, ref_y = GeoConverter.geo_to_xy(ref_lat, ref_lon)
        x, y = GeoConverter.polar_to_xy(ref_x, ref_y, bearing_deg, distance_m)
        return GeoConverter.xy_to_geo(x, y, ref_lat=ref_lat, ref_lon=ref_lon)