class GeoConverter:

    @staticmethod
    def polar_to_xy(ref_x, ref_y, bearing_deg, distance_m):
        x = math.sin(math.radians(bearing_deg)) * distance_m
        y = math.cos(math.radians(bearing_deg)) * distance_m
        return ref_x + x, ref_y + y

    @staticmethod
    def geo_to_xy(lat, lon):
        p = proj.Proj(proj='utm', zone=23, ellps='WGS84')
        return p(lon, lat)

    @staticmethod
    def xy_to_geo(x, y):
        p = proj.Proj(proj='utm', zone=23, ellps='WGS84')
        lon, lat = p(x, y, inverse=True)
        return lat, lon

    @staticmethod
    def polar_to_geo(ref_lat, ref_lon, bearing_deg, distance_m):
        ref_x, ref_y = GeoConverter.geo_to_xy(ref_lat, ref_lon)
        x, y = GeoConverter.polar_to_xy(ref_x, ref_y, bearing_deg, distance_m)
        return GeoConverter.xy_to_geo(x, y)