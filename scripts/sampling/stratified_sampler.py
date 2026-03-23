import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
import os
import warnings
warnings.filterwarnings('ignore')
from scripts.utils.logger import get_logger
from scripts.utils.config_loader import load_config

logger = get_logger("stratified_sampler")
config = load_config()

# ── Land mask ─────────────────────────────────────────────────────────────────
logger.info("Loading land mask...")
world = gpd.read_file(
    "/srv/THESIS/energy_profiling_thesis/data/raw/ne_countries/"
    "ne_110m_admin_0_countries.shp"
)
world_union = world.geometry.buffer(0.5).unary_union


def is_on_land(lat, lon):
    point = Point(lon, lat)
    return world_union.contains(point)


# ── Stratum definitions ───────────────────────────────────────────────────────
# Every region chosen for maximum feature density at that location
# OSM coverage + GHSL + VIIRS + Solar Atlas all verified per region

STRATA = {

    # ── DENSE URBAN ──────────────────────────────────────────────────────────
    "dense_urban": {
        "count": 750,
        "regions": [
            # EUROPE — best OSM coverage globally
            # Berlin Germany
            {"lat": (52.3, 52.7), "lon": (13.1, 13.7)},
            # Hamburg Germany
            {"lat": (53.4, 53.7), "lon": (9.7, 10.2)},
            # Munich Germany
            {"lat": (47.9, 48.3), "lon": (11.3, 11.8)},
            # Cologne + Dusseldorf Germany
            {"lat": (50.8, 51.5), "lon": (6.5, 7.2)},
            # Frankfurt Germany
            {"lat": (49.9, 50.3), "lon": (8.4, 9.0)},
            # Paris France
            {"lat": (48.6, 49.1), "lon": (2.0, 2.7)},
            # Lyon + Marseille France
            {"lat": (43.0, 45.8), "lon": (4.6, 5.6)},
            # London UK
            {"lat": (51.3, 51.7), "lon": (-0.5, 0.3)},
            # Manchester + Leeds UK
            {"lat": (53.3, 53.9), "lon": (-2.5, -1.3)},
            # Amsterdam Netherlands
            {"lat": (52.2, 52.6), "lon": (4.6, 5.1)},
            # Brussels Belgium
            {"lat": (50.6, 51.0), "lon": (4.1, 4.6)},
            # Warsaw Poland
            {"lat": (52.0, 52.5), "lon": (20.7, 21.4)},
            # Krakow Poland
            {"lat": (49.9, 50.3), "lon": (19.7, 20.3)},
            # Prague Czech Republic
            {"lat": (49.9, 50.3), "lon": (14.2, 14.7)},
            # Budapest Hungary
            {"lat": (47.3, 47.7), "lon": (18.8, 19.3)},
            # Bucharest Romania
            {"lat": (44.2, 44.7), "lon": (25.8, 26.3)},
            # Madrid Spain
            {"lat": (40.2, 40.7), "lon": (-3.9, -3.4)},
            # Barcelona Spain
            {"lat": (41.2, 41.6), "lon": (1.9, 2.4)},
            # Milan Italy
            {"lat": (45.3, 45.7), "lon": (8.9, 9.4)},
            # Rome Italy
            {"lat": (41.6, 42.1), "lon": (12.2, 12.7)},
            # Istanbul Turkey
            {"lat": (40.8, 41.3), "lon": (28.5, 29.5)},
            # Ankara Turkey
            {"lat": (39.7, 40.1), "lon": (32.6, 33.1)},

            # NORTH AMERICA
            # New York City USA
            {"lat": (40.4, 41.0), "lon": (-74.3, -73.7)},
            # Los Angeles USA
            {"lat": (33.7, 34.3), "lon": (-118.7, -117.9)},
            # Chicago USA
            {"lat": (41.6, 42.1), "lon": (-88.0, -87.4)},
            # Houston USA
            {"lat": (29.5, 30.1), "lon": (-95.8, -95.0)},
            # Dallas USA
            {"lat": (32.5, 33.1), "lon": (-97.2, -96.5)},
            # Phoenix USA
            {"lat": (33.2, 33.8), "lon": (-112.5, -111.7)},
            # Philadelphia USA
            {"lat": (39.8, 40.2), "lon": (-75.4, -74.9)},
            # San Antonio USA
            {"lat": (29.2, 29.7), "lon": (-98.7, -98.1)},
            # San Diego USA
            {"lat": (32.5, 33.0), "lon": (-117.5, -116.9)},
            # Seattle USA
            {"lat": (47.4, 47.8), "lon": (-122.6, -122.0)},
            # Atlanta USA
            {"lat": (33.5, 34.0), "lon": (-84.7, -84.1)},
            # Miami USA
            {"lat": (25.5, 26.1), "lon": (-80.5, -80.0)},
            # Toronto Canada
            {"lat": (43.5, 43.9), "lon": (-79.7, -79.1)},
            # Montreal Canada
            {"lat": (45.3, 45.7), "lon": (-74.0, -73.4)},
            # Vancouver Canada
            {"lat": (49.0, 49.4), "lon": (-123.4, -122.8)},
            # Mexico City Mexico
            {"lat": (19.1, 19.7), "lon": (-99.4, -98.8)},
            # Guadalajara Mexico
            {"lat": (20.5, 20.9), "lon": (-103.6, -103.0)},

            # EAST ASIA
            # Tokyo Japan
            {"lat": (35.5, 35.9), "lon": (139.4, 139.9)},
            # Osaka Japan
            {"lat": (34.5, 34.9), "lon": (135.3, 135.8)},
            # Seoul South Korea
            {"lat": (37.4, 37.7), "lon": (126.8, 127.2)},
            # Busan South Korea
            {"lat": (35.0, 35.4), "lon": (128.8, 129.3)},
            # Shanghai China
            {"lat": (31.0, 31.5), "lon": (121.2, 121.7)},
            # Beijing China
            {"lat": (39.7, 40.2), "lon": (116.1, 116.7)},
            # Shenzhen China
            {"lat": (22.4, 22.8), "lon": (113.8, 114.4)},
            # Guangzhou China
            {"lat": (23.0, 23.4), "lon": (113.0, 113.6)},
            # Chongqing China
            {"lat": (29.3, 29.8), "lon": (106.3, 106.9)},
            # Wuhan China
            {"lat": (30.3, 30.8), "lon": (114.0, 114.6)},
            # Taipei Taiwan
            {"lat": (24.9, 25.3), "lon": (121.3, 121.7)},

            # SOUTH ASIA
            # Mumbai India
            {"lat": (18.9, 19.3), "lon": (72.7, 73.1)},
            # Delhi India
            {"lat": (28.4, 29.0), "lon": (76.9, 77.5)},
            # Bangalore India
            {"lat": (12.8, 13.2), "lon": (77.4, 77.8)},
            # Chennai India
            {"lat": (12.9, 13.3), "lon": (80.1, 80.5)},
            # Hyderabad India
            {"lat": (17.2, 17.6), "lon": (78.2, 78.7)},
            # Kolkata India
            {"lat": (22.4, 22.8), "lon": (88.2, 88.6)},
            # Dhaka Bangladesh
            {"lat": (23.6, 24.0), "lon": (90.2, 90.6)},
            # Karachi Pakistan
            {"lat": (24.8, 25.2), "lon": (67.0, 67.5)},
            # Lahore Pakistan
            {"lat": (31.3, 31.7), "lon": (74.1, 74.5)},

            # SOUTHEAST ASIA
            # Bangkok Thailand
            {"lat": (13.5, 14.0), "lon": (100.3, 100.9)},
            # Ho Chi Minh Vietnam
            {"lat": (10.6, 11.0), "lon": (106.5, 106.9)},
            # Hanoi Vietnam
            {"lat": (20.8, 21.2), "lon": (105.6, 106.0)},
            # Jakarta Indonesia
            {"lat": (-6.4, -6.0), "lon": (106.6, 107.0)},
            # Manila Philippines
            {"lat": (14.4, 14.8), "lon": (120.9, 121.3)},
            # Kuala Lumpur Malaysia
            {"lat": (3.0, 3.4), "lon": (101.5, 101.9)},
            # Singapore
            {"lat": (1.1, 1.5), "lon": (103.6, 104.1)},

            # MIDDLE EAST
            # Dubai UAE
            {"lat": (25.0, 25.4), "lon": (55.1, 55.5)},
            # Riyadh Saudi Arabia
            {"lat": (24.5, 25.0), "lon": (46.5, 47.0)},
            # Kuwait City
            {"lat": (29.2, 29.6), "lon": (47.8, 48.2)},
            # Doha Qatar
            {"lat": (25.1, 25.5), "lon": (51.3, 51.7)},
            # Tehran Iran
            {"lat": (35.5, 35.9), "lon": (51.1, 51.7)},
            # Cairo Egypt
            {"lat": (29.9, 30.3), "lon": (31.0, 31.5)},

            # AFRICA
            # Lagos Nigeria
            {"lat": (6.3, 6.7), "lon": (3.1, 3.6)},
            # Nairobi Kenya
            {"lat": (-1.5, -1.1), "lon": (36.6, 37.1)},
            # Johannesburg South Africa
            {"lat": (-26.4, -26.0), "lon": (27.8, 28.3)},
            # Cape Town South Africa
            {"lat": (-34.1, -33.7), "lon": (18.3, 18.8)},
            # Addis Ababa Ethiopia
            {"lat": (8.8, 9.2), "lon": (38.6, 39.0)},
            # Accra Ghana
            {"lat": (5.4, 5.8), "lon": (-0.4, 0.1)},
            # Casablanca Morocco
            {"lat": (33.4, 33.8), "lon": (-7.8, -7.3)},
            # Kinshasa DRC
            {"lat": (-4.5, -4.1), "lon": (15.2, 15.6)},

            # SOUTH AMERICA
            # Sao Paulo Brazil
            {"lat": (-23.8, -23.3), "lon": (-46.9, -46.3)},
            # Rio de Janeiro Brazil
            {"lat": (-23.1, -22.7), "lon": (-43.5, -43.0)},
            # Buenos Aires Argentina
            {"lat": (-34.8, -34.4), "lon": (-58.7, -58.2)},
            # Bogota Colombia
            {"lat": (4.4, 4.8), "lon": (-74.3, -73.9)},
            # Lima Peru
            {"lat": (-12.3, -11.8), "lon": (-77.3, -76.8)},
            # Santiago Chile
            {"lat": (-33.7, -33.3), "lon": (-70.9, -70.5)},

            # AUSTRALIA
            # Sydney Australia
            {"lat": (-34.1, -33.7), "lon": (150.8, 151.4)},
            # Melbourne Australia
            {"lat": (-38.0, -37.6), "lon": (144.7, 145.3)},
            # Brisbane Australia
            {"lat": (-27.7, -27.3), "lon": (152.8, 153.3)},
            # Perth Australia
            {"lat": (-32.1, -31.7), "lon": (115.7, 116.2)},
        ]
    },

    # ── INDUSTRIAL ───────────────────────────────────────────────────────────
    "industrial": {
        "count": 750,
        "regions": [
            # EUROPE — verified power plants + OSM industrial tags
            # Ruhr Germany — densest industrial zone on Earth
            {"lat": (51.3, 51.7), "lon": (6.6, 7.8)},
            # Leuna + Halle Germany — chemical industrial
            {"lat": (51.2, 51.6), "lon": (11.8, 12.4)},
            # Frankfurt chemical — BASF Ludwigshafen
            {"lat": (49.4, 49.7), "lon": (8.3, 8.6)},
            # Hamburg port + industrial
            {"lat": (53.4, 53.6), "lon": (9.8, 10.1)},
            # Antwerp Belgium — largest chemical cluster Europe
            {"lat": (51.1, 51.4), "lon": (4.2, 4.6)},
            # Rotterdam Netherlands — largest port Europe
            {"lat": (51.8, 52.0), "lon": (4.2, 4.6)},
            # Silesia Poland — coal + steel
            {"lat": (50.1, 50.5), "lon": (18.5, 19.5)},
            # Ostrava Czech — steel, coal
            {"lat": (49.7, 50.0), "lon": (18.1, 18.5)},
            # Sheffield UK — steel city
            {"lat": (53.3, 53.6), "lon": (-1.7, -1.2)},
            # Marseille France — refinery, port
            {"lat": (43.2, 43.5), "lon": (5.0, 5.5)},
            # Tarragona Spain — petrochemical
            {"lat": (41.0, 41.3), "lon": (1.0, 1.4)},
            # Po Valley Italy — industrial north
            {"lat": (45.2, 45.6), "lon": (9.8, 10.8)},

            # RUSSIA
            # Yekaterinburg Urals — steel, mining
            {"lat": (56.6, 57.0), "lon": (60.4, 61.0)},
            # Chelyabinsk — steel, zinc
            {"lat": (55.0, 55.4), "lon": (61.1, 61.6)},
            # Magnitogorsk — world's largest steel plant
            {"lat": (53.2, 53.6), "lon": (58.8, 59.3)},
            # Novosibirsk industrial
            {"lat": (54.7, 55.1), "lon": (82.6, 83.2)},
            # Kemerovo Kuzbass — coal mining
            {"lat": (55.1, 55.5), "lon": (85.8, 86.4)},
            # Omsk — oil refinery
            {"lat": (54.8, 55.2), "lon": (73.1, 73.6)},

            # USA
            # Houston Texas — largest US refinery complex
            {"lat": (29.6, 30.0), "lon": (-95.3, -94.8)},
            # Beaumont Port Arthur Texas — refinery cluster
            {"lat": (29.8, 30.2), "lon": (-94.3, -93.8)},
            # Gary Indiana — US Steel, largest steel plant USA
            {"lat": (41.5, 41.7), "lon": (-87.5, -87.2)},
            # Pittsburgh Pennsylvania — historic steel
            {"lat": (40.2, 40.6), "lon": (-80.3, -79.8)},
            # Detroit Michigan — auto manufacturing
            {"lat": (42.1, 42.5), "lon": (-83.4, -82.8)},
            # Midland Texas — Permian Basin oil
            {"lat": (31.7, 32.1), "lon": (-102.3, -101.8)},
            # West Virginia coal
            {"lat": (37.8, 38.3), "lon": (-82.3, -81.5)},
            # Appalachian coal — Kentucky
            {"lat": (37.0, 37.5), "lon": (-83.5, -82.5)},

            # CHINA
            # Tangshan — world's largest steel producing city
            {"lat": (39.4, 39.8), "lon": (118.0, 118.5)},
            # Anshan — historic steel base
            {"lat": (40.9, 41.3), "lon": (122.8, 123.3)},
            # Wuhan steel + auto
            {"lat": (30.3, 30.7), "lon": (114.2, 114.7)},
            # Shenzhen electronics industrial
            {"lat": (22.5, 22.9), "lon": (113.9, 114.4)},
            # Guangzhou industrial
            {"lat": (23.1, 23.5), "lon": (113.2, 113.7)},
            # Dalian port industrial
            {"lat": (38.8, 39.2), "lon": (121.4, 122.0)},
            # Datong coal base
            {"lat": (40.0, 40.4), "lon": (113.0, 113.5)},
            # Urumqi Xinjiang — coal, chemical
            {"lat": (43.6, 44.0), "lon": (87.4, 87.9)},

            # MIDDLE EAST
            # Jubail Saudi Arabia — world's largest industrial city
            {"lat": (27.0, 27.4), "lon": (49.5, 50.0)},
            # Yanbu Saudi Arabia — refinery + petrochemical
            {"lat": (24.0, 24.4), "lon": (38.0, 38.5)},
            # Ras Tanura Saudi — world's largest oil terminal
            {"lat": (26.6, 27.0), "lon": (50.0, 50.4)},
            # Kuwait oil infrastructure
            {"lat": (29.3, 29.7), "lon": (47.6, 48.1)},
            # Abu Dhabi industrial — ICAD
            {"lat": (24.3, 24.7), "lon": (54.4, 54.9)},
            # Abadan Iran — historic refinery
            {"lat": (30.2, 30.6), "lon": (48.2, 48.7)},
            # Isfahan Iran — steel + nuclear
            {"lat": (32.4, 32.8), "lon": (51.4, 51.9)},

            # INDIA
            # Jamnagar Gujarat — world's largest refinery complex
            {"lat": (22.3, 22.6), "lon": (69.9, 70.4)},
            # Surat Gujarat — diamond + textile industrial
            {"lat": (21.1, 21.5), "lon": (72.7, 73.2)},
            # Visakhapatnam — steel, port, refinery
            {"lat": (17.5, 17.9), "lon": (83.1, 83.6)},
            # Rourkela + Jamshedpur — steel plants
            {"lat": (22.0, 22.8), "lon": (84.5, 86.5)},
            # Korba Chhattisgarh — power plant cluster
            {"lat": (22.2, 22.6), "lon": (82.4, 82.9)},
            # Singrauli — world's largest coal power cluster
            {"lat": (24.0, 24.4), "lon": (82.4, 82.9)},

            # AUSTRALIA
            # Port Hedland — world's largest bulk export port
            {"lat": (-20.6, -20.2), "lon": (118.4, 118.9)},
            # Pilbara mining — BHP, Rio Tinto iron ore
            {"lat": (-23.5, -22.5), "lon": (119.5, 120.5)},
            # Hunter Valley coal — NSW
            {"lat": (-32.7, -32.3), "lon": (150.8, 151.3)},
            # Gladstone Queensland — LNG, aluminium
            {"lat": (-24.0, -23.6), "lon": (151.0, 151.5)},

            # SOUTH AMERICA
            # Cubatao Brazil — petrochemical
            {"lat": (-23.9, -23.6), "lon": (-46.6, -46.2)},
            # Camaçari Brazil — largest petrochemical SA
            {"lat": (-12.8, -12.4), "lon": (-38.5, -38.1)},
            # Antofagasta Chile — copper smelting
            {"lat": (-23.8, -23.4), "lon": (-70.6, -70.2)},
            # Huasco Chile — iron, LNG
            {"lat": (-28.7, -28.3), "lon": (-71.4, -71.0)},

            # AFRICA
            # Mpumalanga South Africa — highest power plant density Africa
            {"lat": (-26.4, -25.6), "lon": (29.2, 30.5)},
            # Secunda South Africa — coal to liquids
            {"lat": (-26.7, -26.3), "lon": (29.0, 29.5)},
            # Port Harcourt Nigeria — oil refinery
            {"lat": (4.7, 5.1), "lon": (6.9, 7.3)},
            # Luanda Angola — oil infrastructure
            {"lat": (-8.9, -8.5), "lon": (13.1, 13.5)},
            # Suez Egypt — refinery + industrial
            {"lat": (29.8, 30.2), "lon": (32.4, 32.8)},

            # KAZAKHSTAN + CENTRAL ASIA
            # Tengiz Kazakhstan — supergiant oil field
            {"lat": (45.4, 45.8), "lon": (53.1, 53.6)},
            # Karaganda Kazakhstan — steel, coal
            {"lat": (49.7, 50.1), "lon": (73.0, 73.5)},
            # Aktau Kazakhstan — Caspian oil
            {"lat": (43.5, 43.9), "lon": (51.0, 51.5)},
        ]
    },

    # ── SUBURBAN ─────────────────────────────────────────────────────────────
    "suburban": {
        "count": 750,
        "regions": [
            # USA suburbs — all verified residential satellite cities
            # Dallas-Fort Worth suburbs — Plano, Irving, Arlington
            {"lat": (32.7, 33.2), "lon": (-97.3, -96.6)},
            # Atlanta suburbs — Marietta, Alpharetta, Decatur
            {"lat": (33.7, 34.2), "lon": (-84.8, -84.1)},
            # Phoenix suburbs — Scottsdale, Tempe, Mesa, Gilbert
            {"lat": (33.3, 33.7), "lon": (-112.3, -111.5)},
            # Denver suburbs — Aurora, Lakewood, Englewood
            {"lat": (39.6, 40.1), "lon": (-105.3, -104.6)},
            # Minneapolis suburbs — Bloomington, Eden Prairie
            {"lat": (44.7, 45.1), "lon": (-93.6, -93.0)},
            # Washington DC suburbs — Arlington, Bethesda, Alexandria
            {"lat": (38.7, 39.2), "lon": (-77.4, -76.9)},
            # Miami suburbs — Coral Gables, Hialeah, Pembroke Pines
            {"lat": (25.6, 26.2), "lon": (-80.5, -80.1)},
            # Boston suburbs — Cambridge, Newton, Quincy
            {"lat": (42.1, 42.5), "lon": (-71.4, -70.9)},
            # San Francisco Bay Area suburbs — San Jose, Fremont
            {"lat": (37.2, 37.7), "lon": (-122.2, -121.8)},
            # Portland Oregon suburbs
            {"lat": (45.3, 45.7), "lon": (-122.9, -122.3)},

            # CANADA suburbs
            # Toronto GTA — Mississauga, Brampton, Markham
            {"lat": (43.6, 44.0), "lon": (-80.0, -79.2)},
            # Vancouver suburbs — Surrey, Burnaby, Richmond
            {"lat": (49.0, 49.4), "lon": (-123.2, -122.6)},
            # Calgary suburbs
            {"lat": (50.9, 51.3), "lon": (-114.3, -113.7)},

            # EUROPE suburbs
            # Paris suburbs — Versailles, Boulogne, Saint-Denis
            {"lat": (48.7, 49.1), "lon": (2.1, 2.7)},
            # London suburbs — Croydon, Bromley, Ealing
            {"lat": (51.3, 51.7), "lon": (-0.4, 0.2)},
            # Munich suburbs — Augsburg corridor
            {"lat": (48.0, 48.5), "lon": (11.2, 12.0)},
            # Hamburg suburbs — Pinneberg, Harburg
            {"lat": (53.3, 53.7), "lon": (9.7, 10.3)},
            # Amsterdam suburbs — Haarlem, Almere
            {"lat": (52.3, 52.6), "lon": (4.6, 5.3)},
            # Madrid suburbs — Alcala, Mostoles, Leganes
            {"lat": (40.1, 40.6), "lon": (-3.9, -3.3)},
            # Barcelona suburbs — Badalona, Sabadell, Terrassa
            {"lat": (41.3, 41.7), "lon": (1.9, 2.3)},
            # Milan suburbs — Monza, Bergamo, Brescia
            {"lat": (45.4, 45.8), "lon": (9.1, 10.2)},
            # Stockholm suburbs — Solna, Nacka
            {"lat": (59.2, 59.6), "lon": (17.7, 18.4)},
            # Vienna suburbs — Klosterneuburg, Perchtoldsdorf
            {"lat": (47.9, 48.4), "lon": (16.1, 16.6)},

            # AUSTRALIA suburbs
            # Sydney western suburbs — Parramatta, Blacktown
            {"lat": (-33.9, -33.6), "lon": (150.8, 151.1)},
            # Melbourne eastern suburbs — Box Hill, Doncaster
            {"lat": (-37.9, -37.6), "lon": (145.0, 145.4)},
            # Brisbane suburbs — Ipswich, Logan
            {"lat": (-27.8, -27.3), "lon": (152.7, 153.1)},
            # Perth suburbs — Joondalup, Armadale
            {"lat": (-32.1, -31.6), "lon": (115.7, 116.1)},
            # Adelaide suburbs — Marion, Tea Tree Gully
            {"lat": (-35.1, -34.7), "lon": (138.5, 138.9)},

            # JAPAN suburbs
            # Tokyo suburbs — Yokohama, Kawasaki, Saitama
            {"lat": (35.3, 35.8), "lon": (139.3, 139.9)},
            # Osaka suburbs — Kobe, Kyoto, Nara
            {"lat": (34.7, 35.2), "lon": (135.2, 135.9)},

            # SOUTH KOREA suburbs
            # Seoul suburbs — Suwon, Incheon, Goyang
            {"lat": (37.2, 37.7), "lon": (126.6, 127.2)},

            # INDIA suburbs
            # Mumbai suburbs — Thane, Navi Mumbai
            {"lat": (18.9, 19.3), "lon": (72.9, 73.3)},
            # Delhi NCR suburbs — Gurgaon, Noida, Faridabad
            {"lat": (28.3, 28.8), "lon": (77.0, 77.5)},
            # Bangalore suburbs — Electronic City, Whitefield
            {"lat": (12.7, 13.1), "lon": (77.6, 78.0)},
            # Hyderabad suburbs — Secunderabad, Cyberabad
            {"lat": (17.3, 17.7), "lon": (78.3, 78.8)},

            # SOUTH AMERICA suburbs
            # Buenos Aires GBA — Lomas de Zamora, Quilmes
            {"lat": (-34.8, -34.3), "lon": (-58.6, -58.1)},
            # Sao Paulo ABCD — Santo Andre, Sao Bernardo
            {"lat": (-23.8, -23.3), "lon": (-46.6, -46.1)},
            # Santiago suburbs — Puente Alto, Maipu
            {"lat": (-33.7, -33.3), "lon": (-71.0, -70.5)},
            # Lima suburbs — San Juan de Lurigancho
            {"lat": (-12.1, -11.7), "lon": (-77.1, -76.7)},
            # Bogota suburbs — Soacha, Chia
            {"lat": (4.5, 5.0), "lon": (-74.3, -73.8)},

            # MIDDLE EAST suburbs
            # Dubai suburbs — Sharjah, Ajman
            {"lat": (25.2, 25.6), "lon": (55.3, 55.7)},
            # Riyadh suburbs — Al Kharj corridor
            {"lat": (24.6, 25.1), "lon": (46.7, 47.3)},
            # Cairo suburbs — Giza, Heliopolis, New Cairo
            {"lat": (30.0, 30.4), "lon": (31.0, 31.7)},

            # AFRICA suburbs
            # Johannesburg suburbs — Soweto, Sandton, Pretoria
            {"lat": (-26.1, -25.6), "lon": (27.9, 28.4)},
            # Lagos suburbs — Ikeja, Agege
            {"lat": (6.5, 6.9), "lon": (3.2, 3.6)},
            # Nairobi suburbs — Westlands, Karen, Embakasi
            {"lat": (-1.4, -1.0), "lon": (36.7, 37.1)},
            # Cape Town suburbs — Bellville, Mitchell's Plain
            {"lat": (-34.1, -33.8), "lon": (18.5, 18.9)},
        ]
    },

    # ── AGRICULTURAL ─────────────────────────────────────────────────────────
    # Requirements: OSM landuse=farmland, high NDVI, ERA5 solar radiation,
    # low nighttime lights, low imperviousness
    "agricultural": {
        "count": 500,
        "regions": [
            # US Corn Belt — Iowa, Illinois, Indiana — world's top NDVI
            {"lat": (40.5, 43.5), "lon": (-95.5, -86.5)},
            # US Great Plains — Kansas, Nebraska wheat
            {"lat": (37.5, 41.5), "lon": (-101.5, -96.5)},
            # US Mississippi Delta — cotton, rice
            {"lat": (32.5, 36.5), "lon": (-91.5, -88.5)},
            # US Pacific Northwest — Columbia Basin wheat
            {"lat": (45.5, 47.5), "lon": (-120.5, -117.5)},
            # France — Beauce grain plain (highest agricultural NDVI in Europe)
            {"lat": (47.5, 49.5), "lon": (1.0, 3.5)},
            # Germany Bavaria — mixed farming
            {"lat": (48.0, 49.5), "lon": (10.5, 13.5)},
            # Poland — farming plains
            {"lat": (51.0, 53.0), "lon": (18.5, 22.5)},
            # Ukraine — world's most fertile chernozem soil
            {"lat": (47.5, 51.5), "lon": (29.5, 36.5)},
            # Russia — Krasnodar grain, sunflower
            {"lat": (44.5, 47.5), "lon": (38.5, 44.5)},
            # Kazakhstan steppe wheat
            {"lat": (51.5, 54.5), "lon": (61.5, 73.5)},
            # India Punjab + Haryana — highest agricultural intensity India
            {"lat": (29.5, 32.5), "lon": (74.0, 77.5)},
            # India UP Gangetic plain — wheat, rice
            {"lat": (25.5, 28.5), "lon": (78.5, 83.5)},
            # China Huang-Huai-Hai plain — highest crop intensity China
            {"lat": (33.5, 36.5), "lon": (112.5, 118.5)},
            # China Sichuan basin — rice, vegetables
            {"lat": (29.5, 31.5), "lon": (103.5, 106.5)},
            # Argentina Pampas — world top soy, wheat, corn
            {"lat": (-35.5, -29.5), "lon": (-63.5, -57.5)},
            # Brazil Cerrado — Mato Grosso soy frontier
            {"lat": (-15.5, -11.5), "lon": (-56.5, -50.5)},
            # Brazil Rio Grande do Sul — soy, rice
            {"lat": (-31.5, -28.5), "lon": (-54.5, -50.5)},
            # Australia Darling Downs — wheat, sorghum
            {"lat": (-28.5, -26.5), "lon": (149.5, 152.5)},
            # Australia Riverina — rice, cotton, wheat
            {"lat": (-36.5, -34.5), "lon": (144.5, 147.5)},
            # Sub-Saharan Africa — Nigeria farming belt
            {"lat": (9.5, 12.5), "lon": (6.5, 12.5)},
            # East Africa — Ethiopia highlands farming
            {"lat": (7.5, 11.5), "lon": (37.5, 41.5)},
            # Southeast Asia — Mekong rice bowl Thailand
            {"lat": (15.5, 18.5), "lon": (100.5, 104.5)},
            # Mexico — Bajio breadbasket
            {"lat": (20.5, 22.5), "lon": (-102.5, -99.5)},
        ]
    },

    # ── FOREST ───────────────────────────────────────────────────────────────
    # Requirements: high NDVI (>0.6), high tree cover density,
    # low nighttime lights, low imperviousness
    "forest": {
        "count": 500,
        "regions": [
            # Amazon Brazil — world's highest NDVI values
            {"lat": (-8.5, -2.5), "lon": (-67.5, -52.5)},
            # Amazon Peru + Colombia
            {"lat": (-5.5, 1.5), "lon": (-76.5, -68.5)},
            # Congo Basin DRC — second highest NDVI globally
            {"lat": (-3.5, 3.5), "lon": (17.5, 27.5)},
            # Congo Basin Cameroon + Gabon
            {"lat": (-1.5, 5.5), "lon": (10.5, 16.5)},
            # Borneo — Kalimantan — extreme NDVI
            {"lat": (-2.5, 2.5), "lon": (111.5, 117.5)},
            # Sumatra — high NDVI tropical forest
            {"lat": (-1.5, 3.5), "lon": (101.5, 107.5)},
            # Papua New Guinea — high NDVI
            {"lat": (-8.5, -4.5), "lon": (141.5, 147.5)},
            # West Siberia taiga — boreal forest
            {"lat": (57.5, 63.5), "lon": (68.5, 83.5)},
            # Central Siberia taiga
            {"lat": (57.5, 63.5), "lon": (83.5, 103.5)},
            # East Siberia taiga
            {"lat": (57.5, 63.5), "lon": (103.5, 123.5)},
            # Russian Far East — Primorsky high NDVI
            {"lat": (43.5, 49.5), "lon": (131.5, 137.5)},
            # Boreal Canada — Ontario, Quebec
            {"lat": (48.5, 57.5), "lon": (-87.5, -73.5)},
            # Boreal Canada — BC interior, Alberta
            {"lat": (52.5, 59.5), "lon": (-125.5, -113.5)},
            # Pacific Northwest USA — highest NDVI in continental USA
            {"lat": (44.5, 49.0), "lon": (-124.5, -118.5)},
            # Southeast USA — Appalachian forest
            {"lat": (34.5, 37.5), "lon": (-83.5, -78.5)},
            # Scandinavia — Norway, Sweden, Finland boreal
            {"lat": (60.5, 67.5), "lon": (14.5, 27.5)},
            # Central Europe — Bavarian forest, Black Forest
            {"lat": (47.5, 49.5), "lon": (7.5, 13.5)},
            # Southeast Asia — Myanmar, Laos montane forest
            {"lat": (18.5, 23.5), "lon": (97.5, 102.5)},
            # Madagascar — eastern rainforest
            {"lat": (-20.5, -14.5), "lon": (47.5, 50.5)},
        ]
    },

    # ── COASTAL ──────────────────────────────────────────────────────────────
    # Requirements: OSM coastline, wind resource, Solar Atlas data,
    # ERA5 solar radiation, VIIRS moderate
    "coastal": {
        "count": 500,
        "regions": [
            # North Sea — world's highest offshore wind density
            {"lat": (53.5, 56.5), "lon": (4.5, 8.5)},
            # Danish Straits — major wind zone
            {"lat": (54.5, 57.0), "lon": (8.5, 12.5)},
            # UK west coast — Cornwall, Wales, Scotland wind
            {"lat": (50.5, 58.5), "lon": (-6.5, -2.5)},
            # Norway fjord coast — hydro + wind
            {"lat": (58.5, 62.5), "lon": (4.5, 8.5)},
            # Baltic coast — wind energy development
            {"lat": (54.5, 56.5), "lon": (10.0, 14.5)},
            # Mediterranean Spain — wind + solar coast
            {"lat": (36.5, 40.5), "lon": (-4.5, 2.5)},
            # Mediterranean Italy + Greece
            {"lat": (37.5, 41.5), "lon": (14.5, 24.5)},
            # US Atlantic offshore wind — NY, NJ, MA
            {"lat": (40.5, 42.5), "lon": (-74.5, -69.5)},
            # US Atlantic south — Virginia, North Carolina wind
            {"lat": (35.5, 37.5), "lon": (-76.5, -74.5)},
            # US Pacific — California offshore wind zones
            {"lat": (34.5, 40.5), "lon": (-124.5, -120.5)},
            # US Gulf — Louisiana, Mississippi coast
            {"lat": (28.5, 30.5), "lon": (-91.5, -87.5)},
            # Chile Patagonia — world's strongest sustained winds
            {"lat": (-46.5, -36.5), "lon": (-74.5, -71.5)},
            # Brazil northeast coast — highest wind resource in South America
            {"lat": (-5.5, -2.5), "lon": (-37.5, -34.5)},
            # West Africa — Senegal, Mauritania coast, high wind
            {"lat": (14.5, 20.5), "lon": (-17.5, -14.5)},
            # East Africa Kenya coast — consistent trade winds
            {"lat": (-5.5, -1.5), "lon": (39.5, 41.5)},
            # South Africa west coast — Cape wind zone
            {"lat": (-35.0, -32.5), "lon": (17.5, 19.5)},
            # Australia west — high wind resource
            {"lat": (-35.5, -28.5), "lon": (114.5, 117.5)},
            # Australia south — SA + Victoria coast
            {"lat": (-38.5, -35.5), "lon": (139.5, 147.5)},
            # India Gujarat coast — highest wind resource India
            {"lat": (20.5, 23.5), "lon": (68.5, 72.5)},
            # India Tamil Nadu coast — Muppandal wind farm zone
            {"lat": (8.0, 10.5), "lon": (77.5, 80.5)},
            # Vietnam south coast — high wind, solar hybrid
            {"lat": (10.5, 13.5), "lon": (108.5, 110.5)},
            # Taiwan strait — highest offshore wind Asia
            {"lat": (23.0, 25.5), "lon": (119.5, 121.5)},
            # Japan Pacific coast — wind zone
            {"lat": (39.5, 42.5), "lon": (141.5, 143.5)},
        ]
    },

    # ── ARID ─────────────────────────────────────────────────────────────────
    # Requirements: extreme GHI/DNI/PVOUT values, low NDVI, low nighttime lights
    # These are the world's best solar energy zones
    "arid": {
        "count": 500,
        "regions": [
            # Sahara West — Morocco, Algeria, Mauritania
            # World's highest DNI — 6-8 kWh/m²/day
            {"lat": (20.5, 32.5), "lon": (-10.5, 5.5)},
            # Sahara Central — Algeria, Libya, Niger
            {"lat": (22.5, 30.5), "lon": (5.5, 20.5)},
            # Sahara East — Libya, Egypt, Sudan, Chad
            {"lat": (20.5, 30.5), "lon": (20.5, 32.5)},
            # Arabian Peninsula Saudi interior — extreme DNI
            {"lat": (21.5, 27.5), "lon": (43.5, 54.5)},
            # Oman + Yemen arid interior
            {"lat": (18.5, 23.5), "lon": (54.5, 58.5)},
            # Horn of Africa — Somalia, Ethiopia arid
            {"lat": (8.5, 18.5), "lon": (40.5, 47.5)},
            # Namib desert — extreme aridity
            {"lat": (-24.5, -18.5), "lon": (14.5, 17.5)},
            # Kalahari — Botswana, Namibia
            {"lat": (-25.5, -19.5), "lon": (19.5, 25.5)},
            # Southwest USA Mojave + Sonoran — top US solar zones
            {"lat": (32.5, 37.5), "lon": (-117.5, -109.5)},
            # Atacama — world's driest + highest DNI on Earth
            {"lat": (-26.5, -18.5), "lon": (-70.5, -66.5)},
            # Patagonian steppe — cold arid
            {"lat": (-48.5, -38.5), "lon": (-69.5, -64.5)},
            # Australian outback west — extreme solar
            {"lat": (-30.5, -22.5), "lon": (117.5, 127.5)},
            # Australian outback central — NT, SA
            {"lat": (-28.5, -22.5), "lon": (127.5, 137.5)},
            # Gobi desert Mongolia — extreme continental arid
            {"lat": (41.5, 46.5), "lon": (100.5, 112.5)},
            # Iran plateau — high DNI
            {"lat": (29.5, 35.5), "lon": (53.5, 61.5)},
            # Pakistan Balochistan — high DNI
            {"lat": (26.5, 31.5), "lon": (61.5, 67.5)},
            # Thar desert India — high solar, low NDVI
            {"lat": (25.5, 29.5), "lon": (69.5, 73.5)},
        ]
    },

    # ── ALPINE ───────────────────────────────────────────────────────────────
    # Requirements: high elevation, low temperature, wind resource,
    # low population, OSM elevation tags
    "alpine": {
        "count": 375,
        "regions": [
            # Swiss Alps — highest elevation density in Europe
            {"lat": (46.0, 47.5), "lon": (6.5, 10.5)},
            # Austrian Alps
            {"lat": (46.5, 47.8), "lon": (10.5, 15.5)},
            # French Alps — Grenoble, Chamonix
            {"lat": (44.5, 46.0), "lon": (6.0, 7.5)},
            # Pyrenees — Spain + France border
            {"lat": (42.2, 43.5), "lon": (-1.5, 3.5)},
            # Carpathians — Romania Transylvania
            {"lat": (45.5, 47.5), "lon": (23.5, 26.5)},
            # Caucasus — Georgia, Armenia — high elevation
            {"lat": (41.5, 43.5), "lon": (43.5, 47.5)},
            # Himalayas Nepal — world's highest peaks
            {"lat": (27.5, 29.5), "lon": (83.5, 88.5)},
            # Tibetan Plateau — world's largest high altitude plateau
            {"lat": (30.5, 35.5), "lon": (85.5, 95.5)},
            # Hindu Kush — Afghanistan, Pakistan
            {"lat": (34.5, 37.5), "lon": (69.5, 74.5)},
            # Andes Peru + Bolivia — high altiplano
            {"lat": (-16.5, -10.5), "lon": (-75.5, -68.5)},
            # Andes Chile + Argentina border
            {"lat": (-37.5, -30.5), "lon": (-71.5, -69.5)},
            # Rocky Mountains Colorado — 14ers
            {"lat": (37.5, 40.5), "lon": (-107.5, -104.5)},
            # Rocky Mountains Montana + Wyoming
            {"lat": (44.5, 48.5), "lon": (-114.5, -109.5)},
            # East Africa highlands — Ethiopia, Kenya
            {"lat": (4.5, 9.5), "lon": (37.5, 40.5)},
            # New Zealand Southern Alps
            {"lat": (-45.5, -42.5), "lon": (168.5, 171.5)},
            # Scandinavian mountains — Jotunheimen
            {"lat": (61.5, 65.5), "lon": (13.5, 17.5)},
            # Japan Alps — Honshu mountains
            {"lat": (35.5, 37.5), "lon": (137.5, 138.5)},
        ]
    },

    # ── INFORMAL SETTLEMENTS ─────────────────────────────────────────────────
    # Requirements: high GHSL population density + low VIIRS nighttime lights
    # = energy access story, low imperviousness, sparse OSM tags
    "informal_settlements": {
        "count": 375,
        "regions": [
            # Lagos Nigeria — world's fastest growing megacity
            {"lat": (6.35, 6.75), "lon": (3.1, 3.55)},
            # Lagos outskirts — Agege, Alimosho
            {"lat": (6.55, 6.85), "lon": (3.15, 3.45)},
            # Kinshasa DRC — 17 million people, low nighttime lights
            {"lat": (-4.55, -4.15), "lon": (15.15, 15.55)},
            # Nairobi Kenya — Kibera, Mathare slums
            {"lat": (-1.45, -1.15), "lon": (36.75, 37.05)},
            # Dar es Salaam Tanzania — rapid growth
            {"lat": (-7.05, -6.65), "lon": (39.15, 39.45)},
            # Kampala Uganda — informal periphery
            {"lat": (0.15, 0.45), "lon": (32.45, 32.75)},
            # Addis Ababa Ethiopia — informal settlements
            {"lat": (8.85, 9.25), "lon": (38.65, 38.95)},
            # Khartoum Sudan — large informal areas
            {"lat": (15.35, 15.75), "lon": (32.35, 32.65)},
            # Accra Ghana outskirts
            {"lat": (5.45, 5.75), "lon": (-0.35, 0.05)},
            # Dakar Senegal periphery
            {"lat": (14.65, 14.85), "lon": (-17.55, -17.25)},
            # Mumbai India — Dharavi, Govandi
            {"lat": (19.0, 19.25), "lon": (72.75, 73.05)},
            # Delhi India — outer ring informal
            {"lat": (28.55, 28.85), "lon": (77.05, 77.35)},
            # Dhaka Bangladesh — world's densest city
            {"lat": (23.55, 23.95), "lon": (90.25, 90.55)},
            # Chittagong Bangladesh
            {"lat": (22.25, 22.55), "lon": (91.65, 91.95)},
            # Karachi Pakistan — Orangi town, world's largest informal
            {"lat": (24.85, 25.25), "lon": (67.05, 67.35)},
            # Lahore Pakistan outskirts
            {"lat": (31.35, 31.65), "lon": (74.15, 74.45)},
            # Jakarta Indonesia — informal periphery
            {"lat": (-6.35, -6.05), "lon": (106.75, 107.05)},
            # Manila Philippines — Tondo, Baseco
            {"lat": (14.55, 14.75), "lon": (120.95, 121.15)},
            # Ho Chi Minh Vietnam outskirts
            {"lat": (10.65, 10.95), "lon": (106.55, 106.85)},
            # Lima Peru — Villa El Salvador, Cone Norte
            {"lat": (-12.25, -11.75), "lon": (-77.25, -76.85)},
            # Bogota Colombia — Ciudad Bolivar
            {"lat": (4.45, 4.75), "lon": (-74.35, -74.05)},
            # Caracas Venezuela — barrios
            {"lat": (10.35, 10.65), "lon": (-67.05, -66.75)},
            # Rio de Janeiro Brazil — North Zone favelas
            {"lat": (-22.95, -22.65), "lon": (-43.45, -43.15)},
            # Recife Brazil — informal periphery
            {"lat": (-8.25, -7.85), "lon": (-35.15, -34.85)},
            # Salvador Brazil — outskirts
            {"lat": (-13.05, -12.75), "lon": (-38.65, -38.35)},
        ]
    },

    # ── WATER / WETLAND ──────────────────────────────────────────────────────
    # Requirements: OSM water bodies, high NDVI/EVI, ERA5 humidity,
    # VIIRS low nighttime, hydro energy potential
    "water_wetland": {
        "count": 250,
        "regions": [
            # Mekong Delta Vietnam — world's most productive delta
            {"lat": (9.5, 11.5), "lon": (104.5, 106.5)},
            # Ganges-Brahmaputra Delta Bangladesh — world's largest delta
            {"lat": (22.5, 24.5), "lon": (88.5, 91.5)},
            # Irrawaddy Delta Myanmar
            {"lat": (15.5, 17.5), "lon": (95.0, 97.5)},
            # Nile Delta Egypt — high agricultural water use
            {"lat": (30.5, 31.5), "lon": (29.5, 32.5)},
            # Niger Delta Nigeria — oil + wetland
            {"lat": (4.5, 5.8), "lon": (5.8, 7.8)},
            # Congo River basin wetlands
            {"lat": (-2.5, 2.5), "lon": (17.5, 22.5)},
            # Sudd wetland South Sudan — world's largest tropical wetland
            {"lat": (6.5, 9.5), "lon": (29.5, 32.5)},
            # Okavango Delta Botswana — endorheic delta
            {"lat": (-20.5, -18.5), "lon": (22.5, 24.5)},
            # Zambezi floodplain Zambia
            {"lat": (-16.5, -14.5), "lon": (22.5, 25.5)},
            # Lake Victoria basin — Uganda, Kenya, Tanzania
            {"lat": (-1.5, 1.5), "lon": (31.5, 34.5)},
            # Mississippi Delta Louisiana
            {"lat": (28.5, 30.5), "lon": (-91.5, -88.5)},
            # Everglades Florida — unique subtropical wetland
            {"lat": (25.0, 26.5), "lon": (-81.5, -80.5)},
            # Amazon floodplain — varzea forest
            {"lat": (-4.5, 0.5), "lon": (-62.5, -52.5)},
            # Pantanal Brazil + Bolivia — world's largest tropical wetland
            {"lat": (-20.5, -16.5), "lon": (-58.5, -54.5)},
            # Ob-Irtysh basin Russia — massive boreal wetland
            {"lat": (59.5, 64.5), "lon": (67.5, 75.5)},
            # Finnish lake district — high water body density
            {"lat": (61.5, 64.5), "lon": (25.5, 30.5)},
            # Canadian Shield lakes — Ontario, Quebec
            {"lat": (46.5, 52.5), "lon": (-82.5, -74.5)},
            # Murray-Darling basin Australia — major river system
            {"lat": (-36.5, -33.5), "lon": (142.5, 146.5)},
        ]
    }
}


def generate_bbox(lat, lon, size_m=256):
    delta_lat = (size_m / 2) / 111320
    delta_lon = (size_m / 2) / (111320 * np.cos(np.radians(lat)))
    return {
        "min_lat": lat - delta_lat,
        "max_lat": lat + delta_lat,
        "min_lon": lon - delta_lon,
        "max_lon": lon + delta_lon
    }


def is_too_close(lat, lon, existing_points, min_spacing_km=1.0):
    if len(existing_points) == 0:
        return False
    for existing_lat, existing_lon in existing_points:
        dlat = abs(lat - existing_lat) * 111.32
        dlon = abs(lon - existing_lon) * 111.32 * np.cos(np.radians(lat))
        distance = np.sqrt(dlat**2 + dlon**2)
        if distance < min_spacing_km:
            return True
    return False


def sample_stratum(stratum_name, stratum_config, existing_points, min_spacing_km=1.0):
    count = stratum_config["count"]
    regions = stratum_config["regions"]
    points = []
    max_attempts = count * 150
    attempts = 0

    while len(points) < count and attempts < max_attempts:
        region = regions[np.random.randint(len(regions))]
        lat = np.random.uniform(region["lat"][0], region["lat"][1])
        lon = np.random.uniform(region["lon"][0], region["lon"][1])

        all_points = existing_points + [(p["lat"], p["lon"]) for p in points]

        if not is_too_close(lat, lon, all_points, min_spacing_km):
            if is_on_land(lat, lon):
                bbox = generate_bbox(lat, lon)
                points.append({
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "stratum": stratum_name,
                    "bbox_min_lat": round(bbox["min_lat"], 6),
                    "bbox_max_lat": round(bbox["max_lat"], 6),
                    "bbox_min_lon": round(bbox["min_lon"], 6),
                    "bbox_max_lon": round(bbox["max_lon"], 6),
                })
        attempts += 1

    if len(points) < count:
        logger.warning(
            f"{stratum_name}: Only {len(points)}/{count} points — add more regions"
        )
    else:
        logger.info(f"{stratum_name}: {len(points)}/{count} points generated")

    return points


def generate_all_coordinates():
    all_points = []
    existing_points = []

    for stratum_name, stratum_config in STRATA.items():
        logger.info(f"Sampling stratum: {stratum_name}")
        points = sample_stratum(stratum_name, stratum_config, existing_points)
        all_points.extend(points)
        existing_points.extend([(p["lat"], p["lon"]) for p in points])

    return all_points


def save_csv(points, output_path):
    df = pd.DataFrame(points)
    df.insert(0, "id", range(1, len(df) + 1))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} coordinates to {output_path}")
    return df


def plot_world_map(df, output_path):
    m = folium.Map(location=[20, 0], zoom_start=2)

    colors = {
        "dense_urban": "red",
        "industrial": "darkred",
        "suburban": "orange",
        "agricultural": "green",
        "forest": "darkgreen",
        "coastal": "blue",
        "arid": "cadetblue",
        "alpine": "purple",
        "informal_settlements": "pink",
        "water_wetland": "darkblue"
    }

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color=colors.get(row["stratum"], "gray"),
            fill=True,
            fill_opacity=0.8,
            popup=f"{row['stratum']} | {row['lat']}, {row['lon']}"
        ).add_to(m)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    logger.info(f"Map saved to {output_path}")


if __name__ == "__main__":
    logger.info("Starting stratified sampling...")

    points = generate_all_coordinates()

    df = save_csv(
        points,
        "/srv/THESIS/energy_profiling_thesis/data/coordinates/coordinates_5000.csv"
    )

    plot_world_map(
        df,
        "/srv/THESIS/energy_profiling_thesis/outputs/maps/world_map_5000.html"
    )

    print("\n=== Sampling Summary ===")
    print(f"Total coordinates: {len(df)}")
    print("\nPer stratum:")
    print(df["stratum"].value_counts())
    print("\nSample rows:")
    print(df.head(10))