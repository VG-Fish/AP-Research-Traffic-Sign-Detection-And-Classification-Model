from pathlib import Path
from json import load

directory = "mtsd_v2_fully_annotated/annotations"
files = Path(directory).glob("*.json")
sizes = set()

for file in files:
    with open(file, "r") as f:
        data = load(f)
        if data["width"] == 320:
            print(file)
        sizes.add((data["width"], data["height"]))

"""
{(320, 240),
 (480, 320),
 (480, 640),
 (640, 350),
 (640, 360),
 (640, 362),
 (640, 365),
 (640, 420),
 (640, 445),
 (640, 480),
 (720, 480),
 (720, 1280),
 (750, 663),
 (757, 685),
 (766, 689),
 (768, 1024),
 (800, 450),
 (800, 480),
 (800, 600),
 (848, 480),
 (864, 1152),
 (900, 621),
 (900, 675),
 (960, 720),
 (1000, 515),
 (1024, 768),
 (1037, 778),
 (1080, 1260),
 (1088, 816),
 (1088, 1088),
 (1089, 785),
 (1094, 618),
 (1152, 864),
 (1200, 1600),
 (1278, 720),
 (1280, 500),
 (1280, 530),
 (1280, 545),
 (1280, 552),
 (1280, 584),
 (1280, 608),
 (1280, 720),
 (1280, 768),
 (1280, 960),
 (1313, 1750),
 (1360, 1024),
 (1400, 600),
 (1400, 806),
 (1434, 1200),
 (1440, 720),
 (1440, 1080),
 (1521, 1140),
 (1534, 1151),
 (1550, 880),
 (1576, 1180),
 (1600, 1067),
 (1600, 1071),
 (1600, 1184),
 (1600, 1200),
 (1620, 850),
 (1624, 1234),
 (1632, 1224),
 (1728, 1728),
 (1728, 2880),
 (1800, 1223),
 (1800, 1350),
 (1836, 3264),
 (1920, 870),
 (1920, 900),
 (1920, 930),
 (1920, 960),
 (1920, 982),
 (1920, 1040),
 (1920, 1080),
 (1920, 1280),
 (1920, 1440),
 (1920, 1920),
 (1920, 2560),
 (1936, 2592),
 (1944, 2592),
 (1951, 1080),
 (2000, 1333),
 (2000, 1500),
 (2013, 1342),
 (2048, 1024),
 (2048, 1094),
 (2048, 1152),
 (2048, 1232),
 (2048, 1321),
 (2048, 1520),
 (2048, 1536),
 (2081, 1245),
 (2100, 1156),
 (2100, 1169),
 (2100, 1191),
 (2100, 1313),
 (2160, 3840),
 (2200, 1289),
 (2240, 2240),
 (2267, 1700),
 (2304, 1296),
 (2304, 1536),
 (2304, 1728),
 (2322, 4128),
 (2334, 1618),
 (2336, 1752),
 (2341, 1757),
 (2343, 1482),
 (2367, 1121),
 (2400, 1265),
 (2400, 1800),
 (2448, 1836),
 (2448, 2048),
 (2448, 2050),
 (2448, 2448),
 (2448, 3264),
 (2468, 1388),
 (2560, 1080),
 (2560, 1280),
 (2560, 1440),
 (2560, 1536),
 (2560, 1700),
 (2560, 1780),
 (2560, 1920),
 (2576, 1932),
 (2589, 1940),
 (2592, 1456),
 (2592, 1458),
 (2592, 1480),
 (2592, 1552),
 (2592, 1558),
 (2592, 1728),
 (2592, 1737),
 (2592, 1936),
 (2592, 1944),
 (2592, 1952),
 (2600, 1625),
 (2600, 1950),
 (2604, 1738),
 (2608, 1960),
 (2624, 1968),
 (2652, 3976),
 (2666, 2000),
 (2688, 1512),
 (2688, 1520),
 (2700, 1777),
 (2704, 1478),
 (2704, 2028),
 (2720, 2040),
 (2736, 1824),
 (2736, 2052),
 (2740, 2055),
 (2816, 2112),
 (2820, 1410),
 (2848, 2136),
 (2873, 2150),
 (2880, 1728),
 (2880, 2160),
 (2933, 2200),
 (2934, 2200),
 (2972, 2229),
 (2976, 1680),
 (2976, 2976),
 (2988, 3984),
 (2988, 5312),
 (3000, 1426),
 (3000, 1875),
 (3000, 2000),
 (3000, 2246),
 (3000, 2250),
 (3000, 3000),
 (3000, 4000),
 (3005, 1745),
 (3008, 1692),
 (3008, 2000),
 (3008, 2256),
 (3024, 2016),
 (3024, 3024),
 (3024, 4032),
 (3052, 2289),
 (3056, 2292),
 (3068, 1764),
 (3072, 1728),
 (3072, 2304),
 (3075, 2306),
 (3088, 3088),
 (3096, 4128),
 (3104, 1744),
 (3104, 1746),
 (3104, 3000),
 (3104, 3104),
 (3120, 3120),
 (3120, 4160),
 (3120, 4208),
 (3153, 2365),
 (3188, 1899),
 (3200, 1800),
 (3200, 2368),
 (3200, 2400),
 (3204, 1917),
 (3216, 1984),
 (3216, 2144),
 (3232, 2424),
 (3250, 1890),
 (3264, 1410),
 (3264, 1420),
 (3264, 1440),
 (3264, 1508),
 (3264, 1650),
 (3264, 1756),
 (3264, 1824),
 (3264, 1836),
 (3264, 1840),
 (3264, 1968),
 (3264, 2176),
 (3264, 2300),
 (3264, 2448),
 (3280, 1830),
 (3280, 2464),
 (3291, 2465),
 (3296, 2472),
 (3314, 2475),
 (3317, 2480),
 (3328, 1872),
 (3333, 2500),
 (3346, 2500),
 (3360, 2240),
 (3384, 1900),
 (3390, 2000),
 (3400, 1800),
 (3440, 1835),
 (3450, 1940),
 (3456, 2304),
 (3456, 2592),
 (3456, 3456),
 (3456, 4608),
 (3456, 5184),
 (3477, 1950),
 (3488, 1962),
 (3500, 1950),
 (3500, 1969),
 (3552, 2000),
 (3561, 2671),
 (3561, 2750),
 (3570, 1930),
 (3580, 1810),
 (3584, 2016),
 (3600, 2160),
 (3633, 2725),
 (3648, 2059),
 (3648, 2206),
 (3648, 2736),
 (3664, 2748),
 (3680, 2208),
 (3680, 2760),
 (3684, 2851),
 (3754, 2474),
 (3764, 1882),
 (3789, 2592),
 (3790, 2373),
 (3840, 1525),
 (3840, 1538),
 (3840, 1600),
 (3840, 1700),
 (3840, 1740),
 (3840, 1770),
 (3840, 1780),
 (3840, 1800),
 (3840, 1820),
 (3840, 1825),
 (3840, 1830),
 (3840, 1855),
 (3840, 1920),
 (3840, 2045),
 (3840, 2160),
 (3840, 2304),
 (3840, 2880),
 (3850, 2166),
 (3860, 2895),
 (3872, 2592),
 (3888, 1944),
 (3888, 2000),
 (3888, 2592),
 (3920, 2160),
 (3920, 2204),
 (3928, 2944),
 (3934, 2394),
 (3936, 2624),
 (3936, 5248),
 (3968, 2240),
 (3968, 2976),
 (3976, 2652),
 (3984, 2988),
 (4000, 1850),
 (4000, 1950),
 (4000, 2000),
 (4000, 2090),
 (4000, 2100),
 (4000, 2150),
 (4000, 2200),
 (4000, 2240),
 (4000, 2250),
 (4000, 2256),
 (4000, 2425),
 (4000, 2500),
 (4000, 2743),
 (4000, 3000),
 (4000, 3008),
 (4008, 2672),
 (4018, 3015),
 (4032, 2268),
 (4032, 2272),
 (4032, 2474),
 (4032, 2637),
 (4032, 2704),
 (4032, 2728),
 (4032, 2758),
 (4032, 2840),
 (4032, 2933),
 (4032, 3016),
 (4032, 3024),
 (4048, 3036),
 (4056, 3040),
 (4079, 1901),
 (4096, 2048),
 (4096, 2160),
 (4096, 2304),
 (4096, 3072),
 (4128, 2322),
 (4128, 2332),
 (4128, 2487),
 (4128, 2500),
 (4128, 2518),
 (4128, 2690),
 (4128, 2752),
 (4128, 3096),
 (4128, 3104),
 (4160, 2336),
 (4160, 2340),
 (4160, 2420),
 (4160, 2535),
 (4160, 2603),
 (4160, 3120),
 (4192, 3104),
 (4208, 3120),
 (4224, 3136),
 (4224, 3168),
 (4259, 2397),
 (4264, 2570),
 (4288, 2848),
 (4288, 3216),
 (4296, 2417),
 (4320, 2432),
 (4352, 3264),
 (4357, 2451),
 (4378, 2919),
 (4464, 2976),
 (4496, 3000),
 (4592, 3056),
 (4604, 2687),
 (4608, 1540),
 (4608, 1624),
 (4608, 1752),
 (4608, 1960),
 (4608, 1986),
 (4608, 2042),
 (4608, 2112),
 (4608, 2236),
 (4608, 2316),
 (4608, 2376),
 (4608, 2410),
 (4608, 2444),
 (4608, 2592),
 (4608, 2602),
 (4608, 2621),
 (4608, 2780),
 (4608, 2860),
 (4608, 3072),
 (4608, 3252),
 (4608, 3456),
 (4624, 2608),
 (4624, 3468),
 (4624, 3488),
 (4632, 3474),
 (4640, 2610),
 (4640, 3480),
 (4656, 2620),
 (4656, 3488),
 (4656, 3492),
 (4672, 3504),
 (4864, 2736),
 (4896, 3672),
 (4912, 3264),
 (4992, 3744),
 (5049, 3768),
 (5120, 3840),
 (5152, 2896),
 (5152, 3864),
 (5152, 3888),
 (5184, 2658),
 (5184, 2916),
 (5184, 3456),
 (5184, 3888),
 (5248, 2952),
 (5248, 2960),
 (5248, 3936),
 (5312, 2988),
 (5344, 3006),
 (5344, 3008),
 (5344, 4008),
 (5376, 2688),
 (5376, 3024),
 (5376, 3744),
 (5376, 3752),
 (5400, 2700),
 (5472, 2736),
 (5472, 3078),
 (5472, 3648),
 (5632, 4224),
 (5640, 2816),
 (5640, 2820),
 (5660, 2830),
 (5664, 3184),
 (5664, 4240),
 (5702, 3791),
 (5752, 3892),
 (5760, 2880),
 (5888, 3312),
 (5952, 3348),
 (6400, 3200),
 (6720, 4480),
 (6904, 3448),
 (6912, 3456),
 (6934, 2942),
 (7200, 3600),
 (7510, 2179),
 (7582, 2178),
 (7680, 3840),
 (7776, 3888),
 (7940, 3860),
 (8000, 4000),
 (8192, 4096),
 (8704, 4352),
 (9728, 4864),
 (10000, 5000),
 (10394, 5197),
 (10408, 5204),
 (10410, 5205),
 (10416, 5208),
 (10444, 5222),
 (10448, 5224),
 (10484, 5242),
 (11292, 5646),
 (11304, 5652),
 (11616, 5808),
 (12656, 6328),
 (13308, 6654),
 (13312, 3252),
 (22062, 2520)}
"""