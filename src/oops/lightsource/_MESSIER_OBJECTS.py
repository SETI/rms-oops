# All 110 Messier objects, in Messier-number order.
#
# Columns: (formal name, common name or None, RA [deg], Dec [deg], object type,
#   V mag, magnitude type ('V'), pm RA * cos(Dec) [mas/yr], pm Dec [mas/yr],
#   parallax [mas])
#
# Positions: ICRS / J2000.0, from the OpenNGC database, which takes them mostly from NED
#   and SIMBAD. For extended objects this is a conventional centre, not a point;
#   M40 and M45 are given only to ~0.1s / 1" in the source. Given here to 1e-6 deg.
# Object types: OpenNGC classification in words; galaxies carry OpenNGC's Hubble type.
#   M8, M16, M17, M20, M42 and M43 are listed as emission nebulae, though most also contain
#   young clusters; M24 is a Milky Way star cloud, M40 a double star, M73 an asterism.
# Magnitudes: integrated V. The 29 globular clusters use Harris (1996, 2010 edition;
#   CDS VII/202); everything else uses OpenNGC, compiled from heterogeneous sources.
#   For large or diffuse objects these are rough (+/- 0.5 mag or worse), and the light
#   is spread over the object's whole area, so it looks much fainter than a star of the
#   same magnitude.
# Proper motions and parallaxes:
#   - Clusters: the Gaia DR3 mean of the members, from Hunt & Reffert (2023, CDS
#     J/A+A/673/A114), matched by name: all open and globular clusters, plus the clusters
#     within M8 (NGC 6530), M16 (NGC 6611) and M78 (NGC 2068). M39 is their Theia 517
#     and M45 their Melotte 22.
#   - M1 (the Crab pulsar) and the planetary nebulae M27, M57, M76 and M97 (their central
#     stars): Gaia DR3 (CDS I/355). The parallax of M76's central star is poor
#     (0.29 +/- 0.20 mas).
#   - Zero for galaxies, whose motions and parallaxes are immeasurably small; for M17,
#     M20, M42 and M43, which have no cluster in Hunt & Reffert that can be identified
#     with them; and for M24, M40 and M73, which are not physical groups.
#   The proper motion in RA is mu_alpha* = d(RA)/dt * cos(Dec), as catalogs
#   conventionally give it.
# M102: its identity is disputed. Listed here as NGC 5866, the usual modern choice;
#   OpenNGC (following NED) treats it as a duplicate observation of M101.
# Formal names: Messier number spelled out.
# Common names: widely used names, mostly as listed in OpenNGC; None where no name is in
#   common use.
#
# Sources: OpenNGC (CC BY-SA 4.0), github.com/mattiaverga/OpenNGC;
#   Harris globular cluster catalog via CDS; Hunt & Reffert (2023) cluster catalog and
#   Gaia DR3 via CDS.

_MESSIER_OBJECTS = [
    ('Messier 1',                 'Crab Nebula',                83.633208, +22.014472, 'Supernova remnant', +8.40, 'V',   -11.513,    +2.302,  +0.511),
    ('Messier 2',                 None,                        323.362542,  -0.823306, 'Globular cluster',  +6.47, 'V',    +3.470,    -2.184,  +0.025),
    ('Messier 3',                 None,                        205.546792, +28.375444, 'Globular cluster',  +6.19, 'V',    -0.148,    -2.671,  +0.095),
    ('Messier 4',                 None,                        245.897500, -26.525528, 'Globular cluster',  +5.63, 'V',   -12.515,   -19.014,  +0.526),
    ('Messier 5',                 None,                        229.640625,  +2.082694, 'Globular cluster',  +5.65, 'V',    +4.068,    -9.862,  +0.117),
    ('Messier 6',                 'Butterfly Cluster',         265.086458, -32.254167, 'Open cluster',      +4.20, 'V',    -1.368,    -5.847,  +2.177),
    ('Messier 7',                 'Ptolemy Cluster',           268.463250, -34.792833, 'Open cluster',      +3.30, 'V',    +3.080,    -5.366,  +3.583),
    ('Messier 8',                 'Lagoon Nebula',             270.921958, -24.380167, 'Emission nebula',   +5.80, 'V',    +1.333,    -2.058,  +0.764),
    ('Messier 9',                 None,                        259.799083, -18.516250, 'Globular cluster',  +7.72, 'V',    -2.175,    -3.216,  +0.102),
    ('Messier 10',                None,                        254.287458,  -4.099333, 'Globular cluster',  +6.60, 'V',    -4.762,    -6.611,  +0.164),
    ('Messier 11',                'Wild Duck Cluster',         282.774958,  -6.270028, 'Open cluster',      +5.80, 'V',    -1.536,    -4.166,  +0.411),
    ('Messier 12',                None,                        251.810500,  -1.947833, 'Globular cluster',  +6.70, 'V',    -0.200,    -6.807,  +0.182),
    ('Messier 13',                'Hercules Globular Cluster', 250.423458, +36.461306, 'Globular cluster',  +5.78, 'V',    -3.139,    -2.571,  +0.111),
    ('Messier 14',                None,                        264.400667,  -3.245917, 'Globular cluster',  +7.59, 'V',    -3.585,    -5.059,  +0.087),
    ('Messier 15',                None,                        322.493250, +12.166833, 'Globular cluster',  +6.20, 'V',    -0.642,    -3.813,  +0.066),
    ('Messier 16',                'Eagle Nebula',              274.700708, -13.807222, 'Emission nebula',   +6.00, 'V',    +0.228,    -1.608,  +0.540),
    ('Messier 17',                'Omega Nebula',              275.196292, -16.171528, 'Emission nebula',   +7.00, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 18',                None,                        274.993708, -17.101972, 'Open cluster',      +6.90, 'V',    -0.458,    -0.893,  +0.627),
    ('Messier 19',                None,                        255.657000, -26.267944, 'Globular cluster',  +6.77, 'V',    -3.258,    +1.676,  +0.095),
    ('Messier 20',                'Trifid Nebula',             270.675458, -22.971889, 'Emission nebula',   +8.50, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 21',                None,                        271.056042, -22.490056, 'Open cluster',      +5.90, 'V',    +0.538,    -1.464,  +0.801),
    ('Messier 22',                None,                        279.100833, -23.903417, 'Globular cluster',  +5.10, 'V',    +9.837,    -5.605,  +0.284),
    ('Messier 23',                None,                        269.269875, -18.985333, 'Open cluster',      +5.50, 'V',    +0.314,    -1.824,  +1.368),
    ('Messier 24',                'Sagittarius Star Cloud',    274.233833, -18.514556, 'Star cloud',        +4.50, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 25',                None,                        277.944875, -19.114944, 'Open cluster',      +4.60, 'V',    -1.692,    -6.165,  +1.512),
    ('Messier 26',                None,                        281.327750,  -9.383611, 'Open cluster',      +8.87, 'V',    +0.143,    -1.425,  +0.507),
    ('Messier 27',                'Dumbbell Nebula',           299.901583, +22.721028, 'Planetary nebula',  +7.40, 'V',   +10.520,    +3.699,  +2.570),
    ('Messier 28',                None,                        276.137042, -24.869833, 'Globular cluster',  +6.79, 'V',    -0.288,    -8.965,  +0.156),
    ('Messier 29',                None,                        305.990708, +38.507667, 'Open cluster',      +6.60, 'V',    -3.406,    -5.810,  +0.562),
    ('Messier 30',                None,                        325.091750, -23.179083, 'Globular cluster',  +7.19, 'V',    -0.736,    -7.295,  +0.102),
    ('Messier 31',                'Andromeda Galaxy',           10.684792, +41.269056, 'Galaxy Sb',         +3.44, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 32',                None,                         10.674292, +40.865278, 'Galaxy E',          +8.13, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 33',                'Triangulum Galaxy',          23.462042, +30.660222, 'Galaxy Sc',         +5.79, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 34',                None,                         40.530833, +42.746139, 'Open cluster',      +5.20, 'V',    +0.652,    -5.791,  +2.003),
    ('Messier 35',                None,                         92.271083, +24.338639, 'Open cluster',      +5.10, 'V',    +2.253,    -2.895,  +1.154),
    ('Messier 36',                None,                         84.073917, +34.140750, 'Open cluster',      +6.00, 'V',    -0.218,    -3.414,  +0.843),
    ('Messier 37',                None,                         88.076458, +32.553000, 'Open cluster',      +5.60, 'V',    +1.881,    -5.624,  +0.673),
    ('Messier 38',                None,                         82.177042, +35.854917, 'Open cluster',      +6.40, 'V',    +1.545,    -4.430,  +0.876),
    ('Messier 39',                None,                        322.951333, +48.438167, 'Open cluster',      +4.60, 'V',    -7.400,   -19.704,  +3.366),
    ('Messier 40',                None,                        185.567083, +58.084444, 'Double star',       +8.00, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 41',                None,                        101.499750, -20.754222, 'Open cluster',      +4.50, 'V',    -4.369,    -1.362,  +1.363),
    ('Messier 42',                'Orion Nebula',               83.818667,  -5.389667, 'Emission nebula',   +4.00, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 43',                "De Mairan's Nebula",         83.880750,  -5.267472, 'Emission nebula',   +9.00, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 44',                'Beehive Cluster',           130.092500, +19.672056, 'Open cluster',      +3.10, 'V',   -35.943,   -12.903,  +5.404),
    ('Messier 45',                'Pleiades',                   56.869167, +24.105278, 'Open cluster',      +1.20, 'V',   +19.955,   -45.457,  +7.378),
    ('Messier 46',                None,                        115.445083, -14.810000, 'Open cluster',      +6.10, 'V',    -3.871,    +0.407,  +0.599),
    ('Messier 47',                None,                        114.145917, -14.482611, 'Open cluster',      +4.40, 'V',    -7.069,    +1.042,  +2.100),
    ('Messier 48',                None,                        123.429917,  -5.750444, 'Open cluster',      +5.80, 'V',    -1.294,    +1.037,  +1.291),
    ('Messier 49',                None,                        187.444833,  +8.000472, 'Galaxy E',          +8.28, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 50',                None,                        105.668625,  -8.364028, 'Open cluster',      +5.90, 'V',    -0.717,    -0.602,  +1.006),
    ('Messier 51',                'Whirlpool Galaxy',          202.469625, +47.195167, 'Galaxy SABb',       +8.36, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 52',                None,                        351.201667, +61.593167, 'Open cluster',      +6.90, 'V',    -1.895,    -1.191,  +0.606),
    ('Messier 53',                None,                        198.230125, +18.169111, 'Globular cluster',  +7.61, 'V',    -0.150,    -1.332,  +0.040),
    ('Messier 54',                None,                        283.763625, -30.478500, 'Globular cluster',  +7.60, 'V',    -2.675,    -1.386,  +0.013),
    ('Messier 55',                None,                        294.997500, -30.962083, 'Globular cluster',  +6.32, 'V',    -3.431,    -9.321,  +0.182),
    ('Messier 56',                None,                        289.147958, +30.184500, 'Globular cluster',  +8.27, 'V',    -2.019,    +1.615,  +0.066),
    ('Messier 57',                'Ring Nebula',               283.395875, +33.028583, 'Planetary nebula',  +8.80, 'V',    +1.645,    +2.472,  +1.270),
    ('Messier 58',                None,                        189.431333, +11.818194, 'Galaxy Sb',        +10.30, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 59',                None,                        190.509333, +11.647028, 'Galaxy E',          +9.56, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 60',                None,                        190.916583, +11.552694, 'Galaxy E',          +8.79, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 61',                None,                        185.478750,  +4.473639, 'Galaxy Sbc',       +10.25, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 62',                None,                        255.302500, -30.112361, 'Globular cluster',  +6.45, 'V',    -4.966,    -2.946,  +0.147),
    ('Messier 63',                'Sunflower Galaxy',          198.955542, +42.029278, 'Galaxy Sbc',        +8.61, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 64',                'Black Eye Galaxy',          194.181833, +21.682972, 'Galaxy SABa',       +8.52, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 65',                None,                        169.733000, +13.092361, 'Galaxy Sa',         +9.32, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 66',                None,                        170.062333, +12.991528, 'Galaxy Sb',         +8.92, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 67',                None,                        132.833875, +11.811944, 'Open cluster',      +6.90, 'V',   -10.965,    -2.906,  +1.150),
    ('Messier 68',                None,                        189.866708, -26.743028, 'Globular cluster',  +7.84, 'V',    -2.742,    +1.775,  +0.083),
    ('Messier 69',                None,                        277.846792, -32.347972, 'Globular cluster',  +7.64, 'V',    -5.058,    -5.830,  +0.084),
    ('Messier 70',                None,                        280.802667, -32.291889, 'Globular cluster',  +7.87, 'V',    +1.451,    -4.738,  +0.086),
    ('Messier 71',                None,                        298.442125, +18.778389, 'Globular cluster',  +8.19, 'V',    -3.412,    -2.660,  +0.226),
    ('Messier 72',                None,                        313.366292, -12.537056, 'Globular cluster',  +9.27, 'V',    -1.274,    -3.367,  +0.043),
    ('Messier 73',                None,                        314.733208, -12.635500, 'Asterism',          +8.90, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 74',                None,                         24.173958, +15.783667, 'Galaxy Sc',         +9.31, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 75',                None,                        301.520167, -21.922222, 'Globular cluster',  +8.52, 'V',    -0.592,    -2.809,  +0.037),
    ('Messier 76',                'Little Dumbbell Nebula',     25.582042, +51.575472, 'Planetary nebula', +10.10, 'V',    +2.586,    -2.959,  +0.294),
    ('Messier 77',                None,                         40.669625,  -0.013278, 'Galaxy Sb',         +9.29, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 78',                None,                         86.690917,  +0.079306, 'Reflection nebula', +8.00, 'V',    -0.444,    -0.738,  +2.419),
    ('Messier 79',                None,                         81.044125, -24.524222, 'Globular cluster',  +7.73, 'V',    +2.464,    -1.605,  +0.051),
    ('Messier 80',                None,                        244.260458, -22.975111, 'Globular cluster',  +7.33, 'V',    -2.939,    -5.580,  +0.054),
    ('Messier 81',                "Bode's Galaxy",             148.888208, +69.065306, 'Galaxy Sab',        +6.92, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 82',                'Cigar Galaxy',              148.969708, +69.679389, 'Galaxy S?',         +8.30, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 83',                'Southern Pinwheel Galaxy',  204.253958, -29.865417, 'Galaxy Sc',         +7.21, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 84',                None,                        186.265583, +12.886972, 'Galaxy E',          +9.79, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 85',                None,                        186.350458, +18.191500, 'Galaxy S0-a',       +9.05, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 86',                None,                        186.548917, +12.946222, 'Galaxy E',          +8.86, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 87',                'Virgo A',                   187.705917, +12.391111, 'Galaxy E',          +9.00, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 88',                None,                        187.996500, +14.420389, 'Galaxy Sb',        +10.33, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 89',                None,                        188.915875, +12.556333, 'Galaxy E',         +10.08, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 90',                None,                        189.207458, +13.162944, 'Galaxy Sab',        +9.54, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 91',                None,                        188.860208, +14.496333, 'Galaxy Sb',        +10.96, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 92',                None,                        259.280292, +43.136528, 'Globular cluster',  +6.44, 'V',    -4.931,    -0.636,  +0.100),
    ('Messier 93',                None,                        116.121792, -23.853083, 'Open cluster',      +6.20, 'V',    -3.560,    +5.086,  +0.996),
    ('Messier 94',                None,                        192.721083, +41.120444, 'Galaxy SABa',       +8.24, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 95',                None,                        160.990417, +11.703806, 'Galaxy Sb',         +9.77, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 96',                None,                        161.690583, +11.819944, 'Galaxy Sab',        +9.21, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 97',                'Owl Nebula',                168.698792, +55.019028, 'Planetary nebula',  +9.90, 'V',    +0.076,    +1.214,  +1.235),
    ('Messier 98',                None,                        183.451208, +14.900333, 'Galaxy SABb',      +10.84, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 99',                'Coma Pinwheel',             184.706667, +14.416500, 'Galaxy Sc',         +9.84, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 100',               None,                        185.728458, +15.821806, 'Galaxy SABb',       +9.47, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 101',               'Pinwheel Galaxy',           210.802250, +54.348944, 'Galaxy SABc',       +7.90, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 102',               'Spindle Galaxy',            226.622917, +55.763222, 'Galaxy S0-a',       +9.89, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 103',               None,                         23.340875, +60.658000, 'Open cluster',      +7.40, 'V',    -1.394,    -0.598,  +0.366),
    ('Messier 104',               'Sombrero Galaxy',           189.997625, -11.623056, 'Galaxy Sa',         +8.59, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 105',               None,                        161.956625, +12.581611, 'Galaxy E',          +9.27, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 106',               None,                        184.739583, +47.303972, 'Galaxy Sbc',        +9.29, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 107',               None,                        248.133000, -13.053639, 'Globular cluster',  +7.93, 'V',    -1.924,    -5.979,  +0.166),
    ('Messier 108',               None,                        167.879042, +55.674111, 'Galaxy Sc',        +10.05, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 109',               None,                        179.399917, +53.374528, 'Galaxy Sbc',        +9.88, 'V',    +0.000,    +0.000,  +0.000),
    ('Messier 110',               None,                         10.092000, +41.685306, 'Galaxy E',          +8.15, 'V',    +0.000,    +0.000,  +0.000),
]
