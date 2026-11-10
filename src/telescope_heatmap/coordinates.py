"""Coordinate transformations (RA/Dec -> Galactic l, b) and target classification heuristics."""

from __future__ import annotations

import re

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from telescope_heatmap.config import (
    GALACTIC_DISK_LAT_DEG,
    LMC_B_RANGE,
    LMC_L_RANGE,
    SMC_B_RANGE,
    SMC_L_RANGE,
)

# Regex patterns for target classification
_CALIBRATION_CLASS_RE = re.compile(r"(?:^|;\s*)CALIBRATION\b", re.IGNORECASE)
_CALIBRATION_NAME_RE = re.compile(
    r"^(?:DARK|BIAS|CCDFLAT|FLAT|WAVE|TUNGSTEN|ARC|STIMLAMP|NONE|NULL|INTERNAL|"
    r"SKY-?FLAT|EARTH-?CALIB|BKGND|BKGD|NBKGD\d*|SBKGD\d*|POLE-?BKG|AIRGLOW)\b"
    r"|DOORS-?CLOSED|[-_]BKGD$",
    re.IGNORECASE,
)
_CALIBRATION_TITLE_RE = re.compile(
    r"\b(?:WAVELENGTH\s+CALIBRATION|ALIGNMENT\s+ACTIVITIES|SENSITIVITY\s+MONITOR|"
    r"F-?FLAT\s+CHARACTERIZATION|BACKGROUND\s+OBSERVATIONS\s+AROUND\s+SOUTH\s+CELESTIAL\s+POLE)\b",
    re.IGNORECASE,
)

_EXOPLANET_RE = re.compile(
    r"EXTRA-?SOLAR\s+PLANET|EXOPLANET|PROTOPLANETARY|DEBRIS\s+DISK|CIRCUMSTELLAR\s+DISK|"
    r"CIRCUMSTELLAR\s+DUST.*DISK|LOW\s+MASS\s+COMPANION",
    re.IGNORECASE,
)

_EXOPLANET_TITLE_RE = re.compile(
    r"EXTRA-?SOLAR\s+PLANET|EXOPLANET|PROTOPLANETARY|DEBRIS\s+DISK|CIRCUMSTELLAR\s+DISK|"
    r"TRANSITING\s+PLANET|PLANET\s+ATMOSPHERE|TRANSIT\s+SPECTROSCOPY|MATURE\s+PLANETS|"
    r"SECOND\s+PLANET\s+IN\s+PROXIMA|IS\s+THE\s+SECONDARY\s+A\s+PLANET",
    re.IGNORECASE,
)

_EXOPLANET_NAME_RE = re.compile(
    r"^(?:WASP|HAT-?P|HATS|TRAPPIST|TOI|KOI|KELT|K2|XO|TRES|COROT|QATAR|MASCARA|KEPLER|NGTS|"
    r"L-?98-?59|LHS-?(?:475|1140)|LTT-?(?:1445|9779)|GJ-?(?:1132|341|436|486|1214|3470|367|357|9827)|"
    r"GLIESE\s*876|HD-?\d+[B-Z]|HR-?8799|PDS-?70|BETA-?PIC|AU-?MIC|55-?CNC|HIP-?65426|VHS-?1256|PROXIMA)\b",
    re.IGNORECASE,
)

# Standard 3-letter IAU constellation abbreviations, Greek Bayer letters, & Milky Way stellar/ISM/cluster catalogs
_MILKY_WAY_STAR_NAME_RE = re.compile(
    r"^(?:HD|HDE|BD|CD|CPD|HR|HIP|WD|GD|HZ|G191|FEIGE|LTT|LHS|GJ|GL\b|GLIESE|LP|ROSS|WOLF|PSR|"
    r"2?EUVE[_\s]*J|2?RE[_\s]*J|RXS?[_\s]*J?(?!141336|1729\.1)|1H\s*\d+|XTE[_\s]*J|2MASP\s*J|"
    r"LB[-_\s]*\d+|MCT[-_\s]*\d+|BRI[-_\s]*\d+|CBS[-_\s]*\d+|JL[-_\s]*\d+|K1[-_\s]*16|H\s*1504\+65|"
    r"GCRV|SK-|AGK|BPM|RECX\d+|HS[-_\s]*(?:1650|1817|1136)|PG[-_\s]*(?:1658|1123|1234|1520|1057|0749|1626|0952)|"
    r"CAPELLA|SIRIUS|VEGA|ALTAIR|BETELGEUSE|RIGEL|PROCYON|ARCTURUS|ALDEBARAN|ANTARES|SPICA|"
    r"POLLUX|DENEB|REGULUS|FOMALHAUT|ALGOL|MIRA|CANOPUS|GALCEN|CL-ARP|CL-TERZAN|HH\d+|"
    r"ALPHA|BETA|GAMMA|DELTA|EPSILON|ZETA|ETA|THETA|IOTA|KAPPA|LAMBDA|MU\b|NU\b|XI\b|OMICRON|"
    r"PI\b|RHO\b|SIGMA|TAU\b|UPSILON|PHI\b|CHI\b|PSI\b|OMEGA|"
    r"M[-_\s]*(?:2|3|4|5|10|12|13|15|22|30|53|54|55|62|68|70|71|72|75|79|80|92|107)\b|"
    r"NGC[-_\s]*0*(?:246|1360|1851|2506|6302|6543|6752|7009|7293)\b|"
    r"PN\s+|NISM\b|DIFFUSE\s+ISM|LVC\s+|MBM\s*\d+|IRAS\s+CLOUD|ALEXIS|OPH-?CORE|CRAB-?NEBULA)"
    r"|\b(?:HER|CYG|SCO|LMC|SMC|CEN|VELA|CIR|AQL|GX)[-_ ]*X-\d+"
    r"|(?:^|[-_\s])(?:AND|ANT|APS|AQR|AQL|ARA|ARI|AUR|BOO|CAE|CAM|CNC|CVN|CMA|CMI|CAP|CAR|CAS|"
    r"CEN|CEP|CET|CHA|CIR|COL|COM|CRA|CRB|CRV|CRT|CRU|CYG|DEL|DOR|DRA|EQU|ERI|FOR|GEM|GRU|HER|"
    r"HOR|HYA|HYI|IND|LAC|LEO|LMI|LEP|LIB|LUP|LYN|LYR|MEN|MIC|MON|MUS|NOR|OCT|OPH|ORI|PAV|PEG|"
    r"PER|PHE|PIC|PSC|PSA|PUP|PYX|RET|SGE|SGR|SCO|SCL|SCT|SER|SEX|TAU|TEL|TRI|TUC|UMA|UMI|VEL|"
    r"VIR|VOL|VUL)$",
    re.IGNORECASE,
)

_SOLAR_SYSTEM_CLASS_RE = re.compile(
    r"SOLAR\s+SYSTEM|ASTEROID|COMET|DWARF-?PLANET|KUIPER|\bCENTAURS?\b(?!\s*[-A])|TROJAN|TRANS-?NEPTUNIAN|"
    r"(?<!EXTRA-SOLAR\s)(?<!EXTRA\sSOLAR\s)\bPLANET\b(?!\s*NEBULA)",
    re.IGNORECASE,
)

_SOLAR_SYSTEM_TITLE_RE = re.compile(
    r"SOLAR\s+SYSTEM\s+OBJECTS|ASTEROID\s+BELT|COMETARY\b|COMET\s+[CP29]|"
    r"TARGET\s+OF\s+OPPORTUNITY,\s*COMETS|TRANS-?NEPTUNIAN|KUIPER\s+BELT|"
    r"ACTIVE\s+CENTAUR|MOVING\s+TARGET\s+TEST|MARS\s+ATMOSPHERE|ARGON\s+ON\s+TITAN",
    re.IGNORECASE,
)

_SOLAR_SYSTEM_NAME_RE = re.compile(
    r"^(?:MERCURY|VENUS|MARS|JUPITER|SATURN|URANUS|NEPTUNE|PLUTO|MOON|LUNA|"
    r"IO|IOTORUS|EUROPA|GANYMEDE|CALLISTO|TITAN|ENCELADUS|RHEA|DIONE|TETHYS|MIMAS|IAPETUS|"
    r"TITANIA|OBERON|ARIEL|UMBRIEL|MIRANDA|TRITON|CHARON|CERES|VESTA|PALLAS|HAUMEA|MAKEMAKE|"
    r"DIDYMOS|POLYMELE|ORUS|EURYBATES|PSYCHE|ALTJIRA|TYPHON|THEREUS|HUYA|MORS-SOMNUS|"
    r"JUP-DTOH\d*|NAURORA|SAURORA|WANSA|NEAT-Q4|LINEAR[-_A0-9]|MCNAUGHT-HARTLEY|"
    r"HALE-BOPP|HYAKUTAKE|TEMPEL-TUTTLE|BORRELLY|ENCKE|BRADFIELD|D_ARREST|MUELLER)\b"
    r"|(?:^|[-_\s])(?:COMET[-_\s]|ASTEROID[-_\s]|ECLIPTIC-RA)",
    re.IGNORECASE,
)

_MILKY_WAY_CLASS_RE = re.compile(
    r"(?<!EXT-)\bSTAR\b|STELLAR|PROTOSTARS?|BROWN\s+DWARF|WHITE\s+DWARF|PULSAR|NEUTRON\s+STAR|"
    r"NOVA\b|SUPERNOVA\s+REMNANT|PLANETARY\s+NEBULA|HII\s+REGION|REFLECTION\s+NEBULA|"
    r"DARK\s+NEBULA|DUST\s+NEBULA|MOLECULAR\s+CLOUD|MOLECULAR\s+GAS|PHOTODISSOCIATION|\bISM\b|INTERSTELLAR|"
    r"HERBIG-?HARO|GLOBULAR\s*(?:STAR\s*)?CLUSTER|OPEN\s*(?:STAR\s*)?CLUSTER|STELLAR\s+CLUSTER",
    re.IGNORECASE,
)

_MILKY_WAY_TITLE_RE = re.compile(
    r"\b(?:STAR|STARS|STELLAR|BROWN\s+DWARFS?|WHITE\s+DWARFS?|DWARF\s+NOVA|CATACLYSMIC|"
    r"INTERACTING\s+BINARY|RS\s+CVN|PULSAR|NEUTRON\s+STAR|PROTOSTARS?|CRAB\s+NEBULA|"
    r"PLANETARY\s+NEBULA|SUPERNOVA\s+REMNANT|INTERSTELLAR\s+MEDIUM|INTERSTELLAR\s+DUST|"
    r"GALACTIC\s+HALOS?|GALACTIC\s+FOUNTAIN|HIGH-VELOCITY\s+CLOUDS?|GLOBULAR\s+CLUSTER|"
    r"OPEN\s+CLUSTER|MOLECULAR\s+CLOUDS?|HOT-STAR\s+WINDS|CHROMOSPHERIC|CORONA)\b",
    re.IGNORECASE,
)

_DEEP_SPACE_CLASS_RE = re.compile(
    r"GALAX(?:Y|IES)|QUASAR|QSO|\bAGN\b|BLAZAR|SEYFERT|LINER|STARBURST|"
    r"CLUSTER\s+OF\s+GALAXIES|CLUSTERS\s+OF\s+GALAXIES|GALAXY\s+CLUSTER|HIGH[- ]REDSHIFT|"
    r"EXT-STAR|SUPERNOVA(?:E|\s+TYPE|\b)(?!\s*REMNANT)|GRAVITATIONAL\s+LENS|DEEP\s+FIELD|BLANK\s+FIELD|"
    r"PARALLEL\s+FIELD|HIGH\s+LATITUDE\s+FIELD|DAMPED\s+LYMAN|INTERGALACTIC|COSMOLOG",
    re.IGNORECASE,
)

_DEEP_SPACE_NAME_RE = re.compile(
    r"^(?:MRK|MKN|3C|PKS|QSO|ABELL|ACO\b|MACS|GOODS|HUDF|XDF|CDFS|HDF|UDS|CEERS|PRIMER|FRESCO|"
    r"COSMOS|CANDELS|1ES\b|VIIZW|IZW|FAIRALL|PICTOR\s*A|COMA\b|VIRGO|HE[-_\s]*\d|HS[-_\s]*(?:1700|1307|1140|0624)|"
    r"H1821\+643|TONS[-_\s]*180|MS\d{4}|SN[-_\s]*\d{4}|AT[-_\s]*\d{4}|ZTF\d{2}|GRB\s*\d+|RXJ(?:141336|1729\.1)|"
    r"M[-_\s]*(?:31|32|33|51|63|74|81|82|83|87|100|101|104|106)\b|MESSIER[-_\s]*(?:31|33|51|81|82|83|87|101)\b|"
    r"NGC[-_\s]*0*(?:221|224|253|1068|1316|1365|3079|3256|3783|4051|4151|4194|4395|4486|5128|5236|5253|5457|5548|6521|7469|7714)\b|"
    r"CENA-NUC|ESO\s*\d+|IRAS\s*13349|PC\s*0025)",
    re.IGNORECASE,
)

_DEEP_SPACE_TITLE_RE = re.compile(
    r"GALAX(?:Y|IES)|QUASAR|QSO|\bAGN\b|SEYFERT|BLAZAR|EXTRAGALACTIC|INTERGALACTIC|\bIGM\b|"
    r"COSMIC\s+WEB|EARLY\s+UNIVERSE|COSMIC\s+DAWN|REDSHIFT\s+FRONTIER|REIONIZATION|"
    r"HIGH-?Z|HIGH[- ]REDSHIFT|PROTO-?CLUSTER|INTRAGROUP|"
    r"CLUSTERS\s+OF\s+GALAXIES|LENSED|GRAVITATIONAL\s+LENS|HUBBLE\s+SEQUENCE|"
    r"SUPERNOVA(?:E|\b)(?!\s*REMNANT)|STARBURST|PURE\s+PARALLEL|PANORAMIC|CEERS|PRIMER|FRESCO|CANUCS|NGDEEP|GLIMPSE",
    re.IGNORECASE,
)


def add_galactic_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Galactic Longitude (gal_l in [-180, 180] deg) and Latitude (gal_b in [-90, 90] deg).

    Centering gal_l in [-180, 180] places the Galactic Center (l=0°, b=0°) directly at the center
    of 2D projections and 3D spherical meshes.
    """
    if df.empty:
        df = df.copy()
        df["gal_l"] = pd.Series(dtype="float64")
        df["gal_b"] = pd.Series(dtype="float64")
        return df

    out = df.copy()
    ra = out["s_ra"].to_numpy(dtype=np.float64)
    dec = out["s_dec"].to_numpy(dtype=np.float64)

    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
    # Wrap l into [-180, 180) so (l=0, b=0) is centered
    gal_l = ((coords.l.deg + 180.0) % 360.0) - 180.0
    gal_b = coords.b.deg

    out["gal_l"] = gal_l
    out["gal_b"] = gal_b
    return out


def is_calibration_row(df: pd.DataFrame) -> pd.Series:
    """Identify residual calibration rows inside science query results."""
    if df.empty:
        return pd.Series(dtype=bool)

    t_class = df.get("target_classification", pd.Series("", index=df.index)).fillna("").astype(str)
    t_name = df.get("target_name", pd.Series("", index=df.index)).fillna("").astype(str)
    obs_title = df.get("obs_title", pd.Series("", index=df.index)).fillna("").astype(str)
    mt_flag = df.get("mtFlag", pd.Series(False, index=df.index)).fillna(False).astype(bool)

    by_class = t_class.str.contains(_CALIBRATION_CLASS_RE, regex=True)
    by_name = t_name.str.contains(_CALIBRATION_NAME_RE, regex=True)
    by_title = obs_title.str.contains(_CALIBRATION_TITLE_RE, regex=True) & (~mt_flag) & (t_class.str.strip() == "")
    return by_class | by_name | by_title


def classify_targets(df: pd.DataFrame, galactic_disk_lat_deg: float = GALACTIC_DISK_LAT_DEG) -> pd.DataFrame:
    """Classify each science observation into one of four categories and add boolean flags:
      - solar_system
      - exoplanet
      - milky_way
      - deep_space
    Assumes `gal_l` and `gal_b` have already been computed via `add_galactic_coordinates`.
    """
    if df.empty:
        out = df.copy()
        for col in ["is_solar_system", "is_exoplanet", "is_milky_way", "is_deep_space"]:
            out[col] = pd.Series(dtype=bool)
        out["target_category"] = pd.Series(dtype="str")
        return out

    out = df.copy()
    t_class = out.get("target_classification", pd.Series("", index=out.index)).fillna("").astype(str)
    t_name = out.get("target_name", pd.Series("", index=out.index)).fillna("").astype(str)
    obs_title = out.get("obs_title", pd.Series("", index=out.index)).fillna("").astype(str)
    mt_flag = out.get("mtFlag", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    obs_coll = out.get("obs_collection", pd.Series("", index=out.index)).fillna("").astype(str)

    has_explicit_class = t_class.str.strip() != ""
    out["target_system"] = normalize_target_system(out)
    t_sys = out["target_system"]

    # 1. Exoplanet check FIRST so "EXTRA-SOLAR PLANET" / "PROTOPLANETARY DISK" are never confused with Solar System
    has_exoplanet_tag = (
        t_class.str.contains(_EXOPLANET_RE, regex=True)
        | obs_title.str.contains(_EXOPLANET_TITLE_RE, regex=True)
        | t_name.str.contains(_EXOPLANET_NAME_RE, regex=True)
        | obs_coll.isin(["Kepler", "KeplerFFI", "K2", "K2FFI", "TESS"])
    )
    is_exoplanet = has_exoplanet_tag & (~mt_flag)

    # 2. Solar System check:
    #    Only consult obs_title when target_classification is missing (~has_explicit_class) so proposal titles
    #    mentioning "comets" or "Centaurus A" never override an explicit Star/ISM/Galaxy target_classification.
    has_ss_class = t_class.str.contains(_SOLAR_SYSTEM_CLASS_RE, regex=True) & (~has_exoplanet_tag)
    has_ss_name = (
        t_name.str.contains(_SOLAR_SYSTEM_NAME_RE, regex=True)
        & (~has_exoplanet_tag)
        & (~t_name.str.contains(_DEEP_SPACE_NAME_RE, regex=True))
    )
    has_ss_title = (
        (~has_explicit_class)
        & obs_title.str.contains(_SOLAR_SYSTEM_TITLE_RE, regex=True)
        & (~has_exoplanet_tag)
        & (~t_name.str.contains(_MILKY_WAY_STAR_NAME_RE, regex=True))
        & (~t_name.str.contains(_DEEP_SPACE_NAME_RE, regex=True))
    )
    is_solar_system = (mt_flag | has_ss_class | has_ss_name | has_ss_title) & (~is_exoplanet)

    # 3. Milky Way & Local Group vs. Deep Space for remaining fixed-sky targets
    remaining = ~(is_solar_system | is_exoplanet)

    # Extragalactic check: explicit class, known extragalactic catalog name, or (when class is missing) extragalactic title
    has_deep_name = t_name.str.contains(_DEEP_SPACE_NAME_RE, regex=True)
    has_mw_name = t_name.str.contains(_MILKY_WAY_STAR_NAME_RE, regex=True) & (~has_deep_name)

    has_deep_class = t_class.str.contains(_DEEP_SPACE_CLASS_RE, regex=True) | has_deep_name
    has_mw_class = (t_class.str.contains(_MILKY_WAY_CLASS_RE, regex=True) | has_mw_name) & (~has_deep_class)

    # When neither explicit target_classification nor catalog name decided the row, check obs_title keywords
    undecided = remaining & (~has_deep_class) & (~has_mw_class)
    has_deep_title = undecided & obs_title.str.contains(_DEEP_SPACE_TITLE_RE, regex=True)
    has_mw_title = undecided & (~has_deep_title) & obs_title.str.contains(_MILKY_WAY_TITLE_RE, regex=True)

    is_deep_explicit = remaining & (has_deep_class | has_deep_title)
    is_mw_explicit = remaining & (has_mw_class | has_mw_title)

    # Geometric fallback ONLY for rows lacking explicit classification, catalog match, and title match
    still_undecided = remaining & (~is_deep_explicit) & (~is_mw_explicit)
    gal_l = out["gal_l"].to_numpy(dtype=np.float64)
    gal_b = out["gal_b"].to_numpy(dtype=np.float64)

    in_galactic_disk = np.abs(gal_b) < galactic_disk_lat_deg
    in_lmc = (
        (gal_l >= LMC_L_RANGE[0]) & (gal_l <= LMC_L_RANGE[1]) & (gal_b >= LMC_B_RANGE[0]) & (gal_b <= LMC_B_RANGE[1])
    )
    in_smc = (
        (gal_l >= SMC_L_RANGE[0]) & (gal_l <= SMC_L_RANGE[1]) & (gal_b >= SMC_B_RANGE[0]) & (gal_b <= SMC_B_RANGE[1])
    )
    in_local_geometric = pd.Series(in_galactic_disk | in_lmc | in_smc, index=out.index)

    is_milky_way = is_mw_explicit | (still_undecided & in_local_geometric)
    is_deep_space = remaining & (~is_milky_way)

    categories = np.select(
        [is_solar_system, is_exoplanet, is_milky_way, is_deep_space],
        ["solar_system", "exoplanet", "milky_way", "deep_space"],
        default="deep_space",
    )
    out["target_category"] = pd.Series(categories, index=out.index, dtype="str")

    # 4. Enforce system-level category consistency across all non-generic target systems
    generic_systems = {
        "Unidentified / Parallel Fields",
        "HST Pure Parallel Fields (ANY)",
        "Unknown Target",
        "TARGETS",
        "NIRCAM",
    }
    non_generic_mask = ~t_sys.isin(generic_systems)
    if non_generic_mask.any():
        # Priority when unifying a system: if any exposure has an explicit exoplanet or solar_system tag,
        # or take the category with the highest total exposure time (or row count if t_exptime is unavailable)
        if "t_exptime" in out.columns:
            sys_cat_weights = (
                out.loc[non_generic_mask]
                .groupby(["target_system", "target_category"], as_index=False)
                .agg({"t_exptime": "sum"})
                .sort_values(["target_system", "t_exptime"], ascending=[True, False])
                .drop_duplicates("target_system")
                .set_index("target_system")["target_category"]
            )
        else:
            sys_cat_weights = (
                out.loc[non_generic_mask]
                .groupby("target_system")["target_category"]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
            )
        out.loc[non_generic_mask, "target_category"] = t_sys.loc[non_generic_mask].map(sys_cat_weights).astype("str")

    out["is_solar_system"] = out["target_category"] == "solar_system"
    out["is_exoplanet"] = out["target_category"] == "exoplanet"
    out["is_milky_way"] = out["target_category"] == "milky_way"
    out["is_deep_space"] = out["target_category"] == "deep_space"
    out["target_description"] = describe_targets(out)
    return out


_SYSTEM_RULES: list[tuple[re.Pattern[str], str]] = [
    # Solar System Planets & Moons
    (
        re.compile(
            r"^(?:JUPITER|IO\b|IOTORUS|EUROPA|GANYMEDE|CALLISTO|AMALTHEA|JUP-DTOH|NAURORA|SAURORA|WANSA)",
            re.IGNORECASE,
        ),
        "Jupiter System (Jupiter & Moons)",
    ),
    (
        re.compile(
            r"^(?:SATURN|TITAN\b|ENCELADUS|RHEA\b|DIONE\b|TETHYS|MIMAS|IAPETUS|HYPERION|PHOEBE)",
            re.IGNORECASE,
        ),
        "Saturn System (Saturn & Moons)",
    ),
    (
        re.compile(
            r"^(?:URANUS|TITANIA|OBERON|ARIEL\b|UMBRIEL|MIRANDA)",
            re.IGNORECASE,
        ),
        "Uranus System (Uranus & Moons)",
    ),
    (
        re.compile(r"^(?:NEPTUNE|TRITON\b|PROTEUS|NEREID)", re.IGNORECASE),
        "Neptune System (Neptune & Moons)",
    ),
    (
        re.compile(r"^(?:MARS\b|MARS[-_]|PHOBOS|DEIMOS)", re.IGNORECASE),
        "Mars System (Mars & Moons)",
    ),
    (
        re.compile(r"^(?:MOON|LUNA\b)", re.IGNORECASE),
        "Earth Moon (Lunar)",
    ),
    (
        re.compile(r"^VENUS\b", re.IGNORECASE),
        "Venus",
    ),
    (
        re.compile(r"^MERCURY\b", re.IGNORECASE),
        "Mercury",
    ),
    (
        re.compile(r"^(?:PLUTO|CHARON)", re.IGNORECASE),
        "Pluto System (Pluto & Charon)",
    ),
    (
        re.compile(r"^(?:1-)?CERES\b", re.IGNORECASE),
        "1 Ceres (Dwarf Planet)",
    ),
    (
        re.compile(r"^DIDYMOS\b", re.IGNORECASE),
        "65803 Didymos (Asteroid System)",
    ),
    (
        re.compile(r"^ECLIPTIC-RA", re.IGNORECASE),
        "Kuiper Belt Pencil-Beam Survey (ECLIPTIC-RA12H)",
    ),
    (
        re.compile(r"^NEAT-Q4", re.IGNORECASE),
        "Comet C/2001 Q4 (NEAT)",
    ),
    (
        re.compile(r"^LINEAR-2000WM1", re.IGNORECASE),
        "Comet C/2000 WM1 (LINEAR)",
    ),
    (
        re.compile(r"^LINEARA2", re.IGNORECASE),
        "Comet C/2001 A2 (LINEAR)",
    ),
    (
        re.compile(r"^2022E3", re.IGNORECASE),
        "Comet C/2022 E3 (ZTF)",
    ),
    # Famous Deep-Space Surveys, Galaxies & Exoplanet Fields
    (
        re.compile(r"^(?:M-?31\b|NGC-?0*224\b|NGC-?0*221\b)", re.IGNORECASE),
        "M31 (Andromeda Galaxy)",
    ),
    (
        re.compile(r"^(?:NGC-?0*5128\b|CENA-NUC\b)", re.IGNORECASE),
        "Centaurus A (NGC 5128 Active Galaxy)",
    ),
    (
        re.compile(r"^(?:M\s*87\b|NGC-?0*4486\b|VIRGO[_ ]|VIRGOIII)", re.IGNORECASE),
        "M87 / Virgo Cluster Core",
    ),
    (
        re.compile(
            r"^(?:HUDF|XDF-|GOODS-S|GOODS\b|CDFS|CDF-SOUTH|UDF-|INPUT_LULLS|MIRI-XDF|MDS\.MIRI-DEEP|3215_DEEPJWST|DEEP-HST-V1|1287_TRIM)",
            re.IGNORECASE,
        ),
        "HUDF / GOODS-South Deep Field",
    ),
    (
        re.compile(r"^(?:ACO-?S-?1063|ABELL-?S-?1063|GLIMPSE_COMBINED)", re.IGNORECASE),
        "Abell S1063 (GLIMPSE Lensing Cluster)",
    ),
    (
        re.compile(r"^(?:GOODS-N|HDF-N|HDF-1236|1181-MERGED-APT)", re.IGNORECASE),
        "GOODS-North / HDF-North Deep Field",
    ),
    (
        re.compile(r"^(?:HDF-S|HDFS|ANY\s*\(338\.2,\s*-60\.[67]\))", re.IGNORECASE),
        "Hubble Deep Field South (HDF-S)",
    ),
    (
        re.compile(r"^ANY\s*\(53\.[12],\s*-27\.[78]\)", re.IGNORECASE),
        "HUDF / GOODS-South Deep Field",
    ),
    (
        re.compile(r"^(?:NGC-?0*104\b|47-?TUC)", re.IGNORECASE),
        "47 Tucanae (NGC 104 Globular Cluster)",
    ),
    (
        re.compile(r"^(?:NGC-?0*5139\b|OMEGA-?CEN)", re.IGNORECASE),
        "Omega Centauri (NGC 5139 Globular Cluster)",
    ),
    (
        re.compile(r"^UDS-1\b", re.IGNORECASE),
        "UDS-1 (Ultra Deep Survey)",
    ),
    (
        re.compile(r"^MACSJ?0416", re.IGNORECASE),
        "MACS J0416.1-2403 (Galaxy Cluster)",
    ),
    (
        re.compile(r"^KeplerFFI-FIELD", re.IGNORECASE),
        "Kepler Cygnus Field",
    ),
    (
        re.compile(r"^wcs-ktwo", re.IGNORECASE),
        "K2 Ecliptic Campaign Fields",
    ),
    (
        re.compile(r"^TESS[-_\s]*FFI", re.IGNORECASE),
        "TESS All-Sky Survey Sectors",
    ),
    (
        re.compile(
            r"^(?:ANY|PAR-TARGET|PAR|HIGH|HI-LATITUDE-FIELD|HI-LAT|HIGH-GALACTIC-LATITUDE|LOW|LOW-LATITUDE-FIELD|PARALLEL-FIELD|UNKNOWN-TARGET-1|LIST-1)\s*\(",
            re.IGNORECASE,
        ),
        "HST Pure Parallel Fields (ANY)",
    ),
    (
        re.compile(r"^(?:UNKNOWN|SKY|SKY BKGD)\s*\(", re.IGNORECASE),
        "Unidentified / Parallel Fields",
    ),
]

_SUFFIX_STRIP_RE = re.compile(
    r"(?:[-_](?:BACKGROUND|BKGD|UPDATE|CLEAN|OFFSET|CENTER|TILE[-_]?\d+|SUB\d+|"
    r"NIRSPEC|MIRI|NIRCAM|UVIS|WFC3?|ACS|F\d{3,4}[WMN]|MEDIUM\d+|DEEP))$",
    re.IGNORECASE,
)


def _canonicalize_single_name(raw_name: str) -> str:
    """Map a single raw MAST target_name to a canonical target/system name."""
    name = raw_name.strip()
    if not name:
        return "Unknown Target"

    for pattern, canonical in _SYSTEM_RULES:
        if pattern.search(name):
            return canonical

    # Strip common sub-tile or instrument suffixes repeatedly (e.g. HUDF-DEEP-F160W)
    cleaned = name
    for _ in range(2):
        cleaned = _SUFFIX_STRIP_RE.sub("", cleaned).strip()

    # Normalize common catalog prefixes so "Mrk 421", "MRK-421", "MRK421" -> "MRK-421"
    m_cat = re.match(
        r"^(MRK|NGC|IC|HD|GJ|TOI|WASP|HAT-?P|HATS|TRAPPIST|K2|KOI|KELT|ABELL|MACS|UGC|3C|PKS|PG|HS|HE|TON\s*S?)"
        r"[-_\s]*(\d+[A-Z0-9.+_-]*)$",
        cleaned,
        re.IGNORECASE,
    )
    if m_cat:
        prefix = re.sub(r"\s+", "", m_cat.group(1).upper())
        if prefix == "HATP":
            prefix = "HAT-P"
        return f"{prefix}-{m_cat.group(2).upper()}"

    # Normalize variable star / constellation names like "EX Hya", "EX-HYA", "SS Cyg", "SS-CYG", "SSCyg"
    m_var = re.match(
        r"^([A-Z]{1,2}|V\d{3,4})[-_\s]+([A-Z]{3})$",
        cleaned,
        re.IGNORECASE,
    )
    if m_var:
        return f"{m_var.group(1).upper()} {m_var.group(2).capitalize()}"

    return cleaned.upper()


def normalize_target_system(df: pd.DataFrame) -> pd.Series:
    """Assign each row a canonical `target_system` name grouping planets+moons, mosaic tiles, and aliases."""
    if df.empty or "target_name" not in df.columns:
        return pd.Series(dtype="str")

    names = df["target_name"].fillna("").astype(str)
    unique_names = names.unique()
    mapping = {u_name: _canonicalize_single_name(u_name) for u_name in unique_names}
    return names.map(mapping).astype("str")


def _describe_single_row(sys_name: str, cat: str, raw_class: str, raw_title: str) -> str:
    """Return a concise, plain-English description of what a target system actually is."""
    s_up = sys_name.upper()
    c_up = raw_class.upper()
    t_up = raw_title.upper()

    if "KUIPER BELT" in s_up or "ECLIPTIC-RA" in s_up:
        return "Kuiper Belt / Trans-Neptunian Object Pencil-Beam Survey"
    if "SYSTEM (" in s_up or s_up in ("VENUS", "MERCURY", "EARTH MOON (LUNAR)"):
        return "Solar System Planet & Moons"
    if "CERES" in s_up or "DIDYMOS" in s_up or "ASTEROID" in c_up:
        return "Solar System Asteroid / Dwarf Planet"
    if "COMET" in s_up or "COMET" in c_up:
        return "Solar System Comet"
    if cat == "solar_system":
        return "Solar System Moving Target (KBO / Minor Body)"

    if "KEPLER CYGNUS" in s_up or "K2 ECLIPTIC" in s_up:
        return "Wide-Field Exoplanet Transit Survey Field"
    if cat == "exoplanet":
        if "DISK" in c_up or "DISK" in t_up:
            return "Protoplanetary / Circumstellar Debris Disk"
        return "Exoplanet Host Star System (Transit / Direct Spectroscopy)"

    if "HUDF" in s_up or "GOODS-" in s_up or "UDS-1" in s_up or "NEP-TDS" in s_up:
        return "Cosmological Deep Field Survey (Early Universe Galaxies)"
    if (
        "MACS" in s_up
        or "ABELL" in s_up
        or "ACO" in s_up
        or "CLUSTER OF GALAXIES" in c_up
        or "CLUSTERS OF GALAXIES" in c_up
    ):
        return "Massive Galaxy Cluster (Gravitational Lens)"
    if (
        s_up.startswith(("MRK-", "3C-", "PKS-", "PG-", "HS-", "HE-", "TONS-", "QSO-"))
        or "QUASAR" in c_up
        or "BLAZAR" in c_up
        or "SEYFERT" in c_up
    ):
        return "Quasar / Blazar / Active Galactic Nucleus (Supermassive Black Hole)"
    if "M31" in s_up or "M87" in s_up or s_up.startswith(("NGC-", "IC-", "UGC-", "SEXTANS", "ANDROMEDA", "LEO-P")):
        return "External Galaxy (Spiral / Elliptical / Dwarf Galaxy)"
    if "PARALLEL" in s_up or "UNIDENTIFIED" in s_up:
        return "Deep-Space Parallel Survey Field"

    if cat == "milky_way":
        if "CLUSTER" in c_up or s_up.startswith("CL-"):
            return "Milky Way Globular / Open Star Cluster"
        if "ISM" in c_up or "NEBULA" in c_up or "HORSEHEAD" in s_up or s_up.startswith("HH"):
            return "Milky Way Nebula / Interstellar Gas & Star-Forming Cloud"
        if "WD" in c_up or "WHITE DWARF" in c_up or "DA" in c_up:
            return "Milky Way White Dwarf Star"
        if "NOVA" in t_up or s_up in ("EX HYA", "SS CYG", "VW HYI", "UZ FOR", "U GEM", "HER X-1"):
            return "Milky Way Cataclysmic Variable / Interacting Binary Star"
        return "Milky Way Star / Stellar System"

    if raw_class and raw_class != "<NA>":
        return f"Extragalactic Target ({raw_class})"
    return "Extragalactic Deep-Space Target"


def describe_targets(df: pd.DataFrame) -> pd.Series:
    """Generate a plain-English `target_description` column for every row."""
    if df.empty:
        return pd.Series(dtype="str")

    sys_s = df.get("target_system", pd.Series("", index=df.index)).fillna("").astype(str)
    cat_s = df.get("target_category", pd.Series("", index=df.index)).fillna("").astype(str)
    cls_s = df.get("target_classification", pd.Series("", index=df.index)).fillna("").astype(str)
    ttl_s = df.get("obs_title", pd.Series("", index=df.index)).fillna("").astype(str)

    descriptions = [_describe_single_row(s, c, k, t) for s, c, k, t in zip(sys_s, cat_s, cls_s, ttl_s)]
    return pd.Series(descriptions, index=df.index, dtype="str")
