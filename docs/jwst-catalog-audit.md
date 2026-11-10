# JWST catalog and sky sparsity audit

Checked on November 10, 2026 against the local catalogs used by the app.

## What the catalog shows

JWST is substantially sparser than Hubble at the same sky resolution. With all three fixed-sky categories selected and the full available date range:

| Measurement | JWST | Hubble |
| :--- | ---: | ---: |
| Fixed-sky exposures | 43,647 | 569,227 |
| Fixed-sky visits | 17,460 | 236,970 |
| Fixed-sky catalog targets | 6,535 | 68,514 |
| Fixed-sky exposure hours, from the exposure table | 26,314.6 | 121,106.0 |
| Populated Galactic sky bins at NSIDE 64 | 2,715 | 17,432 |
| Populated Galactic sky bins at NSIDE 256 | 3,271 | 30,937 |
| Earliest observation in the science exposure table | May 2, 2022 | April 24, 1990 |
| Latest observation in the science exposure table | November 4, 2026 | November 9, 2026 |

These are measurements of this snapshot, rather than a claim about the complete current archive. Catalog target names can include aliases and subdivisions of the same physical object.

At NSIDE 64, JWST populates 5.52% of the 49,152 equal-area bins, compared with Hubble's 35.47%. At NSIDE 256, those figures fall to 0.416% and 3.93% because the bins become much smaller and many exposures revisit the same locations. **These percentages are occupied pointing bins, not the percentage of sky actually imaged.**

JWST's small fields of view help explain the clustered appearance: NIRCam's two main fields each measure about 2.2 × 2.2 arcminutes, much smaller than a standard heatmap cell. See [NASA's NIRCam description](https://science.nasa.gov/mission/webb/nircam/). This app plots catalog pointing locations rather than reconstructing instrument footprints.

## Raw records reconcile with the stored science exposures

The monthly JWST download cache contains entries from May 2022 through December 2026, with no missing monthly window. December 2026 is an empty future window; November is partial in this snapshot.

| ETL stage | Rows remaining |
| :--- | ---: |
| Raw rows across cached monthly responses | 298,260 |
| After removing six duplicate observation IDs | 298,254 |
| After rejecting one invalid-coordinate or nonpositive-exposure record | 298,253 |
| After separating 15,751 calibration records | 282,502 |
| After excluding 11,420 synthetic composite records | 271,082 |
| After grouping simultaneous instrument/channel records | 47,231 |
| After excluding 3,584 moving Solar System exposures from fixed-sky maps | 43,647 |

The 47,231 grouped exposure records exactly match the JWST rows in the stored exposure table. The fixed-sky exposure hours also match the sum of the HEALPix values at every supported resolution. The reduction in raw row count is therefore accounted for by the existing ETL rules; it does not indicate that the viewer dropped those rows.

This accounting checks that the pipeline applied its current rules consistently. It does not independently validate every target classification or every simultaneous-exposure grouping. The count-only MAST cross-check timed out, so current archive completeness remains unverified.

## Rendering detail is a separate issue

The first memory optimization capped both Fine and Ultra-Fine previews at the same 360 longitude samples. Their underlying HEALPix bins differed, but the rendering ceiling made the finer setting less visually effective.

The preview cap is retained to prioritize memory use and responsiveness. All five scientific HEALPix levels remain available, so pixel values and grouping still change with the dial. The static PNG uses a denser 720 × 360 rendering grid. A denser preview can distinguish nearby populated cells more clearly; it does not turn JWST's localized observations into an all-sky survey.

The counts were obtained with the catalog reader and HEALPix binning in [catalog.py](../src/telescope_heatmap/catalog.py) and [binning.py](../src/telescope_heatmap/binning.py). Raw-row accounting follows [etl.py](../src/telescope_heatmap/etl.py).
