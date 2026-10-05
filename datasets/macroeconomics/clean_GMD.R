# =============================================================================
# clean_GMD.R
# =============================================================================
# Reads the Global Macro Database workbook (sheet "data_final") and writes a
# cleaned copy containing only cross-country-comparable indicators.
#
# What gets dropped, and why:
#   1. forecast_*      -- provenance FLAGS (0/1/NA), not data. They mark which
#                         cells are IMF-style projections rather than observed
#                         history. Optionally used to blank the flagged values
#                         first (see mask_forecast_values below).
#   2. crisis dummies  -- SovDebtCrisis / CurrencyCrisis / BankingCrisis are
#                         binary event indicators, not continuous indicators.
#   3. local-currency  -- levels denominated in each country's own currency
#      levels             (nGDP, cons, govdebt, M2, ...). These are NOT
#                         comparable across countries: 2019 nGDP ranges from
#                         418 (Venezuela) to 2.74e13 (Iran), which measures
#                         currency denomination, not economies. Redenominations
#                         also inject step changes mid-panel (Venezuela x1357 in
#                         2018, Bolivia x122 in 1985). The _USD and _GDP
#                         versions of the same concepts are kept.
#   4. USDfx           -- the exchange rate is itself in local-currency-per-USD,
#                         so its level is arbitrary in exactly the same way.
#
# Kept: USD levels, % of GDP ratios, rates (infl, unemp, interest rates),
#       REER, population, and per-capita USD.
#
# Usage:  Rscript clean_GMD.R GMD.xlsx data/macroeconomics/GMD_clean.xlsx
# =============================================================================

suppressPackageStartupMessages({
  library(readxl)
  library(writexl)
  library(dplyr)
})

# =============================================================================
# CONFIG -- edit these
# =============================================================================
IN_XLSX   <- commandArgs(trailingOnly = TRUE)[1]   # GMD.xlsx from the Global Macro Database
IN_SHEET  <- "data_final"
OUT_XLSX  <- commandArgs(trailingOnly = TRUE)[2]   # e.g. data/macroeconomics/GMD_clean.xlsx
OUT_SHEET <- "data_clean"
ALSO_WRITE_CSV <- TRUE          # csv alongside the xlsx (much faster to reload)

# --- year range -------------------------------------------------------------
YEAR_MIN <- 1980
YEAR_MAX <- 2023

# --- balanced panel ---------------------------------------------------------
# TRUE  = keep only countries present in EVERY year of [YEAR_MIN, YEAR_MAX],
#         so the node set is fixed over time.
BALANCED <- TRUE

# How "present" is judged:
#   "presence" -- a row exists for that country-year (the country existed).
#                 This is the looser test; it does not require complete data.
#   "complete" -- every column in BALANCE_VARS is non-missing that year.
#                 This is what SAIL_MACRO.py does, and it is much stricter.
BALANCE_MODE <- "presence"

# Only used when BALANCE_MODE == "complete". NULL means "all kept indicators",
# which will prune the panel hard. Usually better to name the few you model, e.g.
#   BALANCE_VARS <- c("rGDP_pc_USD", "infl", "CA_GDP", "govdebt_GDP")
BALANCE_VARS <- NULL

# --- what to drop -----------------------------------------------------------
DROP_FORECAST_FLAGS  <- TRUE   # the forecast_* flag columns themselves
DROP_CRISIS_DUMMIES  <- TRUE   # SovDebtCrisis / CurrencyCrisis / BankingCrisis
DROP_LOCAL_CURRENCY  <- TRUE   # levels in each country's own currency
DROP_FX_LEVEL        <- TRUE   # USDfx (local currency per USD -- arbitrary unit)

# Base-100 index series (CPI, deflator, HPI, rHPI). Their level depends on the
# base year, so cross-country level comparisons are only meaningful if the base
# year is shared. Their growth rates are fine. FALSE = keep them.
DROP_BASE100_INDEX   <- FALSE

# Set TRUE to blank out values that GMD flags as projections BEFORE dropping the
# flag columns. Within 1980-2023 this affects 72 cells, all current-account
# (CA / CA_GDP / CA_USD) for CAF, COG, FSM, GAB, LIE, MMR, SDN, TCD, VEN, YEM
# in 2019-2023. Combined with BALANCE_MODE = "complete" this will drop those
# countries rather than silently modelling forecasts as history.
MASK_FORECAST_VALUES <- FALSE

# =============================================================================
# COLUMN GROUPS
# =============================================================================
ID_COLS <- c("countryname", "ISO3", "id", "year", "income_group")

# Levels denominated in local currency units -- not comparable across countries.
LOCAL_CURRENCY_COLS <- c(
  # output
  "nGDP", "rGDP", "rGDP_pc",
  # expenditure components
  "cons", "hcons", "gcons", "inv", "finv", "exports", "imports", "CA",
  # government, all three coverage levels (combined / general / central)
  "govexp",   "gen_govexp",   "cgovexp",
  "govrev",   "gen_govrev",   "cgovrev",
  "govtax",   "gen_govtax",   "cgovtax",
  "govdef",   "gen_govdef",   "cgovdef",
  "govdebt",  "gen_govdebt",  "cgovdebt",
  # money aggregates
  "M0", "M1", "M2", "M3", "M4"
)

CRISIS_DUMMY_COLS <- c("SovDebtCrisis", "CurrencyCrisis", "BankingCrisis")
FX_LEVEL_COLS     <- c("USDfx")
BASE100_INDEX_COLS <- c("CPI", "deflator", "HPI", "rHPI")

# =============================================================================
# READ
# =============================================================================
message("reading ", IN_XLSX, " [", IN_SHEET, "] ...")

# NOTE: this workbook's <dimension> tag is unreliable (the Python loader has to
# call ws.reset_dimensions()). readxl scans real cells rather than trusting the
# tag, but the sanity check below will catch it if the read comes back short.
gmd <- readxl::read_excel(IN_XLSX, sheet = IN_SHEET, guess_max = 100000)
gmd <- as.data.frame(gmd)

message("  read ", nrow(gmd), " rows x ", ncol(gmd), " cols")
if (nrow(gmd) < 50000 || ncol(gmd) < 150) {
  warning("read looks truncated (expected ~56992 x 162) -- check the sheet's ",
          "dimension tag or re-export the workbook")
}

stopifnot(all(c("ISO3", "year") %in% names(gmd)))
gmd$year <- suppressWarnings(as.numeric(gmd$year))

# =============================================================================
# OPTIONAL: blank values that GMD flags as forecasts
# =============================================================================
if (MASK_FORECAST_VALUES) {
  flag_cols <- grep("^forecast_", names(gmd), value = TRUE)
  n_masked <- 0L
  for (fl in flag_cols) {
    target <- sub("^forecast_", "", fl)
    if (target %in% names(gmd)) {
      hit <- !is.na(gmd[[fl]]) & gmd[[fl]] == 1
      n_masked <- n_masked + sum(hit)
      gmd[[target]][hit] <- NA
    }
  }
  message("masked ", n_masked, " forecast-flagged values to NA")
}

# =============================================================================
# DROP COLUMNS
# =============================================================================
drop_cols <- character(0)
if (DROP_FORECAST_FLAGS) drop_cols <- c(drop_cols, grep("^forecast_", names(gmd), value = TRUE))
if (DROP_CRISIS_DUMMIES) drop_cols <- c(drop_cols, CRISIS_DUMMY_COLS)
if (DROP_LOCAL_CURRENCY) drop_cols <- c(drop_cols, LOCAL_CURRENCY_COLS)
if (DROP_FX_LEVEL)       drop_cols <- c(drop_cols, FX_LEVEL_COLS)
if (DROP_BASE100_INDEX)  drop_cols <- c(drop_cols, BASE100_INDEX_COLS)

drop_cols <- intersect(unique(drop_cols), names(gmd))
clean <- gmd[, setdiff(names(gmd), drop_cols), drop = FALSE]

message("dropped ", length(drop_cols), " columns; ", ncol(clean), " remain")
message("  kept indicators: ",
        paste(setdiff(names(clean), ID_COLS), collapse = ", "))

# =============================================================================
# FILTER YEARS
# =============================================================================
clean <- clean[!is.na(clean$year) & clean$year >= YEAR_MIN & clean$year <= YEAR_MAX, ]
message("year filter ", YEAR_MIN, "-", YEAR_MAX, ": ", nrow(clean), " rows, ",
        dplyr::n_distinct(clean$ISO3), " countries")

# =============================================================================
# BALANCED PANEL
# =============================================================================
if (BALANCED) {
  n_years_req <- YEAR_MAX - YEAR_MIN + 1L

  if (BALANCE_MODE == "presence") {
    ok <- clean %>%
      dplyr::group_by(ISO3) %>%
      dplyr::summarise(n = dplyr::n_distinct(year), .groups = "drop") %>%
      dplyr::filter(n == n_years_req) %>%
      dplyr::pull(ISO3)

  } else if (BALANCE_MODE == "complete") {
    bvars <- if (is.null(BALANCE_VARS)) setdiff(names(clean), ID_COLS) else BALANCE_VARS
    bvars <- intersect(bvars, names(clean))
    if (length(bvars) == 0) stop("BALANCE_VARS matched no columns")
    message("  balancing on complete data for ", length(bvars), " variables")

    ok <- clean %>%
      dplyr::filter(stats::complete.cases(dplyr::across(dplyr::all_of(bvars)))) %>%
      dplyr::group_by(ISO3) %>%
      dplyr::summarise(n = dplyr::n_distinct(year), .groups = "drop") %>%
      dplyr::filter(n == n_years_req) %>%
      dplyr::pull(ISO3)

  } else {
    stop("BALANCE_MODE must be 'presence' or 'complete'")
  }

  dropped <- setdiff(unique(clean$ISO3), ok)
  clean <- clean[clean$ISO3 %in% ok, ]
  message("balanced panel (", BALANCE_MODE, "): kept ", length(ok),
          " countries, dropped ", length(dropped))
  if (length(dropped)) message("  dropped: ", paste(sort(dropped), collapse = ", "))
}

clean <- clean[order(clean$ISO3, clean$year), ]

message("final: ", nrow(clean), " rows x ", ncol(clean), " cols, ",
        dplyr::n_distinct(clean$ISO3), " countries, years ",
        min(clean$year), "-", max(clean$year))

# =============================================================================
# SAVE
# =============================================================================
writexl::write_xlsx(setNames(list(clean), OUT_SHEET), path = OUT_XLSX)
message("wrote ", OUT_XLSX, " [", OUT_SHEET, "]")

if (ALSO_WRITE_CSV) {
  out_csv <- sub("\\.xlsx$", ".csv", OUT_XLSX)
  write.csv(clean, out_csv, row.names = FALSE, na = "")
  message("wrote ", out_csv)
}
