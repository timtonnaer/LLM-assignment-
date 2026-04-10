/*===========================================================================
  Risk Disclosure Analysis — Stata Starter Do-File

  Data:    final_panel.dta  (2,515 firm-year obs, 311 firms, 2016–2023)
           event_study_data.dta  (2,341 events, CAR from day -5 to +30)

  Sector codes (sector_code variable):
     1  Communication Services    7  Industrials
     2  Consumer Discretionary    8  Information Technology
     3  Consumer Staples          9  Materials
     4  Energy                   10  Real Estate
     5  Financials               11  Utilities
     6  Health Care

  Key variables:
     roa                  Return on assets (current year)
     roa_t1               Return on assets (next year)
     annual_ret_t1        Stock return (next year)
     rev_growth_t1        Revenue growth (next year)
     vagueness_ratio      Share of vague language in new risk text (0–1)
     risk_update_intensity  Count of new/expanded risk sentences
     risk_cyber           Count of cyber risk sentences added
     risk_supply_chain    Count of supply chain risk sentences added
     risk_operational     Count of operational risk sentences added
     risk_regulatory      Count of regulatory risk sentences added
     risk_financing       Count of financing risk sentences added
     risk_macroeconomic   Count of macroeconomic risk sentences added
     leverage             Long-term debt / total assets
     log_assets           Log of total assets
===========================================================================*/

clear all
set more off
cd "C:/YOUR/PATH/HERE"          // <-- change this to your folder


/*---------------------------------------------------------------------------
  1. LOAD DATA
---------------------------------------------------------------------------*/
use "final_panel.dta", clear

* Keep only firm-years with LLM classifications
keep if n_classified > 0

* Label sector codes
label define sector_lbl  ///
    1 "Communication Services" 2 "Consumer Discretionary"  ///
    3 "Consumer Staples"       4 "Energy"                   ///
    5 "Financials"             6 "Health Care"              ///
    7 "Industrials"            8 "Information Technology"   ///
    9 "Materials"             10 "Real Estate"             ///
   11 "Utilities"
label values sector_code sector_lbl

* Panel setup (cik is string — encode to numeric first)
encode cik, gen(firm_id)
xtset firm_id year_new


/*---------------------------------------------------------------------------
  2. SUMMARY STATISTICS
---------------------------------------------------------------------------*/
estpost summarize roa roa_t1 annual_ret_t1 rev_growth_t1 ///
    vagueness_ratio risk_update_intensity                 ///
    risk_cyber risk_supply_chain risk_operational         ///
    risk_regulatory risk_financing risk_macroeconomic     ///
    leverage log_assets, detail

* Or a simple tabstat:
tabstat roa roa_t1 vagueness_ratio risk_update_intensity  ///
        risk_cyber risk_supply_chain leverage log_assets, ///
    stats(n mean sd min p25 p50 p75 max) col(stats) format(%8.4f)


/*---------------------------------------------------------------------------
  3. KEY REGRESSIONS (replicating Python OLS results)
  All use HC3-equivalent robust SEs: vce(robust)
---------------------------------------------------------------------------*/

* --- Reg 1: Future ROA ~ vagueness + intensity + controls + sector FE
reg roa_t1 vagueness_ratio risk_update_intensity leverage log_assets ///
    i.sector_code, vce(robust)
estimates store reg1

* --- Reg 2: Future ROA ~ risk type counts + controls + sector FE
reg roa_t1 risk_cyber risk_supply_chain risk_operational  ///
    risk_regulatory risk_financing risk_macroeconomic     ///
    leverage log_assets i.sector_code, vce(robust)
estimates store reg2

* --- Reg 3: vagueness only → current ROA (strongest single-variable result)
reg roa vagueness_ratio leverage log_assets i.sector_code, vce(robust)
estimates store reg3

* --- Reg 4: vagueness only → future ROA
reg roa_t1 vagueness_ratio leverage log_assets i.sector_code, vce(robust)
estimates store reg4

* --- Reg 5: Interaction — vagueness × risk_financing → future ROA
gen vag_x_financing = vagueness_ratio * risk_financing
reg roa_t1 vagueness_ratio risk_financing vag_x_financing ///
    leverage log_assets i.sector_code, vce(robust)
estimates store reg5

* --- Reg 6: Interaction — vagueness × risk_regulatory → future ROA
gen vag_x_regulatory = vagueness_ratio * risk_regulatory
reg roa_t1 vagueness_ratio risk_regulatory vag_x_regulatory ///
    leverage log_assets i.sector_code, vce(robust)
estimates store reg6

* Output table (requires estout: ssc install estout)
* esttab reg1 reg2 reg3 reg4 using "regression_table.rtf", ///
*     replace b(4) se(4) star(* 0.10 ** 0.05 *** 0.01)    ///
*     title("Risk Disclosure and Future ROA") nogaps


/*---------------------------------------------------------------------------
  4. FIGURES
---------------------------------------------------------------------------*/

* ── Figure 1: Vagueness ratio vs future ROA (scatter + fit line) ────────────
twoway ///
    (scatter roa_t1 vagueness_ratio if sector_code==2, mcolor("31 119 180") msymbol(circle) msize(small) malpha(40)) ///
    (scatter roa_t1 vagueness_ratio if sector_code==5, mcolor("255 127 14") msymbol(circle) msize(small) malpha(40)) ///
    (scatter roa_t1 vagueness_ratio if sector_code==6, mcolor("44 160 44")  msymbol(circle) msize(small) malpha(40)) ///
    (scatter roa_t1 vagueness_ratio if sector_code==8, mcolor("148 103 189") msymbol(circle) msize(small) malpha(40)) ///
    (lfit roa_t1 vagueness_ratio, lcolor(red) lwidth(medthick) lpattern(dash)) ///
    , legend(order(1 "Consumer Disc." 2 "Financials" 3 "Health Care" 4 "IT" 5 "OLS trend") ///
             size(small) rows(2)) ///
    xtitle("Vagueness Ratio (year t)") ytitle("Future ROA (year t+1)") ///
    title("Vaguer Risk Disclosures Predict Lower Future ROA") ///
    note("coef = -0.030, p = 0.004  |  N = 2,376  |  Controls: leverage, log assets, sector FE") ///
    scheme(s2color)
graph export "fig1_vagueness_vs_roa.png", replace width(1400)


* ── Figure 2: Bar chart — cyber disclosers vs non-disclosers ────────────────
gen cyber_disc = (risk_cyber > 0) if !missing(risk_cyber)
label define cyber_lbl 0 "Did NOT disclose cyber risk" 1 "DID disclose cyber risk"
label values cyber_disc cyber_lbl

graph bar roa_t1, over(cyber_disc) ///
    bar(1, color("149 179 215")) bar(2, color("26 110 176")) ///
    blabel(bar, format(%5.3f) size(medsmall)) ///
    ytitle("Mean Future ROA (year t+1)") ///
    title("Firms Disclosing Cyber Risk Earn Higher Future Profits") ///
    note("OLS coef = +0.0037, p = 0.002  |  N = 2,515") ///
    scheme(s2color)
graph export "fig2_cyber_vs_roa.png", replace width(900)


* ── Figure 3: Bar chart — supply chain disclosers vs non-disclosers ─────────
gen sc_disc = (risk_supply_chain > 0) if !missing(risk_supply_chain)
label define sc_lbl 0 "Did NOT disclose supply chain risk" 1 "DID disclose supply chain risk"
label values sc_disc sc_lbl

graph bar roa_t1, over(sc_disc) ///
    bar(1, color("180 215 180")) bar(2, color("35 139 69")) ///
    blabel(bar, format(%5.3f) size(medsmall)) ///
    ytitle("Mean Future ROA (year t+1)") ///
    title("Firms Disclosing Supply Chain Risk Earn Higher Future Profits") ///
    note("OLS coef = +0.0040, p = 0.040  |  N = 2,515") ///
    scheme(s2color)
graph export "fig3_supplychain_vs_roa.png", replace width(900)


* ── Figure 4: Vague vs concrete language bar chart ──────────────────────────
* Split at median vagueness
summarize vagueness_ratio, detail
local med_vag = r(p50)
gen concrete = (vagueness_ratio <= `med_vag') if !missing(vagueness_ratio)
label define conc_lbl 0 "Vague language (above median)" 1 "Concrete language (below median)"
label values concrete conc_lbl

graph bar roa_t1, over(concrete) ///
    bar(1, color("192 57 43")) bar(2, color("26 110 176")) ///
    blabel(bar, format(%5.3f) size(medsmall)) ///
    ytitle("Mean Future ROA (year t+1)") ///
    title("Concrete Risk Language Predicts Better Future Performance") ///
    note("OLS coef = -0.023, p = 0.061  |  N = 2,376") ///
    scheme(s2color)
graph export "fig4_vague_vs_concrete.png", replace width(900)


* ── Figure 5: Risk update intensity over time (line chart) ──────────────────
collapse (mean) risk_update_intensity, by(year_new sector_code)

twoway ///
    (connected risk_update_intensity year_new if sector_code==2, lcolor("31 119 180")  msymbol(circle)) ///
    (connected risk_update_intensity year_new if sector_code==5, lcolor("255 127 14")  msymbol(square)) ///
    (connected risk_update_intensity year_new if sector_code==6, lcolor("44 160 44")   msymbol(triangle)) ///
    (connected risk_update_intensity year_new if sector_code==7, lcolor("214 39 40")   msymbol(diamond)) ///
    (connected risk_update_intensity year_new if sector_code==8, lcolor("148 103 189") msymbol(plus)) ///
    , legend(order(1 "Consumer Disc." 2 "Financials" 3 "Health Care" 4 "Industrials" 5 "IT") ///
             size(small)) ///
    xtitle("Year") ytitle("Avg. Risk Update Intensity") ///
    title("Risk Update Intensity Over Time by Sector") ///
    xline(2020, lcolor(red) lpattern(dash) lwidth(thin)) ///
    note("Dashed line = 2020 (COVID)") ///
    scheme(s2color)
graph export "fig5_intensity_timeseries.png", replace width(1400)


/*---------------------------------------------------------------------------
  5. EVENT STUDY FIGURES (using event_study_data.dta)
---------------------------------------------------------------------------*/
use "event_study_data.dta", clear

* Rename CAR columns — the + was replaced with p and - with m
* e.g. car_dp10 = CAR at day +10, car_dm5 = CAR at day -5

* Split into high vs low intensity at median
summarize risk_update_intensity, detail
local med_int = r(p50)
gen intensity_group = (risk_update_intensity > `med_int')
label define int_lbl 0 "Low Intensity" 1 "High Intensity"
label values intensity_group int_lbl

* Reshape to long format for plotting
keep cik year_new intensity_group car_dm5 car_dm4 car_dm3 car_dm2 car_dm1 ///
     car_dp0 car_dp1 car_dp2 car_dp3 car_dp4 car_dp5 ///
     car_dp10 car_dp15 car_dp20 car_dp25 car_dp30

reshape long car_d, i(cik year_new) j(day_str) string

* Convert day string to numeric
gen day = real(subinstr(subinstr(day_str, "p", "", .), "m", "-", .))

* Mean CAR by day and group
collapse (mean) mean_car=car_d (semean) se_car=car_d, by(intensity_group day)

gen ci_hi = mean_car + 1.96 * se_car
gen ci_lo = mean_car - 1.96 * se_car

* ── Event study: High vs Low intensity CAR drift ────────────────────────────
twoway ///
    (rarea ci_hi ci_lo day if intensity_group==1, color("26 110 176%20") ) ///
    (rarea ci_hi ci_lo day if intensity_group==0, color("127 127 127%15") ) ///
    (connected mean_car day if intensity_group==1, lcolor("26 110 176") lwidth(medthick) msymbol(none)) ///
    (connected mean_car day if intensity_group==0, lcolor("127 127 127") lwidth(medthick) lpattern(dash) msymbol(none)) ///
    , legend(order(3 "High Intensity" 4 "Low Intensity") size(medsmall)) ///
    xline(0, lcolor(black) lpattern(shortdash)) ///
    yline(0, lcolor(gray) lwidth(thin)) ///
    xtitle("Trading Days Relative to 10-K Filing") ///
    ytitle("Cumulative Abnormal Return (CAR)") ///
    title("Market Reaction: High vs Low Risk Disclosure Intensity") ///
    note("Shaded bands = 95% CI  |  N = 2,341 events") ///
    scheme(s2color)
graph export "fig6_event_study_car.png", replace width(1400)
