# Tableau Dashboard Build Guide

## US Flight Delay Rate Analysis

This guide explains how to rebuild the Tableau workbook using the refreshed CSV files in:

`sql_exports/`

## 1. Data scope to display

Add the following note to every dashboard:

> Analysis includes operated, non-diverted U.S. flights from 2019 and 2022–2025. Data for 2020 and 2021 were not downloaded, and incomplete 2026 data were excluded. A delayed flight is defined as a flight departing 15 or more minutes late.

The refreshed analytical dataset contains:

- 34,388,401 operated, non-diverted flights
- 7,051,594 flights departing 15 or more minutes late
- A 20.51% departure-delay rate
- 18 airlines
- 385 origin airports
- 8,440 routes

The cancellation and diversion CSV files should not be used. The cleaned dataset contains only operated, non-diverted flights, so it cannot support a valid cancellation or diversion analysis.

## 2. Connect the CSV files

In Tableau Desktop or Tableau Public:

1. Select **Connect → Text File**.
2. Open the first CSV from `sql_exports/`.
3. Select **Data → New Data Source → Text File** for each additional CSV.
4. Give every source a short, readable name.

Recommended names:

| CSV file | Tableau data-source name |
|---|---|
| `22_24_overview_kpi_card_for_operated_non_diverted_flights.csv` | KPI Overview |
| `19_19_overall_delay_rate_by_year.csv` | Year |
| `10_10_delay_rate_by_month.csv` | Month |
| `11_11_delay_rate_by_quarter.csv` | Quarter |
| `12_12_delay_rate_by_day_of_week.csv` | Day of Week |
| `13_13_delay_rate_by_scheduled_departure_hour.csv` | Departure Hour |
| `14_14_peak_vs_off_peak_delay_rate.csv` | Peak Hour |
| `15_15_weekday_vs_weekend_delay_rate.csv` | Weekend |
| `03_01_airline_delay_rate.csv` | Airline Rate |
| `04_02_airline_delay_volume.csv` | Airline Volume |
| `05_03_airline_departure_delay_severity_among_delayed_flights.csv` | Airline Departure Severity |
| `06_04_airline_arrival_delay_severity_among_delayed_flights.csv` | Airline Arrival Severity |
| `20_21_delay_rate_by_airline_and_year.csv` | Airline by Year |
| `08_06_delay_rate_by_origin_airport.csv` | Airports |
| `09_08_delay_rate_by_route.csv` | Routes |
| `16_16_most_common_delay_causes_among_delayed_operated_flights.csv` | Delay Causes |
| `18_18_delay_rate_by_distance_group.csv` | Distance Rate |
| `17_17_delay_severity_by_distance_group_among_delayed_operated_flights.csv` | Distance Severity |

### If you reuse the existing packaged workbook

The current packaged workbook still contains extracts created from older files in `query_results/`. Opening the workbook alone will therefore continue to show the old embedded results.

For each existing worksheet:

1. Add the appropriate refreshed CSV as a new data source.
2. Open the worksheet.
3. Select **Data → Replace Data Source**.
4. Choose the old source under **Current** and its refreshed equivalent under **Replacement**.
5. Confirm that Tableau matched the fields correctly.
6. Repair any field marked with a red exclamation point.
7. Refresh the extract.
8. Save the workbook under a new name before overwriting the previous packaged workbook.

Because several old sources used different field names, creating a clean workbook from the instructions below may be easier and more reliable than replacing every old source.

## 3. Verify field types

Use the following field types:

- **Dimensions:** airline name, airport code, city name, route, delay cause, and distance group
- **Discrete whole numbers:** year, month, quarter, day of week, and departure hour
- **Measures:** flight counts, delayed-flight counts, delay rates, averages, medians, and 90th percentiles

The `delay_rate_pct` and `pct_of_delayed_flights` fields already contain values such as `20.51`, not `0.2051`. Format them as **Number (Custom)** with one or two decimal places and add `%` as a suffix. Do not use Tableau's standard Percentage format because that would incorrectly display `20.51` as `2,051%`.

Format flight counts with display units of **Thousands (K)** or **Millions (M)** and no more than one decimal place.

## 4. Create readable labels

### Month Name

Create:

```tableau
CASE [month]
WHEN 1 THEN "Jan"
WHEN 2 THEN "Feb"
WHEN 3 THEN "Mar"
WHEN 4 THEN "Apr"
WHEN 5 THEN "May"
WHEN 6 THEN "Jun"
WHEN 7 THEN "Jul"
WHEN 8 THEN "Aug"
WHEN 9 THEN "Sep"
WHEN 10 THEN "Oct"
WHEN 11 THEN "Nov"
WHEN 12 THEN "Dec"
END
```

Sort `Month Name` by the original `month` field in ascending order.

### Day Name

The source follows the standard flight-data order in which 1 is Monday and 7 is Sunday:

```tableau
CASE [day_of_week]
WHEN 1 THEN "Mon"
WHEN 2 THEN "Tue"
WHEN 3 THEN "Wed"
WHEN 4 THEN "Thu"
WHEN 5 THEN "Fri"
WHEN 6 THEN "Sat"
WHEN 7 THEN "Sun"
END
```

Sort `Day Name` by `day_of_week`.

### Peak-Hour Label

Peak hours are scheduled departures from **5:00–9:59 a.m.** and **4:00–7:59 p.m.** All remaining scheduled departure hours are off-peak. This matches the `is_peak_hour` field created in `clean_flights.py` and grouped by the SQL export query.

```tableau
IF [is_peak_hour] = 1 THEN "Peak"
ELSE "Off-Peak"
END
```

### Weekend Label

```tableau
IF [is_weekend] = 1 THEN "Weekend"
ELSE "Weekday"
END
```

## 5. Standard rate-and-volume chart

Use this structure for the year, month, quarter, day-of-week, departure-hour, peak-hour, weekend, and distance charts.

1. Place the time or category dimension on **Columns**.
2. Place `SUM(total_operated_flights)` on **Rows**.
3. Set its Marks type to **Bar** and use a light blue color.
4. Place `AVG(delay_rate_pct)` beside it on **Rows**.
5. Right-click the second measure and select **Dual Axis**.
6. Set the delay-rate Marks type to **Line** with circles and use orange.
7. Keep separate axes:
   - Left axis: Number of Flights
   - Right axis: Departure Delay Rate (%)
8. Do **not** synchronize the axes because they use different units.
9. Add these fields to the tooltip:
   - Total operated flights
   - Delayed flights
   - Delay rate
10. Keep the count bars visually lighter than the rate line.

This chart prevents a rate from being interpreted without considering the number of flights behind it.

# Dashboard 1: Executive Overview

Recommended size: **1,200 × 850 pixels**

## Sheet 1: KPI — Total Flights

Source: **KPI Overview**

1. Set Marks to **Text**.
2. Place `SUM(total_operated_flights)` on Text.
3. Format the value as `34.4M`.
4. Add the label `Operated Flights`.

## Sheet 2: KPI — Delayed Flights

1. Duplicate the first KPI sheet.
2. Replace the measure with `SUM(total_delayed_flights)`.
3. Format the value as `7.1M`.
4. Label it `15+ Minutes Late`.

## Sheet 3: KPI — Delay Rate

1. Duplicate the KPI sheet.
2. Use `AVG(overall_delay_rate_pct)`.
3. Format it as `20.51%`.
4. Label it `Departure Delay Rate`.

## Sheet 4: KPI — Airlines

Use `SUM(total_airlines)` and label it `Airlines`.

## Sheet 5: KPI — Airports

Use `SUM(total_origin_airports)` and label it `Origin Airports`.

## Sheet 6: KPI — Routes

Use `SUM(total_routes)` and label it `Routes`.

## Sheet 7: Delayed vs Under 15 Minutes Late

Source: **KPI Overview**

Create:

```tableau
SUM([total_operated_flights]) - SUM([total_delayed_flights])
```

Name it `Under 15 Minutes Late`.

1. Set Marks to **Pie**.
2. Place **Measure Names** on Color.
3. Place **Measure Values** on Angle and Label.
4. Filter Measure Names to:
   - `total_delayed_flights`
   - `Under 15 Minutes Late`
5. Use orange for delayed flights and blue for flights under 15 minutes late.
6. Apply **Percent of Total** to Measure Values on the label.
7. Title the sheet `Departure Delay Classification`.

The blue group is not necessarily exactly on time. It includes every flight departing less than 15 minutes late, including early departures and delays of 1–14 minutes.

## Sheet 8: Delay Rate and Flight Count by Year

Source: **Year**

Use the standard rate-and-volume chart:

- Columns: `year`
- Bars: `SUM(total_operated_flights)`
- Line: `AVG(delay_rate_pct)`

Keep the missing 2020 and 2021 years visible as a coverage note rather than implying a continuous annual trend.

## Executive Overview layout

1. Add a horizontal container at the top.
2. Place the six KPI cards inside it.
3. Add a second horizontal container below it.
4. Place the departure-delay classification chart on the left.
5. Place the yearly rate-and-volume chart on the right.
6. Add the data-scope note at the bottom.

# Dashboard 2: Seasonal and Hourly Patterns

Recommended size: **1,200 × 900 pixels**

## Sheet 9: Delay Rate and Flight Count by Month

Source: **Month**

- Columns: `Month Name`
- Bars: `SUM(total_operated_flights)`
- Line: `AVG(delay_rate_pct)`
- Title: `Departure Delay Rate and Flight Count by Month`

## Sheet 10: Delay Rate and Flight Count by Quarter

Source: **Quarter**

- Columns: `quarter`
- Bars: `SUM(total_operated_flights)`
- Line: `AVG(delay_rate_pct)`
- Title: `Departure Delay Rate and Flight Count by Quarter`

Edit the displayed quarter aliases to Q1, Q2, Q3, and Q4.

## Sheet 11: Delay Rate and Flight Count by Day of Week

Source: **Day of Week**

- Columns: `Day Name`
- Bars: `SUM(total_operated_flights)`
- Line: `AVG(delay_rate_pct)`
- Title: `Departure Delay Rate and Flight Count by Day of Week`

## Sheet 12: Delay Rate and Flight Count by Scheduled Departure Hour

Source: **Departure Hour**

- Columns: `dep_hour`
- Bars: `SUM(total_operated_flights)`
- Line: `AVG(delay_rate_pct)`
- Title: `Departure Delay Rate and Flight Count by Scheduled Departure Hour`

Keep all 24 hours. The very low overnight flight counts provide important context for interpreting the overnight delay rates.

## Sheet 13: Peak vs Off-Peak

Source: **Peak Hour**

- Columns: `Peak-Hour Label`
- Bars: `SUM(total_operated_flights)`
- Line or circles: `AVG(delay_rate_pct)`
- Title: `Peak vs Off-Peak Departure Performance`

## Sheet 14: Weekday vs Weekend

Source: **Weekend**

- Columns: `Weekend Label`
- Bars: `SUM(total_operated_flights)`
- Line or circles: `AVG(delay_rate_pct)`
- Title: `Weekday vs Weekend Departure Performance`

## Seasonal dashboard layout

Use a 2 × 2 tiled layout:

- Top left: Month
- Top right: Quarter
- Bottom left: Day of week
- Bottom right: Scheduled departure hour

Place the peak/off-peak and weekday/weekend comparisons in a smaller row beneath the main grid if space permits.

# Dashboard 3: Airline Performance

Recommended size: **1,200 × 900 pixels**

## Sheet 15: Delay Rate by Airline

Source: **Airline Rate**

1. Place `airline_name` on Rows.
2. Place `AVG(delay_rate_pct)` on Columns.
3. Set Marks to Bar.
4. Sort descending by delay rate.
5. Place `SUM(total_operated_flights)` and `SUM(delayed_flights)` in the tooltip.
6. Place the delay rate on Label.
7. Use a sequential color scale.
8. Title the sheet `Departure Delay Rate by Airline`.

## Sheet 16: Delayed-Flight Volume by Airline

Source: **Airline Volume**

1. Place `airline_name` on Rows.
2. Place `SUM(delayed_flights)` on Columns.
3. Sort descending.
4. Place `SUM(total_operated_flights)` and `AVG(delay_rate_pct)` in the tooltip.
5. Title the sheet `Delayed-Flight Volume by Airline`.

Place this chart beside the airline delay-rate chart. A high number of delayed flights may result from high flight volume rather than an unusually high delay rate.

## Sheet 17: Airline Delay Rate by Year

Source: **Airline by Year**

1. Place `year` on Columns.
2. Place `AVG(delay_rate_pct)` on Rows.
3. Set Marks to Line with circles.
4. Place `airline_name` on Color.
5. Place `total_operated_flights` and `delayed_flights` in the tooltip.
6. Show `airline_name` as a single-value or multiple-value filter.
7. Initially display only a small number of airlines to avoid 18 overlapping lines.
8. Title the sheet `Airline Departure Delay Rate by Year`.

## Sheet 18: Departure-Delay Severity by Airline

Source: **Airline Departure Severity**

1. Place `airline_name` on Rows.
2. Place **Measure Values** on Columns.
3. Filter Measure Names to:
   - `avg_dep_delay_when_delayed`
   - `median_dep_delay_when_delayed`
   - `p90_dep_delay_when_delayed`
4. Place Measure Names on Color.
5. Set Marks to Circle or Bar.
6. Place `delayed_flights` in the tooltip.
7. Title the sheet `Departure-Delay Severity Among Delayed Flights`.

Use the median to represent a typical delayed flight and the 90th percentile to represent severe but less exceptional delays.

## Optional Sheet 19: Arrival-Delay Severity by Airline

Repeat the preceding chart using **Airline Arrival Severity** and the arrival-delay measures.

## Airline dashboard actions

Add a filter action:

1. Select **Dashboard → Actions → Add Action → Filter**.
2. Use `Delay Rate by Airline` as the source sheet.
3. Target `Airline Delay Rate by Year` and the severity sheets.
4. Run the action on **Select**.
5. Clearing the selection should show all values.

# Dashboard 4: Airports and Routes

Recommended size: **1,200 × 850 pixels**

The airport export contains airports with at least 200,000 flights. The route export contains routes with at least 50,000 flights. These thresholds prevent very small groups from producing unstable rankings.

## Sheet 20: Airports with the Highest Delay Rates

Source: **Airports**

1. Place `origin` on Rows.
2. Place `AVG(delay_rate_pct)` on Columns.
3. Sort descending.
4. Filter to the top 10 by `AVG(delay_rate_pct)`.
5. Place `origin_city_name`, `total_operated_flights`, `delayed_flights`, and `avg_dep_delay` in the tooltip.
6. Show total operated flights on the label or in the tooltip.
7. Title the sheet `Highest Departure Delay Rates Among High-Volume Airports`.

## Sheet 21: Airports with the Lowest Delay Rates

Duplicate the previous sheet and filter to the bottom 10 by `AVG(delay_rate_pct)`.

Title it `Lowest Departure Delay Rates Among High-Volume Airports`.

## Sheet 22: Routes with the Highest Delay Rates

Source: **Routes**

1. Place `route` on Rows.
2. Place `AVG(delay_rate_pct)` on Columns.
3. Sort descending.
4. Filter to the top 10 by delay rate.
5. Add origin city, destination city, total flights, and delayed flights to the tooltip.
6. Title the sheet `Highest Departure Delay Rates Among High-Volume Routes`.

## Sheet 23: Routes with the Lowest Delay Rates

Duplicate the preceding sheet and filter to the bottom 10.

Title it `Lowest Departure Delay Rates Among High-Volume Routes`.

## Airports and routes layout

Use a 2 × 2 tiled grid:

- Top left: Highest airport delay rates
- Top right: Lowest airport delay rates
- Bottom left: Highest route delay rates
- Bottom right: Lowest route delay rates

Use the same fixed delay-rate axis across each pair of comparable charts whenever practical.

# Dashboard 5: Delay Causes and Severity

Recommended size: **1,200 × 850 pixels**

## Sheet 24: Delay Causes

Source: **Delay Causes**

1. Place `main_delay_cause` on Rows.
2. Place `SUM(delayed_flights)` on Columns.
3. Sort descending.
4. Place `AVG(pct_of_delayed_flights)` on Label.
5. Place both the count and percentage in the tooltip.
6. Title the sheet `Primary Cause Among Delayed Flights`.

The cause categories are mutually exclusive only because the cleaned data assigns one main reported cause to each delayed flight with positive cause-specific minutes. Flights without positive reported cause minutes are labeled `No Reported Cause`. Describe the categories as the selected primary reported cause, not as every factor contributing to a delay.

## Sheet 25: Delay Rate and Flight Count by Distance Group

Source: **Distance Rate**

Use the standard rate-and-volume chart:

- Columns: `distance_group`
- Bars: `SUM(total_operated_flights)`
- Line: `AVG(delay_rate_pct)`
- Title: `Departure Delay Rate and Flight Count by Distance Group`

The numeric prefixes already preserve the correct short-to-long ordering.

## Sheet 26: Departure-Delay Severity by Distance Group

Source: **Distance Severity**

1. Place `distance_group` on Columns.
2. Place Measure Values on Rows.
3. Filter Measure Names to:
   - `avg_dep_delay`
   - `median_dep_delay`
   - `p90_dep_delay`
4. Place Measure Names on Color.
5. Use circles connected by lines or grouped bars.
6. Add `delayed_flights` to the tooltip.
7. Title the sheet `Departure-Delay Severity by Distance Group`.

## Sheet 27: Arrival-Delay Severity by Distance Group

Duplicate the departure-severity sheet and replace its measures with:

- `avg_arr_delay`
- `median_arr_delay`
- `p90_arr_delay`

Title it `Arrival-Delay Severity by Distance Group`.

## Causes and severity layout

- Left half: Delay-cause bar chart
- Top right: Distance delay rate and flight count
- Bottom right: Departure- and arrival-severity charts

# 6. Dashboard-wide formatting

Use a consistent design:

- Blue: flight counts or flights under 15 minutes late
- Orange or red-orange: delay rates and delayed flights
- Dark gray: titles and labels
- White or very light gray: dashboard background

Recommended formatting:

- Dashboard title: 22–26 pt
- Sheet title: 13–16 pt
- KPI value: 24–32 pt
- KPI label: 10–12 pt
- Tooltip numbers: comma-separated
- Delay rates: one or two decimals plus `%`
- Delay minutes: one decimal for averages and whole numbers for medians and percentiles

Remove unnecessary:

- Gridlines
- Zero lines when they do not add meaning
- Repeated legends
- Heavy borders
- Field-name labels that repeat the chart title

# 7. Tooltips

Use a consistent tooltip structure:

```text
Category: <dimension>
Operated flights: <total_operated_flights>
Delayed flights: <delayed_flights>
Departure delay rate: <delay_rate_pct>%
```

For severity charts, add:

```text
Average delay: <average> minutes
Median delay: <median> minutes
90th percentile: <p90> minutes
```

# 8. Final checks before publishing

Confirm the following:

1. No chart displays 2026.
2. The yearly view contains 2019 and 2022–2025.
3. The gap for 2020–2021 is clearly documented.
4. The overall delay rate is 20.51%.
5. The overall operated-flight count is 34,388,401.
6. The overall delayed-flight count is 7,051,594.
7. Delay-rate fields are not accidentally multiplied by 100.
8. Every rate chart also displays its underlying flight count.
9. Airport rankings use the 200,000-flight threshold.
10. Route rankings use the 50,000-flight threshold.
11. Cancellation and diversion charts are not included.
12. Titles consistently say **departure delay rate**, not only **delay rate**.
13. The pie-chart category says **Under 15 Minutes Late**, not **On Time**.

# 9. Recommended workbook order

Publish the dashboards in this order:

1. Executive Overview
2. Seasonal and Hourly Patterns
3. Airline Performance
4. Airports and Routes
5. Delay Causes and Severity

This order moves from the overall result to time patterns, operating entities, geographic comparisons, and finally the causes and severity of delays.
