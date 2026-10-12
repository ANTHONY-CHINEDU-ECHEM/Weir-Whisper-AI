# Weir Whisper AI

### Multi agent GraphRAG investigator for tray weeping and separation efficiency loss in distillation columns

<p align="center">
  <img src="docs/figures/architecture.png" width="100%" alt="Weir Whisper AI architecture">
</p>

<table>
<tr><td><b>Domain</b></td><td>Refining and petrochemicals, distillation operations, process reliability</td></tr>
<tr><td><b>Problem</b></td><td>Tray weeping silently destroys separation efficiency and energy, and the evidence needed to explain it is scattered across six systems</td></tr>
<tr><td><b>Approach</b></td><td>Physics informed soft sensors, a gradient boosted detector, a plant knowledge graph, hybrid GraphRAG retrieval, a Bayesian multi agent investigator and a verifier that refuses ungrounded claims</td></tr>
<tr><td><b>Data</b></td><td>One simulated year of a six column site: 12.6 million historian values, 7,279 alarms, 5,921 shift log entries, 545 work orders, 12,732 lab results, gamma scans, MOCs and an engineering library</td></tr>
<tr><td><b>Headline results</b></td><td>27 of 27 weeping scenarios detected with a median onset delay of five minutes; correct root cause on 54 of 54 matched episodes; yearly cost estimated within 18.5 percent of the true counterfactual; 100 percent of 1,296 report claims verified</td></tr>
<tr><td><b>Stack</b></td><td>Python, NumPy, pandas, scikit learn, SciPy, NetworkX, FastAPI, Matplotlib, optional Anthropic Claude backend</td></tr>
</table>

***

## Project brief

A distillation tray only works when vapour rising through its holes is fast enough to hold up the liquid flowing across it. When the vapour slows down, liquid drains straight through the perforations instead of crossing the deck and contacting the vapour. Engineers call this weeping. A lightly weeping tray still separates; a heavily weeping tray behaves as if it were not there. The column keeps running, nothing trips, no alarm says "weeping", but product purity drifts, the advanced controller pushes more reflux and more reboiler steam to compensate, and the site pays for it twice: once in energy and again in product that has to be downgraded or reprocessed. Weeping is caused by many different things. A feed rate cut, a sticking steam valve, a fouled reboiler, a pressure setpoint change, a cold feed after a preheater trips, or valve units knocked out of a tray during an upset all end in the same symptom, and each needs a completely different fix.

The investigation that follows an efficiency loss event is slow, manual and often inconclusive. The historian holds the trends, the DCS holds the alarms and operator moves, the shift logs hold what operators saw in their own words (with their own spellings of every tag), the maintenance system holds the work orders, the laboratory system holds the quality results, and the evidence that a valve has failed three times this year sits in closure notes nobody reads together. An engineer has to pull all of this into one picture, decide between half a dozen plausible causes, put a cost on the event and check whether it has happened before. In practice the cheap, visible cause gets blamed, the expensive recurring one is missed, and the same valve is repacked again. The industry phrase for it is "we fixed the symptom".

Weir Whisper AI is built to do that investigation the way a senior process engineer would, and to show its working. It reads one year of plant data across six columns, detects every weeping episode and every look alike (flooding, foaming, analyzer drift), builds a knowledge graph that links every alarm, log entry, work order, lab result, scan and MOC to the asset and the time window it belongs to, and then hands each episode to a team of agents. A signal analyst reads the hydraulics, a records analyst reads the human evidence through hybrid GraphRAG retrieval, a root cause agent weighs nine competing hypotheses, an impact agent prices the event against a normal behaviour model, and a recurrence agent links it to earlier failures on the same asset. Every sentence of the final report cites the records it rests on, every number must exist in a registry of computed facts, and a verifier deletes anything that does not pass. The name comes from the outlet weir on every tray: the column always tells you what is wrong, if you listen to all of it at once.

Because a real plant never tells you the true cause or the true cost of an event, the project ships its own physics based plant simulator. It runs every column three times (as it happened, with weeping switched off, and fully healthy) so the true cost of each event is known exactly, and every stage of the system can be scored against ground truth instead of against opinion.

***

## Table of contents

1. Business questions
2. The dataset
3. How the system works
4. Results
5. What the investigations found
6. A report, start to finish
7. Running the project
8. Repository structure
9. Testing and quality
10. Limitations and honest caveats
11. Extending to a real site

***

## Business questions

1. When did each column lose separation efficiency, and was it tray weeping or something that only looks like it?
2. Why did it happen, and which of the competing explanations can be ruled out with evidence?
3. What did each event cost in steam, off specification product and carbon?
4. Which causes keep coming back on the same asset, and what has the site spent on them?
5. Can every statement in an investigation be traced to a record a reviewer can open?

***

## The dataset

### Why a simulator

Real distillation incident data with labelled causes and known counterfactual costs does not exist in the public domain. The project therefore generates a complete, internally consistent site: the Brackmere Olefins and Refining Complex (fictional). The generator is a dynamic model, not a random number table. Every column steps every five minutes through the same physics and control loops that make weeping happen and make it hard to diagnose.

### The plant

<table>
<tr><th>Column</th><th>Service</th><th>Trays</th><th>Tray type</th><th>Design feed (t/h)</th><th>Design weep margin</th></tr>
<tr><td>C101</td><td>Debutaniser</td><td>36</td><td>moving valve</td><td>118</td><td>1.42</td></tr>
<tr><td>C201</td><td>Depropaniser</td><td>45</td><td>sieve</td><td>64</td><td>1.30</td></tr>
<tr><td>C301</td><td>Deethaniser</td><td>40</td><td>moving valve</td><td>41</td><td>1.55</td></tr>
<tr><td>C401</td><td>C3 Splitter</td><td>96</td><td>sieve</td><td>36</td><td>1.33</td></tr>
<tr><td>C501</td><td>Naphtha Splitter</td><td>32</td><td>fixed valve</td><td>135</td><td>1.31</td></tr>
<tr><td>C601</td><td>Condensate Stabiliser</td><td>28</td><td>sieve</td><td>88</td><td>1.40</td></tr>
</table>

Hole area and flooding capacity are derived from each design basis so that the design weep margin and flood fraction are honoured exactly at design feed. The derived geometry is written into a datasheet for every column, which the knowledge library and the soft sensor both use.

### Physics inside the simulator

The tray hydraulics follow the classical sieve tray method (Coulson and Richardson): the weep point minimum hole velocity from the K2 coefficient and the clear liquid height, the Francis weir crest, the dry plate and residual heads, and a Souders Brown flooding capacity. Each section (rectifying and stripping) gets its own vapour and liquid traffic from the reboiler duty, the reflux and the feed thermal condition q. The weep fraction feeds into Murphree efficiency, efficiency feeds a shortcut separation model, and composition responds with a thirty minute lag.

Three control loops are closed at every step, because they are what turns a simple cause into a confusing investigation:

1. **Steam flow control** through a valve with a two parameter stick and slip model, so a sticking valve produces the real sawtooth output and square wave flow.
2. **Sensitive tray temperature control** of boilup that is not pressure compensated, so raising column pressure quietly cuts steam.
3. **Overhead composition control** from the online analyzer, so a drifting analyzer burns steam with no real quality problem.

Every column also carries hidden as built deviations that the soft sensor never sees: the true hole area differs from the datasheet by up to twelve percent, the true weep coefficient is shifted, the reflux meter is biased, the steam meter drifts seasonally, the differential pressure transmitters drift and get re zeroed, and the feed latent heat wanders with feed composition. This is what keeps physics alone from being a perfect detector.

### Scenarios

<table>
<tr><th>Cause</th><th>Count</th><th>Mechanism in the simulator</th></tr>
<tr><td>Feed turndown</td><td>13</td><td>Feed cut to 45 to 58 percent of design with the steam ratio left in charge</td></tr>
<tr><td>Steam valve stiction</td><td>6</td><td>Deadband of 17 to 24 percent and slip jump on the reboiler valve, recurring on C201 and C501</td></tr>
<tr><td>Reboiler fouling</td><td>2</td><td>Heat transfer coefficient falls over six to ten weeks until the valve saturates</td></tr>
<tr><td>Tray damage</td><td>1</td><td>High pressure trip and relief valve lift, then 60 to 110 percent more open area in one section until repair</td></tr>
<tr><td>Pressure rise</td><td>3</td><td>Condenser limitation or an MOC raises pressure 22 to 34 percent without moving the temperature setpoint</td></tr>
<tr><td>Subcooled feed</td><td>3</td><td>Feed preheater out of service, feed q rises and the rectifying section is starved of vapour</td></tr>
<tr><td>Flooding (look alike)</td><td>5</td><td>High rate campaigns to 113 to 122 percent of design</td></tr>
<tr><td>Foaming (look alike)</td><td>2</td><td>Contamination raises and destabilises dP and costs efficiency</td></tr>
<tr><td>Analyzer drift (look alike)</td><td>3</td><td>Positive bias ramp on the overhead analyzer that the APC believes</td></tr>
</table>

Scenarios are placed after a 24 day fault free commissioning window, never overlap on one column and are spread across a full year from 1 October 2025 to 30 September 2026.

### What the generator writes

<table>
<tr><th>Source</th><th>Volume</th><th>Content</th></tr>
<tr><td>Historian</td><td>6 x 105,120 rows, 12,614,400 values</td><td>Twenty tags per column at five minutes: flows, temperatures, section dP, pressure, levels, analyzers, steam valve output and chest pressure</td></tr>
<tr><td>Alarm journal</td><td>7,279 alarms</td><td>Debounced DCS alarms with limits rationalised on the commissioning window, including realistic alarm floods during valve cycling</td></tr>
<tr><td>Operator actions</td><td>63 actions</td><td>Setpoint, mode and bypass changes</td></tr>
<tr><td>Shift logs</td><td>1,460 logs, 5,921 items</td><td>Free text from two console areas and four crews, with tag aliases (FV207, FV 207, fv207), typos, routine noise, neighbouring column notes and six percent deliberate misattribution</td></tr>
<tr><td>Work orders</td><td>545 orders</td><td>Corrective, preventive, inspection and engineering orders with failure codes and closure notes that escalate on repeat failures</td></tr>
<tr><td>Laboratory</td><td>12,732 results</td><td>Overhead and bottoms samples every eight hours, contaminant screens, analyzer validations</td></tr>
<tr><td>Gamma scans</td><td>11 reports</td><td>Damage findings, healthy scans that rule damage out, and baselines</td></tr>
<tr><td>MOC records</td><td>11 records</td><td>Including the pressure setpoint changes that start pressure driven weeping</td></tr>
<tr><td>Engineering library</td><td>22 documents</td><td>Weeping fundamentals, a symptom matrix, guides for every cause, the low rate procedure, three historical investigations and six generated datasheets</td></tr>
<tr><td>Ground truth</td><td>per sample and per scenario</td><td>True weep fraction, efficiency, flood fraction and composition; scenario windows; true costs from the counterfactual runs; the evidence map linking each scenario to every record it produced</td></tr>
</table>

The ground truth folder is read only by the detector (as labelled history) and the evaluation suite. Nothing in the investigation path can see it.

<p align="center">
  <img src="docs/figures/year_overview_c201.png" width="100%" alt="A year of C201 data with scenario windows">
</p>

The year on C201 shows why this is hard. Four stiction episodes hide in steam noise that looks much like normal operation at this scale, the differential pressure transmitter drifts upward over the year, and the two turndowns are only obvious because feed drops. The stripping section dP residual is the cleanest single signal, but it moves for flooding and transmitter drift as well.

<p align="center">
  <img src="docs/figures/cause_signatures.png" width="100%" alt="Fingerprints of four weeping causes">
</p>

Each cause leaves its own fingerprint once you look at the right tags: the sawtooth valve output against square wave steam flow for stiction, the collapse of both section dP values at turndown, the valve climbing for weeks and pinning at 100 percent before a fouled reboiler gives up, and the rectifying section starving while the stripping section holds when the feed goes cold. The shaded bands are the true weeping periods from the simulator.

***

## How the system works

### 1. Ingestion and data quality

`weir_whisper/ingest` loads every source into one store, normalises timestamps, maps historian tags to measurement roles through the instrument index, and runs data quality checks: sampling regularity, missing values, flatlined tags, physical limits, duplicate identifiers and work orders closed before they were opened. A steam valve pinned at 100 percent is reported as saturation rather than as a dead sensor, because on this plant it is a symptom.

### 2. Hydraulic soft sensor and normal behaviour models

`weir_whisper/analytics/softsensor.py` turns raw tags into the quantities an engineer reasons with. Stripping vapour comes from steam flow and latent heats; rectifying vapour is the average of the overhead material balance and the reboiler balance corrected for feed q. From these it computes the weep margin and flood fraction for each section with the datasheet geometry, the expected tray pressure drop, the F factor, and an apparent tray efficiency by inverting the shortcut separation model with the measured reflux, boilup and analyzer readings.

Normal behaviour models are fitted per column on the commissioning window: expected steam from feed, expected temperature spread, calibration factors for each dP transmitter and a reboiler heat transfer index. Every feature handed to the models is a ratio or a residual against these, which is what makes one model work across a 1.2 metre deethaniser and a 3.4 metre C3 splitter.

### 3. Detection and episode segmentation

`weir_whisper/analytics/detector.py` trains a histogram gradient boosting classifier over 35 dimensionless features to predict five states per sample: normal, weeping, flooding, foaming and analyzer fault. Training keeps every abnormal sample and a random share of normal ones, then rebalances with class weights. Three protocols are run:

1. **Forward split**: train before 1 June 2026, test on June to September. This is the headline number.
2. **Blocked temporal cross fitting**: four folds with a one day embargo either side, so every month is scored by a model that never saw it. These out of fold probabilities drive the investigations.
3. **Leave one column out**: train on five columns, score the sixth.

Two baselines run on the same split: a pure physics rule (estimated weep margin below one or section dP fifteen percent below expected) and an unsupervised isolation forest. Probabilities become episodes through smoothing, hysteresis, a thirty minute minimum and gap merging; weeping segments closer than 36 hours are consolidated into one incident so intermittent stiction weeping becomes one investigation, not twenty.

### 4. Plant knowledge graph

`weir_whisper/graph/builder.py` builds a NetworkX multigraph of 14,832 nodes and 39,551 edges: site, areas, columns, 90 equipment items, 84 instruments, every evidence record, 74 detected episodes, 16 symptom families, 9 cause hypotheses and the library. Edge types are PART_OF, MENTIONS (from the entity linker), ON_ASSET, REPORTS (record to symptom), INDICATES (symptom to cause, weighted from the symptom matrix), DESCRIBES (library to cause), DETECTED_ON, OVERLAPS (episode to every record in its window, weighted by temporal distance) and PRECEDES.

The entity linker in `weir_whisper/nlp/entities.py` resolves every tag shaped token against the site registry whatever the spelling, resolves nicknames such as "the depropaniser", keeps unknown tags as unresolved mentions, and tags text with sixteen symptom families. It knows that a "seal weep at the gland" is a leak, not tray weeping.

### 5. Hybrid GraphRAG retrieval

`weir_whisper/retrieval` indexes 9,580 documents: shift log items, work orders, gamma scans, MOCs, operator actions, flagged lab results, hourly alarm bursts (each keeping the identifiers of its underlying alarms) and library chunks split on headings. Four channels are fused with reciprocal rank fusion:

1. **BM25**, implemented directly over a sparse matrix, with tag normalisation so FV207 matches the same tag written with a dash, a space or in lower case.
2. **Dense**, latent semantic analysis over word n grams and character n grams, which absorbs typos such as "proflie" and matches "hunting" to "cycling". A sentence transformer encoder can be switched in with one environment variable.
3. **Graph and time**, records attached to the same column in the knowledge graph inside the investigation window, ranked by temporal proximity.
4. **Knowledge**, library chunks that share symptom families with the query.

A linear reranker scores the fused candidates on interpretable features: fused rank, lexical and semantic scores, same asset, temporal proximity, inside window, entity overlap, symptom overlap and record type. The default weights are engineering judgement; the evaluation suite refits them on investigations before June and tests them on the rest.

### 6. The agents

`weir_whisper/agents` holds six agents that share an evidence index (what can be cited) and a fact registry (what numbers can be stated):

<table>
<tr><th>Agent</th><th>Job</th></tr>
<tr><td>Signal analyst</td><td>Turns the episode signature into named observations (feed, weep margin, section dP, steam cycling, valve saturation, pressure, feed temperature, trip history, flood, level noise, analyzer versus lab, apparent efficiency, detector mix), each with log likelihood ratios over the nine hypotheses that encode the symptom matrix</td></tr>
<tr><td>Records analyst</td><td>Collects every record the graph ties to the window plus the reranked retrieval hits, extracts symptom families, and converts structured cues (failure codes, preheater bypasses, pressure MOCs, trips, gamma scan findings, contaminant results) into evidence, weighted below hard signals because operators sometimes blame the wrong cause</td></tr>
<tr><td>Root cause</td><td>Evidence weighted Bayesian scoring over turndown, valve stiction, reboiler fouling, tray damage, pressure rise, subcooled feed, flooding, foaming and analyzer drift; reports the strongest support for the winner and the strongest evidence against each alternative</td></tr>
<tr><td>Impact</td><td>Steam against the normal model, off specification tonnage from a one hour analyzer composite (switching to interpolated lab results when the analyzer disagrees with the lab), carbon, and the apparent efficiency loss</td></tr>
<tr><td>Recurrence</td><td>Assigns each diagnosis to a family keyed on the physical asset, counts earlier episodes, earlier work orders and historical investigations, and rolls families into a register with intervals and cumulative cost</td></tr>
<tr><td>Writer and verifier</td><td>The writer turns the analysis into claims that cite evidence identifiers. The verifier checks that every claim has a citation, that every cited record exists, that time stamped evidence sits inside the window, and that every number matches the fact registry at the displayed precision; claims that fail are removed and listed</td></tr>
</table>

The writer runs offline by default: deterministic templates, no network, fully reproducible. With `WEIR_LLM=anthropic` and an `ANTHROPIC_API_KEY`, Claude writes the summary, the root cause explanation and the recommendations as JSON claims from the evidence index and the fact registry. The verifier then filters the model output exactly as it filters the templates, and any section that comes back empty falls back to the offline text. The model can improve the prose; it cannot add a fact.

***

## Results

All numbers below come from `python run.py all` with the default profile and seed. They are written to `outputs/metrics`, and a copy of that run is committed in `docs/metrics` so every figure in this section can be traced.

### Detection

<p align="center">
  <img src="docs/figures/detection_pr.png" width="62%" alt="Precision recall on the held out period">
</p>

<table>
<tr><th>Held out period (June to September)</th><th>Average precision</th><th>ROC AUC</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
<tr><td>Gradient boosted detector</td><td>0.998</td><td>1.000</td><td>0.994</td><td>0.937</td><td>0.965</td></tr>
<tr><td>Physics rule baseline</td><td>0.983</td><td>0.999</td><td>0.825</td><td>0.980</td><td>0.896</td></tr>
<tr><td>Isolation forest baseline</td><td>0.694</td><td>0.986</td><td>0.627</td><td>0.961</td><td>0.759</td></tr>
<tr><td>Detector on a column it never saw (pooled)</td><td>0.864</td><td>0.995</td><td>0.876</td><td>0.361</td><td>0.511</td></tr>
</table>

Thresholds are chosen on the training period only (out of fold scores for the learned models) and then frozen. The physics rule is a strong baseline, which is the point of a good soft sensor: it already ranks samples almost perfectly. What it cannot do is hold precision, because every drifting transmitter and every column with a biased hole area moves its operating point. The learned detector keeps precision at 0.994 at almost the same recall and adds the look alike classes. Scored column by column, the leave one column out model reaches average precision between 0.948 and 0.998, so the features transfer; the pooled number is lower because a frozen threshold is not calibrated for a column the model has never seen, which is exactly why a new column should get a short calibration window before alerts go live.

<table>
<tr><th>Event level (whole year, out of fold)</th><th>Value</th></tr>
<tr><td>Weeping scenarios with material weeping detected</td><td>27 of 27</td></tr>
<tr><td>Detected weeping episodes that were real weeping</td><td>48 of 54 (88.9 percent)</td></tr>
<tr><td>of which unscheduled light weeping during routine feed dips on low margin columns</td><td>15</td></tr>
<tr><td>False alarm episodes</td><td>6, or 0.08 per column per month</td></tr>
<tr><td>Median onset delay after the first weeping sample</td><td>5 minutes (90th percentile 1.2 hours)</td></tr>
<tr><td>Look alike scenarios detected as look alikes</td><td>10 of 10</td></tr>
<tr><td>Look alike scenarios also flagged as weeping</td><td>1 (a flooding campaign that ended in a weeping dip)</td></tr>
<tr><td>Five class macro F1, whole year</td><td>0.959</td></tr>
</table>

The fifteen unscheduled detections deserve a word. The C401 splitter and C201 depropaniser run with small weep margins, so ordinary feed dips push a few trays briefly below the weep point. Nobody scheduled those events, but the trays really did weep, and the simulator's ground truth confirms it sample by sample. They are counted as correct detections, and the investigator files them as minor so they cannot inflate recurrence counts.

### Root cause

<p align="center">
  <img src="docs/figures/root_cause_confusion.png" width="55%" alt="Root cause confusion matrix">
</p>

<table>
<tr><th>Variant (54 episodes matched to a true scenario)</th><th>Top 1 accuracy</th><th>Top 3 accuracy</th><th>Calibration error</th></tr>
<tr><td>Signals and records (production)</td><td>100.0 percent</td><td>100.0 percent</td><td>0.016</td></tr>
<tr><td>Signals only</td><td>96.3 percent</td><td>100.0 percent</td><td>0.051</td></tr>
<tr><td>Records only</td><td>96.3 percent</td><td>100.0 percent</td><td>0.116</td></tr>
<tr><td>Learned classifier on episode signatures (leave one out)</td><td>94.4 percent</td><td></td><td></td></tr>
</table>

The ablation is the interesting part. Hydraulic signals alone miss the cases where two causes look alike in the data, and records alone miss the cases where operators blamed the wrong thing or wrote nothing. Together they are right every time on this dataset, and the posterior is well calibrated (expected calibration error 0.016). The expert evidence model needs no labelled incidents, which matters on a real site where the first investigation has no training history; the learned signature classifier, which does need labels, is close behind at 94.4 percent.

### Impact

<p align="center">
  <img src="docs/figures/cost_parity.png" width="58%" alt="Estimated against true cost">
</p>

<table>
<tr><th>Measure (28 weeping scenarios)</th><th>Value</th></tr>
<tr><td>True cost from the counterfactual simulation</td><td>GBP 5,503,610</td></tr>
<tr><td>Estimated cost from plant data only</td><td>GBP 4,486,953 (18.5 percent below)</td></tr>
<tr><td>Median absolute error per event above GBP 5,000</td><td>14.9 percent</td></tr>
<tr><td>Events within 25 percent of the truth</td><td>72.7 percent</td></tr>
<tr><td>Rank correlation between estimated and true cost</td><td>0.958</td></tr>
<tr><td>Off specification product, estimated against true</td><td>103,306 t against 122,787 t</td></tr>
<tr><td>Apparent efficiency loss, mean absolute error</td><td>1.65 percentage points</td></tr>
</table>

Valve stiction, tray damage and subcooled feed are priced within one percent in aggregate. The estimate is conservative where a cause runs for longer than the weeping it produces: a fouling reboiler starts costing quality before the trays weep, and a pressure rise costs quality through lower relative volatility as well as through weeping, so the part of the event outside the detected window is not counted.

### Retrieval

<p align="center">
  <img src="docs/figures/retrieval_ablation.png" width="75%" alt="Retrieval ablation">
</p>

Each true scenario becomes a plain question ("why did C201 lose separation efficiency between these dates") and is scored against the evidence map of every human record that scenario produced. Capped recall divides by the smaller of the relevant set and k, so long incidents with hundreds of records are not punished for the size of their evidence.

<table>
<tr><th>Mode (38 investigations)</th><th>Capped recall at 10</th><th>Precision at 10</th><th>MRR</th><th>nDCG at 10</th></tr>
<tr><td>BM25</td><td>0.042</td><td>0.040</td><td>0.134</td><td>0.048</td></tr>
<tr><td>Dense (LSA)</td><td>0.087</td><td>0.087</td><td>0.204</td><td>0.093</td></tr>
<tr><td>Hybrid (reciprocal rank fusion)</td><td>0.063</td><td>0.063</td><td>0.162</td><td>0.068</td></tr>
<tr><td>Hybrid plus graph and time channel</td><td>0.178</td><td>0.171</td><td>0.603</td><td>0.255</td></tr>
<tr><td>Hybrid plus graph plus reranker</td><td>0.834</td><td>0.747</td><td>0.969</td><td>0.851</td></tr>
</table>

This is the clearest result in the project. Pure text search barely works, because the question shares almost no words with "FV 207 hunting, steam to E201 swinging by about 3.2 t/h". The graph knows which records belong to that asset and that week; the reranker knows that the same asset, the same window and the right record type matter more than word overlap. On a strict temporal split, refitting the reranker on 32 earlier investigations lifts nDCG at 10 on the 6 later ones from 0.896 to 0.927 and capped recall at 5 from 0.867 to 1.000. The right guide from the library appears in the top three for every cause (knowledge hit rate 1.000).

### Recurrence, faithfulness and linking

<table>
<tr><th>Check</th><th>Result</th></tr>
<tr><td>Recurrence linking, pairwise over matched episodes</td><td>precision 1.000, recall 1.000 (35 true pairs, 10 recurring families)</td></tr>
<tr><td>Report claims that passed the verifier</td><td>1,296 of 1,296 across 74 reports</td></tr>
<tr><td>Numbers checked against the fact registry</td><td>2,407</td></tr>
<tr><td>Clean claims wrongly rejected</td><td>0 percent</td></tr>
<tr><td>Injected wrong numbers caught</td><td>52 of 53 (98.1 percent)</td></tr>
<tr><td>Injected fake evidence identifiers caught</td><td>74 of 74</td></tr>
<tr><td>Injected out of window evidence caught</td><td>74 of 74</td></tr>
<tr><td>Injected uncited claims caught</td><td>74 of 74</td></tr>
<tr><td>Scenario shift log items linked to the right column</td><td>807 of 807, although 31 percent use a variant spelling</td></tr>
</table>

The single number change that escaped was a swap onto another value the same sentence legitimately states (the steam cost and the carbon cost happened to round to the same figure). That is the residual risk of any value matching verifier, and the reason the strict mode checks a sentence only against the facts it declares.

***

## What the investigations found

<p align="center">
  <img src="docs/figures/cost_by_cause.png" width="70%" alt="Cost by diagnosed cause">
</p>

<p align="center">
  <img src="docs/figures/recurrence_timeline.png" width="100%" alt="Recurrence timeline">
</p>

**Insight 1. The expensive causes are the slow ones.** Reboiler fouling is the costliest diagnosed cause on the site (GBP 1.41 million), and the C101 reboiler alone is the most expensive asset of the year (GBP 0.99 million). The valve output climbs for weeks before the trays weep, so by the time a dP alarm fires the money is already gone. The reboiler heat transfer index computed by the soft sensor fell five percent below its reference 50 days (C101) and 31 days (C601) before the first weeping sample, which is the window in which a clean could have been planned.

**Insight 2. The C201 steam valve is a recurring failure, not four separate events.** FV207 stuck four times in the year at intervals of about six weeks. The work order closure notes show the repair escalating from packing adjustment to positioner replacement to new trim, and a 2023 investigation in the library had already concluded that packing adjustment does not cure this valve. The four occurrences cost about GBP 0.75 million together. The recurrence agent surfaces the history and the recommendation changes from "repack" to "overhaul and add a stiction index to the loop report".

**Insight 3. Pressure moves are a hidden weeping cause.** Pressure rise is the second costliest cause (GBP 1.14 million estimated) and every pressure rise was accompanied by an unchanged temperature setpoint. The temperature controller read the hotter tray as excess boilup and cut steam, so the trays wept while the higher pressure also made the split harder, and product purity was lost. The fix is a pressure compensated temperature and an MOC rule that moves both setpoints together, not more steam.

**Insight 4. Turndown weeping is cheap per event but systematic.** Thirteen rate cuts produced weeping on every column because the steam to feed ratio was left in charge. Each event costs little; together they show that the low rate procedure in the library is not being applied, and that the open action from the 2024 C401 investigation (a minimum steam limit in the ratio block) would remove the whole family.

**Insight 5. Tray damage is the most dangerous miss.** After the C301 trip the soft sensor reported a healthy weep margin for months, because it assumes the design hole area. The tell was section dP far below expected at normal load, the trip in the alarm journal and a gamma scan request in the shift log. The investigator ties these together in the first report on the episode, citing the trip, the dP shortfall and, once it arrives, the gamma scan; in the simulated year the column ran damaged for about three months before the repair.

***

## A report, start to finish

Every investigation is written as Markdown and JSON in `outputs/reports`, and rolled into `register.html` and `register.csv`. Two complete examples are committed in `docs/sample_report_stiction.md` and `docs/sample_report_tray_damage.md`. An abridged reading of the third FV207 stiction episode, with dates written out:

> **C201 Depropaniser: reboiler steam control valve stiction (weeping), GBP 317,334**
>
> C201 showed tray weeping from 7 February 2026 15:20 to 19 February 2026 00:05 (272.8 hours), worst in the stripping (bottom) section. [SIG_DETECTOR] [SIG_DP] [SIG_EFFICIENCY]
>
> The most probable cause is reboiler steam control valve stiction (posterior 1.00). [SIG_STEAM_CYCLING] [WO410039] [SL20260209D2.1]
>
> This is occurrence 3 of this cause on FV207; earlier episodes cost GBP 417,558. [DE0011] [DE0021] [WO410007] [WO410031]
>
> Reboiler steam flow variability was 20.1 percent and the valve output moved without flow response in 14 percent of samples. [SIG_STEAM_CYCLING]
>
> Low vapour load during feed turndown was considered and rejected: median feed was 92 percent of design. [SIG_FEED]
>
> Off specification product 5,693 t (2,268 t overhead, 3,425 t bottoms) judged on the online analyzer composite, costing GBP 313,095. [SIG_IMPACT]
>
> Recommendation: overhaul or replace the FV207 trim and packing; packing adjustment alone has not held on this valve. [KB004] [KB014]

The report continues with a 34 entry evidence timeline (work orders, shift log notes such as "FIC207 in auto but flow will not settle, looks like a sticky valve", lab exceedances, two operator moves to manual, a healthy gamma scan that rules out tray damage, and the controller returned to auto after repair) and a verification block.

<p align="center">
  <img src="docs/figures/evidence_subgraph.png" width="90%" alt="Evidence subgraph">
</p>

The subgraph above is the slice of the knowledge graph behind one of these investigations: the episode, the asset and its tags, the cited shift log notes and work order, the cycling symptom that indicates the stiction hypothesis, the guides that describe it, and the earlier episode it recurs from.

***

## Running the project

### Requirements

Python 3.10 or newer. Install the package and its dependencies from the repository root:

```bash
pip install .
```

The Claude backend and the sentence transformer encoder are optional extras (`pip install .[llm]` and `pip install .[sbert]`).

### One command

```bash
python run.py all
```

This generates the dataset, runs quality checks and analytics, trains and cross fits the detector, builds the graph and index, investigates every episode, runs the evaluation suite and renders the figures. It takes about thirteen minutes on a two core machine, almost all of it detector training. For a smoke run of a few minutes on three columns and four months:

```bash
WEIR_PROFILE=quick python run.py all
```

### Stage by stage

```bash
python run.py generate            # simulator and every data source
python run.py analyse             # quality, soft sensor, detector, episodes, impact
python run.py graph               # knowledge graph
python run.py index               # hybrid retrieval index
python run.py investigate         # every episode, plus the recurrence register
python run.py investigate DE0034  # one episode
python run.py ask "Does the C201 steam valve keep failing?"
python run.py evaluate            # all metrics into outputs/metrics
python run.py figures             # figures into docs/figures
python run.py serve               # REST API on port 8077
```

After the first `evaluate`, the learned reranker weights are saved; rerun `python run.py index` to have the investigator use them.

### Asking questions

```text
Question: Which column lost the most money and why?

C101 Debutaniser lost GBP 1,927,383 across 7 episodes, mostly to reboiler fouling limiting boilup. [DE0047] [DE0010] [DE0019]
C401 C3 Splitter lost GBP 990,929 across 19 episodes, mostly to column pressure rise without temperature compensation. [DE0017] [DE0058] [DE0004]
C201 Depropaniser lost GBP 808,044 across 22 episodes, mostly to reboiler steam control valve stiction. [DE0034] [DE0011] [DE0021]
```

### REST API

<table>
<tr><th>Route</th><th>Purpose</th></tr>
<tr><td>GET /health</td><td>Service status, episode count, index and graph size, writer backend</td></tr>
<tr><td>GET /columns</td><td>Column design summaries</td></tr>
<tr><td>GET /episodes</td><td>Detected episodes, filterable by column, kind and month</td></tr>
<tr><td>GET /episodes/{id}</td><td>One episode with its impact</td></tr>
<tr><td>POST /investigate/{id}</td><td>Run the agents and return the verified report as JSON</td></tr>
<tr><td>GET /reports/{id}.md</td><td>The report as Markdown</td></tr>
<tr><td>GET /recurrence</td><td>Recurrence register</td></tr>
<tr><td>POST /ask</td><td>Free text question over the investigation history</td></tr>
<tr><td>POST /search</td><td>Hybrid GraphRAG search with a mode switch for ablation</td></tr>
</table>

```python
import httpx

base = "http://127.0.0.1:8077"
report = httpx.post(f"{base}/investigate/DE0034").json()
print(report["headline"])
hits = httpx.post(f"{base}/search", json={"query": "steam valve hunting", "column": "C201", "k": 5}).json()
```

### Using Claude as the writer

```bash
export ANTHROPIC_API_KEY=your_key
WEIR_LLM=anthropic python run.py investigate DE0034
```

The model receives the evidence index and fact registry only, never raw data. Its claims pass through the same verifier as the templates; the verification block in each report records which backend wrote it and anything that was removed. The evaluation numbers above use the offline writer so they are reproducible without a key.

***

## Repository structure

```text
weir_whisper_ai/
  run.py                        command line entry point
  config/
    plant.yaml                  site, economics and column design basis
    default.yaml                full year profile
    quick.yaml                  four month, three column smoke profile
    tiny.yaml                   profile used by the automated tests
  weir_whisper/
    simulator/                  hydraulics, scenarios, dynamic plant model, alarms,
                                text records, engineering library, dataset writer
    ingest/                     data store and data quality checks
    analytics/                  soft sensor, features, detector, episodes, impact
    nlp/                        entity linker and symptom extraction
    graph/                      knowledge graph builder and queries
    retrieval/                  corpus, BM25, dense encoder, hybrid GraphRAG retriever
    agents/                     signal analyst, records analyst, root cause, recurrence,
                                writer, verifier, orchestrator and question answering
    llm/                        offline and Anthropic backends, prompts
    evaluation/                 ground truth access, matching, the evaluation suite
    reporting/                  Markdown and HTML rendering, figures
    api/                        FastAPI application
    pipeline.py                 stage orchestration
  tests/                        unit tests and a full pipeline test
  data/sample/                  committed extracts of every data source
  docs/figures/                 README figures
  docs/sample_report_*.md       two complete investigation reports
  docs/metrics/                 the metrics, register and recurrence files behind every number above
```

Generated data, models and reports are written to `data/raw`, `data/processed` and `outputs`, which are excluded from version control; `data/sample` holds small committed extracts (fourteen days of C201 historian, forty records of each text source, two hundred lab results and the instrument index) so the formats can be inspected without running anything.

***

## Testing and quality

```bash
pytest
```

<table>
<tr><th>Test file</th><th>What it proves</th></tr>
<tr><td>test_hydraulics.py</td><td>Design margins and flood fractions are honoured exactly; weep margin rises with vapour and falls with pressure; damaged trays weep at design load</td></tr>
<tr><td>test_simulator.py</td><td>Normal operation does not weep; turndown does; the weep suppressed counterfactual holds design efficiency; stiction makes steam cycle; scenarios never overlap and respect the commissioning window</td></tr>
<tr><td>test_entities_and_retrieval.py</td><td>Every tag spelling resolves to one entity; nicknames and symptoms are found; unknown tags are kept as unresolved; BM25 ranks the right record; library chunking respects headings</td></tr>
<tr><td>test_agents.py</td><td>The posterior is a distribution and follows evidence; temperature softens it; the verifier accepts grounded claims and rejects fake citations, out of window evidence, uncited claims and wrong numbers; a fake language model's hallucinated claims are removed</td></tr>
<tr><td>test_end_to_end.py</td><td>The whole pipeline runs on the tiny profile, every report claim cites evidence that exists, metrics clear sanity floors, and every API route answers</td></tr>
</table>

All 23 tests pass in under a minute. The package is clean under pyflakes.

***

## Limitations and honest caveats

1. **The data is simulated.** The physics and control loops are real and the noise, drift and human text are deliberately messy, but a simulator's causes are cleaner than a real plant's. Perfect root cause accuracy here is a statement about the method on clean fingerprints, not a promise for a live site. Expect lower numbers on real data, with records doing more of the work.
2. **The evidence weights are engineering judgement.** They encode the symptom matrix and were written with knowledge of the mechanisms the simulator implements. The learned signature classifier is reported alongside so the gap between expert and learned reasoning is visible.
3. **Hydraulics use textbook correlations.** The weep coefficient is a smooth fit to the published chart. It is good for ranking and soft sensing, not for rating a tray.
4. **Costs are bounded by the detected window.** Causes that cost money before or beyond the weeping they produce are underestimated, which is why the aggregate estimate is conservative.
5. **Flooding and foaming are not priced against ground truth**, because their true counterfactual (the same throughput without the hydraulic limit) is not well defined. Their estimated costs are still reported.
6. **The dense channel is LSA by default** so the project runs offline. A sentence transformer can be switched in, at the cost of a model download.

***

## Extending to a real site

1. Replace the generator with connectors to the historian, alarm journal, maintenance system, LIMS and document store; the data store interface is the only seam.
2. Fill `config/plant.yaml` from the real datasheets and pick a validated fault free window for the normal behaviour models.
3. Start with the expert evidence model and the offline writer; add labelled investigations as they accumulate and refit the reranker and the signature classifier on them.
4. Keep the verifier in front of any language model. On a real site it is the part that earns trust.
