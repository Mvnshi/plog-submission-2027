# Selected external-only checkpoint

ball_free_sportec_v1.json SHA256:
d41d788cf7ce43b16f8a6b81a8dd931849cdb44a282e9874762c25fa79c57971.
It is an inspectable24-feature numpy multinomial logistic model with training-only normalization.
Its probabilities are not calibrated. The shipped weights suffice for inference; no fitting,
private research checkout or external dataset acquisition is required to regenerate the entry.

Training: Sportec J03WMX/J03WN1/J03WOH;84950 samples (every fifth labelled frame),300 full-batch
steps, learning rate.1,L2.001,inverse-frequency class weights. No competition inputs or labels,
manual corrections, target adaptation or borrowed pretrained weights. Historical independent
training runs reproduced identical bytes; this source export does not repeat or require training.

Manuel Bassek, Robert Rein, Hendrik Weber and Daniel Memmert (2025), An integrated dataset of
spatiotemporal and event data in elite soccer, https://doi.org/10.1038/s41597-025-04505-y;
Sportec Open DFL v1, https://doi.org/10.6084/m9.figshare.28196177.v1, CC BY4.0
https://creativecommons.org/licenses/by/4.0/. Credit Sportec Solutions and DFL as data providers.
PLOG normalized native observations into team features and fitted this checkpoint; it is not
a provider model. The checkpoint is distributed under CC BY4.0 with this attribution.
Implementation source remains MIT licensed, including the upstream LICENSE notice.
