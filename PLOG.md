# PLOG — selected possession-inference entry

The selected frozen V1 combines team-relative geometry, an external-only ball-free logistic
model, independent observed stoppage evidence and exact H/A/D Viterbi decoding. Player slots
are anonymous fragmented tracklets; features aggregate by team, not persistent identities.
Missing ball never means dead ball. Short internal gaps up to.4s may be reconstructed with
reduced ownership confidence; reconstructed coordinates never supply D observations.

Only tactical ATD metadata/raw tracking is selected. No video, broadcast alignment, Smart
Tagging, competition GT, hand correction or target learning is used. frame=ATD record ID-1;
published halves are processed independently. Other frames are X. Selected switch costs are
30/20, config SHA25662870ec2017962f73ee7e5972176461698510bb9c94599aacaeb9039293c05d3.
The selected model/source hashes are pinned in release/frozen-runtime.json. No research
challenger passed all promotion requirements; experimental code/history is not shipped.

Training sources, fit settings, modifications, licenses and selected weight hash are detailed
in models/README.md. Native Sportec targets map inactive BallStatus to D, active BallPossession
1/2 to H/A and unknown rows to X. Matches WMX/WN1/WOH trained the model; WPY/WQQ selected the
configuration; WOY/WR9 supplied a single historical final audit and are consumed. Native target
conventions and automatic tactical tracking differ; external scores are not competition scores.

Historical mean official spellF1: validation native.452612/corrupted.422813; final external
audit native.433637/corrupted.401036. Only two final matches provide weak generalization
evidence. No updated test, new accuracy claim or private competition score is asserted by
this export. The official metric/validator files are unchanged; public GT payloads are omitted.

The operator reported successful local V1 validation and byte reproduction. Hosted tools do
not inspect, attest or copy that protected output. RUNBOOK.md documents installation and the
offline selected command, official199319-row validation and independent actual-byte audit.
Known limitations include brief control changes, contested/aerial balls, uncertain stationary
scenes, long occlusions and source-domain shift. Synthetic QA establishes execution contracts,
not match accuracy. The operator performs final private revalidation/submission.
