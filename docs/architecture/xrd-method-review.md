# One workbench, separate method judgments

## Problem and usage

The workbench already retained blind traces, but hid them until phase and peak judgments were complete. Method review now opens the same queue directly onto chronology. Outcome review remains a separate mode with its scientific sign-off and reveal conditions.

The browser reads `GET /api/attempts/snapshots/{digest}/trajectory-review`, selects recorded messages/calls, then posts a `TrajectoryReviewRequest` to the same URL. A correction names its prior `submission_id`. The export reads `LabelStore.export_trajectory_reviews(use)` into a separate permission-filtered collection.

## Shape and ownership

`snapshot.method_trace` projects scrubbed historical observations from existing receipts. `trace(..., clip_text=False)` preserves full shown text while the existing outcome trace keeps its preview behavior. The method projection has no final-fit/check summaries or executor-only output. The browser derives the before-turn display from earlier turns, not presentation step groups or timestamps. Same-turn tool calls share that boundary.

`EventRef(turn_index, call_index=None)` identifies a recorded message or a call in its turn. This permits final prose without tool calls to be the consequential error. `TrajectoryReviewRequest` keeps verdict, explanation, evidence, earliest error, actual later repair and proposed advice distinct. The server validates reference membership and possible chronology; it does not infer scientific correctness.

`LabelStore` is the sole owner of method judgments, reusing snapshot access, participation permissions and exposure stamping. Content-addressed immutable records are linked by atomic, immutable successor slots per snapshot and reviewer. A competing correction cannot overwrite another; stale writes fail. An interrupted write can leave an unreferenced record, which reads and exports ignore. No mutable current pointer or file lock is needed.

Thin routes adapt transport. The existing exporter handles pseudonyms, scope, schema and manifest hashes. `trajectory_reviews.jsonl` is a separately versioned, optional addition to v1 bundles so older bundles remain readable. No method verdict is converted into scientific sign-off, judge input, preference or reward.

## Synthesis decision

Four Architect candidates compared an integrated mode, a separate reviewer page, and reuse of existing step feedback. The independent cross-judge and parent selected candidate A's integrated mode: it preserved final prose and historical information boundaries while reusing the existing owners. B supplied explicit idempotency and correction references. The cross-judge required actual repair events to remain distinct from proposed advice. C's copied evidence text was rejected because immutable receipt references already preserve the source; a second evidence copy would add synchronization. D's new page/module was rejected because it duplicated navigation without improving custody.

## Reconciliation and limits

The implementation strengthens the initial sketch with content-addressed records plus atomic successor links, preserving integrity and conflicting-write detection. Both Pass and Fail require evidence links. These changes were accepted during parent review before integration. The separate outcome API is not a secrecy boundary for future observations: this is a retrospective review protocol, not a claim that the reviewer cannot access or remember an outcome.

The current record covers one overall method judgment and its explanatory evidence. It does not impose a canonical tool sequence, create a learned process grader, or activate training. More granular per-decision judgments can be added only when scientific review shows that the existing step marks and method record are insufficient.

Verification covers chronology, final prose, same-turn evidence, full shown text, saved corrections, concurrent retry, permission/role/cohort boundaries, record hashes, export compatibility and absence of reveal effects. Browser proof uses disposable review records against copied captured traces.
