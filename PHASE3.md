# Project Phase 3: Final Membership Inference Attack

CMU 18-734 / 17-731, Fall 2026

In Phase 3, evaluate your attack on the final target and explain how well it
transfers from validation. You may reuse your Phase 2 attack, trained shadow
models, and an eligible validation submission. Use the same attack method and
hyperparameters for the validation and final submissions you report.

## Schedule

All times are Eastern Standard Time (EST, UTC-5).

| Milestone | Date and time |
| --- | --- |
| Validation submissions close | Sunday, November 29 at 11:59 p.m. |
| Final submissions open | Monday, November 30 at 12:05 a.m. |
| Final predictions due on Codabench | Tuesday, December 1 at 11:59 p.m. |
| Final report and slides due on Gradescope | Thursday, December 3 at 11:59 p.m. |

## 1. Predictions on Codabench

Use the [course competition](https://www.codabench.org/competitions/18268/).
Choose one successful `val` submission made by the validation deadline and
keep its ID for your report. You may choose a Phase 2 validation submission
if it uses the attack configuration you submit for Phase 3.

Generate final predictions with the final target model and
`data/final/test.json`. In [the starter notebook](MIA_phase2.ipynb), set
`PHASE = "final"` and rerun the notebook. The submission ZIP must contain
`final/prediction.csv` at its root. Write one finite numeric score per line,
in the original order of the 2,000 test examples, without a header or index.
Higher scores indicate stronger evidence of membership.

The notebook creates and checks the archive. You can also package and
validate an existing `data/final/prediction.csv` from the repository root:

```bash
python scripts/make_submission.py --phase final
python scripts/validate_submission.py project_phase2_final_submission.zip --phase final
```

Choose one group member to upload the checked ZIP to the `final` phase.
Each group gets one final submission. Keep the final submission ID for
your report. Course staff will announce the final rankings after the
submission window closes.

## 2. Final report and slides on Gradescope

Submit **one PDF per group** to
[Project Phase 3 on Gradescope](https://www.gradescope.com/courses/1411687/assignments/9021000).
Combine your report, its code appendix, and your presentation slides in that
order. List all group members in the report and add them to the submission
in Gradescope. Assign the pages to the corresponding questions:

1. **Methodology and Analysis.** Use at most three methodology pages.
   Include an attack pipeline diagram, an explanation of the major steps,
   and the settings needed to reproduce your attack. Explain what changed
   since Phase 2 or why you reused your method. Discuss your experiments and
   the limitations of your approach.
2. **Codabench Results.** Include your team name, Codabench account(s), both
   submission IDs, and a table of the available scores. Include the
   validation score screenshot and compare the two results when available.
   If the final score has yet to be released, give the final ID and mark
   the score as pending. Staff verify both results on Codabench.
3. **Core Attack Code.** Include the code for the scoring and calibration
   used in the submitted attack in an appendix.
4. **Presentation and Q&A.** Append slides that explain the problem,
   method, key experiments, results, and limitations. Presentation times
   will be announced in class.

The results pages, code appendix, and slides are separate from the
three-page methodology limit.

## Grading

| Component | Points | Criteria |
| --- | ---: | --- |
| Methodology and analysis | 8 | Attack diagram and explanation (4), experimental analysis and limitations (2), reproduction settings (2) |
| Codabench results | 6 | Lower of the designated validation TPR and final TPR, using the formula below |
| Core attack code | 2 | Code for the attack that produced the reported results |
| Presentation and Q&A | 4 | Clarity of the method and experiments (2), explanation of design choices and limitations in discussion (2) |
| **Total** | **20** | |

The metric is **TPR at FPR strictly below 0.01**. Let
`x = min(TPR_val, TPR_final)` for the two designated submissions:

| Lower TPR `x` | Results points |
| --- | ---: |
| `x >= 0.4` | 6 |
| `0.01 <= x < 0.4` | `6 * x / 0.4` |
| `x < 0.01` | 0 |

For example, validation TPR 0.50 and final TPR 0.30 earn 4.5 out of 6
results points. Both submissions must be valid and on time for results
credit.

The top three groups by **final TPR** earn **1, 0.5, and 0.25 bonus points**,
respectively. Tied groups share the bonus points for the positions they
occupy.
