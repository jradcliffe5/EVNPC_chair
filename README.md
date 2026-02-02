# EVN PC scripts

Utility scripts to help the EVN Programme Committee chair prepare reviewer templates, collect feedback, and summarise results.

## Prerequisites
- Python 3.9 or newer.
- Poppler's `pdftotext` binary available on `PATH` (used by `proposal_to_review_template.py`).
- Proposal PDFs exported from the EVN submission system.
- Optional: a current list of PC members in `EVN_pc_members.txt` (`Name Email` per line, plus optional `E25A001#1` fixed preferences).

## Typical workflow
1. **Generate review templates**
   - Gather the proposal PDFs in a directory.
   - Run `python proposal_to_review_template.py -p path/to/proposals -m EVN_pc_members.txt -o chair_templates.txt`.
   - The script extracts experiment codes, titles, PIs, networks, and wavelengths, then produces review forms separated by `=` lines. When `--pc-members` is supplied it auto-assigns reviewers, writes the forms to `chair_templates.txt`, and stores reviewer allocations in `reviewer_assignments.txt` (override with `--assignments`). Use `--reviewers-per-proposal`, `--max-per-member`, `--max-first-per-member`, and `--max-second-per-member` to control load balancing. The generated assignment CSV now ends with a “Conflicts” section listing every proposal plus the PC members who were excluded as conflicts. Supply `--conflicts-file extra_conflicts.txt` to preload manual exclusions in the same format as that appendix (e.g. `E26A004: Alice Smith, Bob Jones`). Add `--science-tags-file science_tags.txt` to export inferred science categories (Galactic, Extragalactic, Spectral Line, Transient, AGN, Supernovae, Other) for every proposal.
   - Add `*` to the surname of the PC member who should pick up any leftover reviews (e.g. `Jack Radcliffe*`). Those chairs are preferred when per-role counts are tied. Use `--member-summary` to dump the per-reviewer HTML table (ready to paste into Outlook) to a file such as `reviewer_assignments.html`.
2. **Distribute and collect reviews**
   - Share the generated template with PC members (or the HTML form derived from it).
   - After receiving Google Form responses, download the CSV and the uploaded files. Rename the files with `python rename_reviews_from_csv.py pc_chair/EVN_PC_review_submission.csv --prefix E25A001 --source-dir "Copy of EVN.../Review submission (File responses)" --dest-dir reviews`. Use `--dry-run` to preview changes and `--skip-missing` to ignore missing files.
3. **Build the LaTeX summary**
   - Ensure the renamed review text files live in a directory such as `./reviews`.
   - Convert them into a consolidated summary with `python reviews_to_latex.py -r reviews -a reviewer_assignments.txt -o review_summary.tex`.
   - The output LaTeX file lists each proposal with reviewer initials, grades, and comments. Supply `--title` and `--version` to customise the document header.

## Additional Notes
- `template.py` contains the low-level renderer used by `proposal_to_review_template.py`; it can also be used directly by piping ampersand-delimited records from legacy sources.
- Sample member data is provided in `EVN_pc_members.txt`, including example email addresses. Update it each cycle so the automatic assignment logic remains accurate.
- Run any script with `-h`/`--help` to view the full set of options and usage examples.
- Need reminder drafts for manual sending? Run `python review_reminder.py --export-emails drafts_dir ...` to save each email as a `.txt` file ready for copy/paste.
