# Reviewed Lexicon CSV Contract

```csv
resource_id,resource_version,category,source_form,target_form,review_status,source_reference,license_note
```

- The file is UTF-8 and every field is required.
- `review_status` must be `approved`.
- `resource_id` and `resource_version` identify the exact reviewed snapshot.
- `category` must be declared by the probability manifest.
- Duplicate category/source mappings and conflicting targets are invalid.
- `source_reference` records provenance; `license_note` records the basis for permitted use.
- The provider reads local CSV only. It does not crawl or scrape KWF.
