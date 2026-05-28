### gliner2_extract
schema-driven information extraction with GLiNER2

use when the task needs fast extraction from provided text without a general LLM round trip.

args:
- `task`: one of `entities`, `classify`, `json`, `relations`
- `text`: source text to analyze
- `schema`: JSON array or object describing labels, classes, fields, or relation types
- `include_confidence`: optional boolean
- `include_spans`: optional boolean for tasks that support spans

defaults:
- `entities` can omit `schema` to use the configured memory entity types
- `classify` and `json` require an object schema
- `relations` accepts an array or object schema

example:
~~~json
{
  "thoughts": ["Need typed entities from the provided text."],
  "headline": "Extracting entities",
  "tool_name": "gliner2_extract",
  "tool_args": {
    "task": "entities",
    "text": "Ada Lovelace worked with Charles Babbage in London.",
    "schema": ["person", "location"],
    "include_confidence": true
  }
}
~~~
