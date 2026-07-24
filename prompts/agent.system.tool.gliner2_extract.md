### gliner2_extract
schema-driven information extraction with GLiNER2

use when the task needs typed extraction from provided text without a general
LLM round trip. do not use it for open-ended writing, summarization, or facts
that are not present in the source text.

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
- confidence threshold, maximum text size, schema size, and model length come from plugin settings

result:
- returns JSON from the configured local model or hosted API
- an error explains missing runtime, credential, device, schema, or size requirements
- API mode sends the supplied text and schema to the configured endpoint

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
