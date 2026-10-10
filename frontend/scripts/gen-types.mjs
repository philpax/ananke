import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import openapiTS, { astToString } from "openapi-typescript";
import { format } from "prettier";

const schema = JSON.parse(
  execFileSync(
    "cargo",
    [
      "run",
      "--quiet",
      "--manifest-path",
      "../Cargo.toml",
      "--package",
      "ananke",
      "--no-default-features",
      "--example",
      "dump-openapi",
    ],
    { encoding: "utf8", maxBuffer: 16 * 1024 * 1024 },
  ),
);
// Backend-only features remain in the daemon schema, outside this UI's contract.
for (const path of Object.keys(schema.paths)) {
  if (path.startsWith("/api/oneshot")) delete schema.paths[path];
}
for (const name of Object.keys(schema.components.schemas)) {
  if (name.startsWith("Oneshot")) delete schema.components.schemas[name];
}
const output = astToString(await openapiTS(schema));
writeFileSync(
  "src/api/types.ts",
  await format(output, { parser: "typescript" }),
);
writeFileSync(
  "src/api/schema.json",
  await format(JSON.stringify(schema.components.schemas), { parser: "json" }),
);
