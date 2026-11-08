import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import betterTailwind from "eslint-plugin-better-tailwindcss";
import { getDefaultSelectors } from "eslint-plugin-better-tailwindcss/defaults";
import tseslint from "typescript-eslint";
import { defineConfig, globalIgnores } from "eslint/config";

// Every source file is on the design tokens, so the Tailwind rules cover
// all of `src`.
const styled = ["src/**"];

export default defineConfig([
  globalIgnores(["dist", "src/api/types.ts"]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
  },
  {
    files: styled,
    extends: [betterTailwind.configs["recommended-error"]],
    settings: {
      "better-tailwindcss": {
        entryPoint: "src/index.css",
        // The defaults already cover `cn`, `clsx`, `cva`, and `twMerge`; the
        // extra selector lints class strings held in `*Variants` and
        // `*ClassName` constants, which the defaults would skip.
        selectors: [
          ...getDefaultSelectors(),
          {
            kind: "variable",
            name: "(?:Variants|ClassName)$",
            match: [{ type: "strings" }],
          },
        ],
      },
    },
    rules: {
      // Prettier owns line length; the wrapping rule fights it.
      "better-tailwindcss/enforce-consistent-line-wrapping": "off",
      "better-tailwindcss/no-restricted-classes": [
        "error",
        {
          restrict: [
            {
              pattern: String.raw`^text-\[\d`,
              message: "Use the type scale (text-2xs to text-xl).",
            },
            {
              pattern: String.raw`\[#|^(?:bg|text|border|fill|stroke)-\[(?:rgb|hsl)`,
              message: "Use a colour token.",
            },
            {
              pattern: String.raw`^rounded(?!-full$)`,
              message: "Corners are square; only rounded-full is allowed.",
            },
            { pattern: "^shadow", message: "Surfaces are flat." },
            {
              pattern: String.raw`^(?:gap|[pm][xytrbl]?|space-[xy])-\[\d+px\]`,
              message: "Use the spacing scale.",
            },
            {
              pattern: String.raw`-(?:red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose|slate|gray|zinc|neutral|stone)-\d{2,3}(?:\/\d+)?$`,
              message: "Use a colour token, not the default palette.",
            },
          ],
        },
      ],
    },
  },
]);
