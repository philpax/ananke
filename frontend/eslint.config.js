import js from "@eslint/js";
import globals from "globals";
import hooks from "eslint-plugin-react-hooks";
import tailwind from "eslint-plugin-better-tailwindcss";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: [
      "dist/**",
      "node_modules/**",
      "test-results/**",
      "playwright-report/**",
      "src/api/types.ts",
    ],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    languageOptions: { globals: { ...globals.browser, ...globals.node } },
    plugins: { "react-hooks": hooks, "better-tailwindcss": tailwind },
    settings: { "better-tailwindcss": { entryPoint: "src/index.css" } },
    rules: {
      ...hooks.configs.recommended.rules,
      ...tailwind.configs.recommended.rules,
    },
  },
);
