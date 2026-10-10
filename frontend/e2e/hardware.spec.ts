import { expect, test } from "@playwright/test";

import { mockBackend } from "./fixtures";

for (const count of [1, 2, 3, 4, 8, 16, 17])
  test(`hardware layout for ${count} GPUs at all approved boundaries`, async ({
    page,
  }) => {
    await mockBackend(page, count);
    await page.goto("/");
    for (const width of [880, 899, 900, 960, 1179, 1180, 1420, 1920]) {
      await page.setViewportSize({ width, height: 900 });
      await expect(page.locator("[data-header-mode]")).toHaveAttribute(
        "data-header-mode",
        width < 900 ? "compact" : "desktop",
      );
      if (width >= 900) {
        await expect(page.locator("[data-gpu-mode] > div")).toHaveCount(count);
        const gpu = await page.locator("[data-gpu-mode]").boundingBox();
        const cluster = await page
          .getByLabel("Hardware memory", { exact: true })
          .boundingBox();
        if (!gpu || !cluster) throw new Error("Missing hardware cluster");
        expect(Math.abs(gpu.width / cluster.width - 0.75)).toBeLessThan(0.004);
        expect(gpu.width).toBeLessThanOrEqual(840);
        expect(cluster.width).toBeLessThanOrEqual(1120);
        const scroll = await page
          .getByLabel("Hardware memory", { exact: true })
          .evaluate((el) => el.scrollWidth <= el.clientWidth);
        expect(scroll).toBe(true);
        const rows = count <= 3 ? 1 : count <= 8 ? 2 : count <= 12 ? 3 : 4;
        const expected =
          count > 16 || Math.ceil(count / rows) * 140 > gpu.width
            ? "dense"
            : count <= 3
              ? "full"
              : "compact";
        await expect(page.locator("[data-gpu-mode]")).toHaveAttribute(
          "data-gpu-mode",
          expected,
        );
        expect(
          await page
            .locator("[data-gpu-mode]")
            .evaluate((el) => getComputedStyle(el).borderRightWidth),
        ).toBe("1px");
        expect(
          await page
            .locator("[data-gpu-mode] > div")
            .first()
            .evaluate((el) => getComputedStyle(el).borderLeftWidth),
        ).toBe("0px");
      }
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
    }
  });
