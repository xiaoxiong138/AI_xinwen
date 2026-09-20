const fs = require("fs");
const path = require("path");
const { pathToFileURL } = require("url");
const { chromium } = require("playwright");

function parseArgs(argv) {
  const result = { outputDir: "artifacts/email_ui_audit", files: [] };
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--output-dir") {
      result.outputDir = argv[index + 1];
      index += 1;
    } else {
      result.files.push(argv[index]);
    }
  }
  return result;
}

function safeName(value) {
  return value.replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
}

function browserExecutable() {
  const candidates = [
    process.env.PLAYWRIGHT_BROWSER_EXE,
    process.env.PROGRAMFILES && path.join(process.env.PROGRAMFILES, "Google", "Chrome", "Application", "chrome.exe"),
    process.env["PROGRAMFILES(X86)"] && path.join(process.env["PROGRAMFILES(X86)"], "Google", "Chrome", "Application", "chrome.exe"),
    process.env.PROGRAMFILES && path.join(process.env.PROGRAMFILES, "Microsoft", "Edge", "Application", "msedge.exe"),
    process.env["PROGRAMFILES(X86)"] && path.join(process.env["PROGRAMFILES(X86)"], "Microsoft", "Edge", "Application", "msedge.exe"),
  ].filter(Boolean);
  return candidates.find((candidate) => fs.existsSync(candidate));
}

async function inspectPage(page) {
  return page.evaluate(() => {
    const px = (value) => {
      const parsed = Number.parseFloat(String(value || ""));
      return Number.isFinite(parsed) ? parsed : 0;
    };
    const visible = (element) => {
      const style = window.getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
    };
    const outside = [];
    for (const element of document.body.querySelectorAll("*")) {
      if (!visible(element)) continue;
      const rect = element.getBoundingClientRect();
      if (rect.left < -1 || rect.right > window.innerWidth + 1) {
        outside.push({
          tag: element.tagName.toLowerCase(),
          className: String(element.className || "").slice(0, 100),
          left: Math.round(rect.left),
          right: Math.round(rect.right),
        });
      }
      if (outside.length >= 10) break;
    }
    const brokenImages = Array.from(document.images)
      .filter((image) => image.complete && image.naturalWidth === 0)
      .map((image) => image.src)
      .slice(0, 10);
    const bodyText = String(document.body.innerText || "").replace(/\s+/g, " ").trim();
    const itemContainers = Array.from(document.querySelectorAll("article.v10-entry, .v10-more-paper"));
    const itemLabel = (element) => {
      const title = element.querySelector(".v10-title, .v10-more-title");
      return String(title ? title.textContent : element.textContent || "").replace(/\s+/g, " ").trim().slice(0, 140);
    };
    const missingSourceNoteItems = itemContainers
      .filter((element) => !element.querySelector(".v11-source-note"))
      .map(itemLabel)
      .slice(0, 10);
    const missingClaimLabelItems = itemContainers
      .filter((element) => !element.querySelector(".v11-claim-label"))
      .map(itemLabel)
      .slice(0, 10);
    const contentElements = Array.from(
      document.querySelectorAll(
        ".v10-body, .v10-paper-plain, .v10-more-plain, .v10-paper-tech, .v10-more-tech"
      )
    ).filter(visible);
    const contentStyles = contentElements.map((element) => window.getComputedStyle(element));
    const contentFontSizes = contentStyles.map((style) => px(style.fontSize)).filter((value) => value > 0);
    const contentLineHeights = contentStyles.map((style) => px(style.lineHeight)).filter((value) => value > 0);
    const contentLineHeightRatios = contentStyles
      .map((style) => {
        const fontSize = px(style.fontSize);
        return fontSize > 0 ? px(style.lineHeight) / fontSize : 0;
      })
      .filter((value) => value > 0);
    const contentFontSizePx = contentFontSizes.length ? Math.min(...contentFontSizes) : 0;
    const contentLineHeightPx = contentLineHeights.length ? Math.min(...contentLineHeights) : 0;
    const contentLineHeightRatio = contentLineHeightRatios.length
      ? Math.min(...contentLineHeightRatios)
      : 0;
    const container = document.querySelector(".container");
    const containerRect = container ? container.getBoundingClientRect() : null;
    const content = document.querySelector(".content");
    const reportContentStyle = content ? window.getComputedStyle(content) : null;
    return {
      title: document.title,
      bodyTextChars: bodyText.length,
      bodyHeight: document.body.scrollHeight,
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
      horizontalOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      outsideCount: outside.length,
      outsideSamples: outside,
      articleCount: document.querySelectorAll("article.v10-entry").length,
      itemContainerCount: itemContainers.length,
      sourceNoteCount: document.querySelectorAll(".v11-source-note").length,
      claimLabelCount: document.querySelectorAll(".v11-claim-label").length,
      missingSourceNoteItemCount: missingSourceNoteItems.length,
      missingSourceNoteItemSamples: missingSourceNoteItems,
      missingClaimLabelItemCount: missingClaimLabelItems.length,
      missingClaimLabelItemSamples: missingClaimLabelItems,
      editorialDecisionCount: document.querySelectorAll(".v10-decision").length,
      editorialDecisionSourceCount: document.querySelectorAll(".v10-decision[data-source-key]").length,
      imageCount: document.images.length,
      brokenImageCount: brokenImages.length,
      brokenImageSamples: brokenImages,
      detailsElementCount: document.querySelectorAll("details").length,
      contentSampleClassName: contentElements.length
        ? contentElements.map((element) => String(element.className || "")).join(",")
        : "",
      contentFontSizePx,
      contentLineHeightPx,
      contentLineHeightRatio,
      containerWidthPx: containerRect ? containerRect.width : 0,
      contentPaddingLeftPx: reportContentStyle ? px(reportContentStyle.paddingLeft) : 0,
      contentPaddingRightPx: reportContentStyle ? px(reportContentStyle.paddingRight) : 0,
      bodyBackground: window.getComputedStyle(document.body).backgroundColor,
      bodyColor: window.getComputedStyle(document.body).color,
    };
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.files.length) {
    throw new Error("Usage: email_ui_audit.js [--output-dir DIR] report.html [report_part2.html ...]");
  }
  const outputDir = path.resolve(args.outputDir);
  fs.mkdirSync(outputDir, { recursive: true });
  const modes = [
    { name: "desktop", viewport: { width: 1440, height: 1000 }, colorScheme: "light", imagesDisabled: false },
    { name: "mobile", viewport: { width: 390, height: 844 }, colorScheme: "light", imagesDisabled: false },
    { name: "mobile-dark", viewport: { width: 390, height: 844 }, colorScheme: "dark", imagesDisabled: false },
    { name: "mobile-no-images", viewport: { width: 390, height: 844 }, colorScheme: "light", imagesDisabled: true },
  ];
  const executablePath = browserExecutable();
  const browser = await chromium.launch({
    headless: true,
    ...(executablePath ? { executablePath } : {}),
  });
  const rows = [];
  try {
    for (const fileValue of args.files) {
      const reportPath = path.resolve(fileValue);
      if (!fs.existsSync(reportPath)) throw new Error(`Report does not exist: ${reportPath}`);
      for (const mode of modes) {
        const context = await browser.newContext({ viewport: mode.viewport, colorScheme: mode.colorScheme });
        const page = await context.newPage();
        if (mode.imagesDisabled) {
          await page.route("**/*", async (route) => {
            if (route.request().resourceType() === "image") await route.abort();
            else await route.continue();
          });
        }
        await page.goto(pathToFileURL(reportPath).href, { waitUntil: "load" });
        if (mode.imagesDisabled) {
          await page.addStyleTag({ content: "img, picture { display: none !important; }" });
        }
        const metrics = await inspectPage(page);
        const failures = [];
        if (!metrics.title) failures.push("missing_title");
        if (metrics.bodyTextChars < 500) failures.push("empty_or_too_short_body");
        if (metrics.horizontalOverflow > 1 || metrics.outsideCount > 0) failures.push("horizontal_overflow");
        if (metrics.brokenImageCount > 0 && !mode.imagesDisabled) failures.push("broken_images");
        if (metrics.editorialDecisionCount !== metrics.editorialDecisionSourceCount) {
          failures.push("editorial_decision_source_mismatch");
        }
        if (metrics.missingSourceNoteItemCount > 0) failures.push("item_source_note_missing");
        if (metrics.missingClaimLabelItemCount > 0) failures.push("item_claim_label_missing");
        const mobileMode = mode.viewport.width <= 760;
        const minimumContentFontSize = mobileMode ? 15 : 15.5;
        if (metrics.contentFontSizePx < minimumContentFontSize) failures.push("content_font_too_small");
        if (metrics.contentLineHeightRatio < 1.6) failures.push("content_line_height_too_tight");
        if (
          mobileMode &&
          (metrics.contentPaddingLeftPx < 14 || metrics.contentPaddingRightPx < 14)
        ) {
          failures.push("mobile_content_padding_too_small");
        }
        if (!mobileMode && metrics.containerWidthPx > 900) failures.push("desktop_container_too_wide");
        const baseName = safeName(path.basename(reportPath, path.extname(reportPath)));
        const screenshotPath = path.join(outputDir, `${baseName}_${mode.name}.png`);
        await page.screenshot({ path: screenshotPath, fullPage: true });
        rows.push({
          reportPath,
          mode: mode.name,
          imagesDisabled: mode.imagesDisabled,
          screenshotPath,
          ...metrics,
          failures,
          passed: failures.length === 0,
        });
        await context.close();
      }
    }
  } finally {
    await browser.close();
  }
  const payload = {
    generatedAt: new Date().toISOString(),
    reportCount: args.files.length,
    renderCount: rows.length,
    passed: rows.every((row) => row.passed),
    failedRenderCount: rows.filter((row) => !row.passed).length,
    rows,
  };
  const metricsPath = path.join(outputDir, "metrics.json");
  fs.writeFileSync(metricsPath, JSON.stringify(payload, null, 2), "utf8");
  process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
  process.exitCode = payload.passed ? 0 : 1;
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message || error}\n`);
  process.exitCode = 1;
});
