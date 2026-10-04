import os
from typing import Any, Optional
from playwright.sync_api import sync_playwright, Playwright, Browser, BrowserContext, Page

SNAPSHOT_JS = """
() => {
    // Remove previous data-agent-id attributes
    document.querySelectorAll('[data-agent-id]').forEach(el => el.removeAttribute('data-agent-id'));

    const candidates = Array.from(document.querySelectorAll(
        'a[href], button, input, select, textarea, [role="button"], [role="link"], [role="menuitem"], [role="tab"]'
    ));

    function isVisible(el) {
        if (!el) return false;
        if (el.type === 'hidden') return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
    }

    function getLabel(el) {
        const ariaLabel = el.getAttribute('aria-label');
        if (ariaLabel && ariaLabel.trim()) return ariaLabel.trim();

        const labelledBy = el.getAttribute('aria-labelledby');
        if (labelledBy) {
            const labelEl = document.getElementById(labelledBy);
            if (labelEl && labelEl.innerText.trim()) return labelEl.innerText.trim();
        }

        if (el.id) {
            try {
                const labelEl = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
                if (labelEl && labelEl.innerText.trim()) return labelEl.innerText.trim();
            } catch (e) {}
        }

        const parentLabel = el.closest('label');
        if (parentLabel && parentLabel.innerText.trim()) {
            return parentLabel.innerText.trim();
        }

        const placeholder = el.getAttribute('placeholder');
        if (placeholder && placeholder.trim()) return placeholder.trim();

        const title = el.getAttribute('title');
        if (title && title.trim()) return title.trim();

        const tag = el.tagName.toLowerCase();
        if (['button', 'a'].includes(tag) || el.getAttribute('role') === 'button' || el.getAttribute('role') === 'link') {
            const text = (el.innerText || el.textContent || '').trim();
            if (text) return text;
        }

        if (el.value && typeof el.value === 'string' && tag !== 'select') {
            return el.value.trim();
        }

        return el.getAttribute('name') || el.id || '';
    }

    const elements = [];
    let counter = 1;

    for (const el of candidates) {
        if (isVisible(el)) {
            const idStr = String(counter++);
            el.setAttribute('data-agent-id', idStr);
            const tag = el.tagName.toLowerCase();
            const role = el.getAttribute('role') || (
                tag === 'a' ? 'link' :
                tag === 'button' ? 'button' :
                tag === 'input' ? (el.type || 'input') :
                tag
            );
            const label = getLabel(el);
            let val = '';
            if (tag === 'select') {
                val = el.options[el.selectedIndex]?.text || el.value || '';
            } else if (tag === 'input' || tag === 'textarea') {
                val = el.value || '';
            }
            const disabled = Boolean(el.disabled || el.getAttribute('aria-disabled') === 'true');
            elements.push({
                id: idStr,
                tag: tag,
                role: role,
                label: label,
                value: val,
                disabled: disabled
            });
        }
    }

    const alertEls = Array.from(document.querySelectorAll('[role="alert"]'));
    const alerts = alertEls
        .filter(el => {
            const style = window.getComputedStyle(el);
            return style.display !== 'none' && style.visibility !== 'hidden';
        })
        .map(el => (el.innerText || el.textContent || '').trim())
        .filter(Boolean);

    let pageText = document.body ? (document.body.innerText || '') : '';
    if (pageText.length > 3000) {
        pageText = pageText.substring(0, 3000) + '... [truncated]';
    }

    return {
        url: window.location.href,
        title: document.title,
        text: pageText,
        alerts: alerts,
        elements: elements
    };
}
"""

class BrowserSession:
    """
    Task-agnostic Playwright browser wrapper.
    Provides snapshotting, element interaction by stable agent-ids, and safe error handling.
    """

    def __init__(self, headless: Optional[bool] = None, timeout_ms: int = 5000):
        if headless is None:
            # HEADLESS=1 enables headless, default is visible (headless=False)
            env_val = os.getenv("HEADLESS", "0").strip().lower()
            self.headless = env_val in ("1", "true", "yes")
        else:
            self.headless = headless

        self.timeout_ms = timeout_ms
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._elements_cache: dict[str, dict] = {}

    def _ensure_page(self) -> Page:
        if self.page is None or self.page.is_closed():
            if self._playwright is None:
                self._playwright = sync_playwright().start()
            if self._browser is None:
                self._browser = self._playwright.chromium.launch(headless=self.headless)
            if self._context is None:
                self._context = self._browser.new_context(viewport={"width": 1280, "height": 800})
            self.page = self._context.new_page()
            self.page.set_default_timeout(self.timeout_ms)
        return self.page

    def goto(self, url: str) -> dict[str, Any]:
        """Navigate to a URL."""
        try:
            page = self._ensure_page()
            page.goto(url, timeout=self.timeout_ms)
            try:
                page.wait_for_load_state("domcontentloaded", timeout=self.timeout_ms)
            except Exception:
                pass
            return {"ok": True, "observation": f"Navigated to {url}"}
        except Exception as e:
            return {"ok": False, "error": str(e), "observation": f"Failed to navigate to {url}: {e}"}

    def snapshot(self) -> dict[str, Any]:
        """
        Returns visible text, alerts, and numbered interactive elements.
        Tags elements with data-agent-id attributes for interaction.
        """
        try:
            page = self._ensure_page()
            data = page.evaluate(SNAPSHOT_JS)
            self._elements_cache = {el["id"]: el for el in data.get("elements", [])}

            # Build readable summary string for observation
            lines = [
                f"Page URL: {data.get('url', '')}",
                f"Page Title: {data.get('title', '')}",
            ]
            alerts = data.get("alerts", [])
            if alerts:
                lines.append(f"Alerts: {' | '.join(alerts)}")

            lines.append("Interactive Elements:")
            for el in data.get("elements", []):
                val_part = f", value='{el['value']}'" if el.get("value") else ""
                dis_part = " [disabled]" if el.get("disabled") else ""
                lines.append(f"  [{el['id']}] <{el['tag']}> role={el['role']} label='{el['label']}'{val_part}{dis_part}")

            data["ok"] = True
            data["observation"] = "\n".join(lines)
            return data
        except Exception as e:
            return {
                "ok": False,
                "error": str(e),
                "observation": f"Failed to take snapshot: {e}",
                "url": "",
                "title": "",
                "text": "",
                "alerts": [],
                "elements": [],
            }

    def describe(self, element_id: str) -> dict[str, Any]:
        """
        Returns label, tag, and role of an element without performing actions on it.
        Used by the approval policy to inspect actions safely.
        """
        try:
            page = self._ensure_page()
            locator = page.locator(f'[data-agent-id="{element_id}"]')
            if locator.count() == 0:
                return {
                    "ok": False,
                    "error": f"Element [{element_id}] not found or is stale. Please take a new snapshot.",
                    "observation": f"Element [{element_id}] not found or is stale. Please take a new snapshot.",
                }

            tag = locator.evaluate("el => el.tagName.toLowerCase()")
            role = locator.evaluate("el => el.getAttribute('role') || el.tagName.toLowerCase()")
            cached = self._elements_cache.get(str(element_id), {})
            label = cached.get("label") or locator.evaluate(
                "el => el.innerText || el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.value || ''"
            ).strip()

            return {
                "ok": True,
                "element": {"id": str(element_id), "tag": tag, "role": role, "label": label},
                "observation": f"Element [{element_id}]: tag={tag}, role={role}, label='{label}'",
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "observation": f"Failed to describe element [{element_id}]: {e}"}

    def form_values(self) -> dict[str, str]:
        """
        Read-only inspection of current form field values on the page.
        Does not mutate DOM or change snapshot element ids.
        """
        try:
            page = self._ensure_page()
            script = """
            () => {
                const results = {};
                const inputs = Array.from(document.querySelectorAll('input:not([type="hidden"]), select, textarea'));
                for (const el of inputs) {
                    let label = '';
                    if (el.id) {
                        try {
                            const lbl = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
                            if (lbl && lbl.innerText.trim()) label = lbl.innerText.trim();
                        } catch (e) {}
                    }
                    if (!label && el.placeholder) label = el.placeholder;
                    if (!label && el.name) label = el.name;
                    if (!label && el.id) label = el.id;
                    if (!label) label = 'field';

                    let val = '';
                    if (el.tagName.toLowerCase() === 'select') {
                        val = el.options[el.selectedIndex]?.text || el.value || '';
                    } else {
                        val = el.value || '';
                    }
                    results[label] = val;
                }
                return results;
            }
            """
            return page.evaluate(script) or {}
        except Exception:
            return {}

    def click(self, element_id: str) -> dict[str, Any]:
        """Click an element by its numeric snapshot id."""
        try:
            page = self._ensure_page()
            locator = page.locator(f'[data-agent-id="{element_id}"]')
            if locator.count() == 0:
                return {
                    "ok": False,
                    "error": f"Element [{element_id}] not found or is stale. Please take a new snapshot.",
                    "observation": f"Element [{element_id}] not found or is stale. Please take a new snapshot.",
                }

            cached = self._elements_cache.get(str(element_id), {})
            label = cached.get("label", element_id)

            locator.click(timeout=self.timeout_ms)
            page.wait_for_timeout(300)
            return {"ok": True, "observation": f"Clicked [{element_id}] '{label}'"}
        except Exception as e:
            return {"ok": False, "error": str(e), "observation": f"Failed to click element [{element_id}]: {e}"}

    def type(self, element_id: str, text: str) -> dict[str, Any]:
        """Type text into an input or textarea element by its snapshot id."""
        try:
            page = self._ensure_page()
            locator = page.locator(f'[data-agent-id="{element_id}"]')
            if locator.count() == 0:
                return {
                    "ok": False,
                    "error": f"Element [{element_id}] not found or is stale. Please take a new snapshot.",
                    "observation": f"Element [{element_id}] not found or is stale. Please take a new snapshot.",
                }

            cached = self._elements_cache.get(str(element_id), {})
            label = cached.get("label", element_id)

            locator.fill(text, timeout=self.timeout_ms)
            page.wait_for_timeout(200)
            return {"ok": True, "observation": f"Typed '{text}' into [{element_id}] '{label}'"}
        except Exception as e:
            return {"ok": False, "error": str(e), "observation": f"Failed to type into element [{element_id}]: {e}"}

    def select(self, element_id: str, option: str) -> dict[str, Any]:
        """Select an option by visible label or value in a select element."""
        try:
            page = self._ensure_page()
            locator = page.locator(f'[data-agent-id="{element_id}"]')
            if locator.count() == 0:
                return {
                    "ok": False,
                    "error": f"Element [{element_id}] not found or is stale. Please take a new snapshot.",
                    "observation": f"Element [{element_id}] not found or is stale. Please take a new snapshot.",
                }

            cached = self._elements_cache.get(str(element_id), {})
            label = cached.get("label", element_id)

            try:
                locator.select_option(label=option, timeout=self.timeout_ms)
            except Exception:
                locator.select_option(value=option, timeout=self.timeout_ms)

            page.wait_for_timeout(200)
            return {"ok": True, "observation": f"Selected option '{option}' on [{element_id}] '{label}'"}
        except Exception as e:
            return {"ok": False, "error": str(e), "observation": f"Failed to select option on element [{element_id}]: {e}"}

    def screenshot(self, path: str) -> dict[str, Any]:
        """Take a screenshot of the current page."""
        try:
            page = self._ensure_page()
            page.screenshot(path=path)
            return {"ok": True, "observation": f"Screenshot saved to {path}"}
        except Exception as e:
            return {"ok": False, "error": str(e), "observation": f"Failed to take screenshot: {e}"}

    def close(self) -> dict[str, Any]:
        """Close page, context, and browser."""
        try:
            if self.page and not self.page.is_closed():
                self.page.close()
            if self._context:
                self._context.close()
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
            self.page = None
            self._context = None
            self._browser = None
            self._playwright = None
            self._elements_cache.clear()
            return {"ok": True, "observation": "Browser session closed"}
        except Exception as e:
            return {"ok": False, "error": str(e), "observation": f"Failed to close browser: {e}"}
