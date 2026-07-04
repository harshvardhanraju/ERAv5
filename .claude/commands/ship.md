Complete the ERA v5 assignment shipping workflow. Run all four steps below in order. Do not skip any step.

## Context
- GitHub repo: https://github.com/harshvardhanraju/ERAv5
- Blog repo: https://github.com/harshvardhanraju/harshvardhanraju.github.io (live at harshvardhanraju.github.io)
- Netlify team: harshvardhan-rajug's team (already authenticated)
- Git identity: name="Harsha Raju", email="harshvardhan.rajug@gmail.com"

Determine the current session number from the folder being worked on (Session1 → s1, Session2 → s2, etc.).

---

## Step 1 · Git commit to ERAv5

```bash
cd /Users/harsha/dev/ERAv5
git config user.name "Harsha Raju"
git config user.email "harshvardhan.rajug@gmail.com"
git add SessionN/
git commit -m "<what was built: topic + key techniques>"
git push origin main
```

Confirm push succeeded and note the commit SHA.

---

## Step 2 · Deploy to Netlify

```bash
cd /Users/harsha/dev/ERAv5/SessionN
netlify deploy --prod --dir . --no-build --site-name harsha-erav5-sN
```

If the site doesn't exist yet it will be created. Note the Production URL — you need it for Step 3 and Step 4.

---

## Step 3 · Write and publish blog post

Create a new file: `/Users/harsha/dev/harshvardhanraju.github.io/posts/<session-slug>.html`

The blog is **static HTML** — no build step, no Jekyll. Copy the structure of `posts/why-neural-networks-work.html` exactly (same `<head>`, nav, footer, and CSS class names). Update the content.

**Post structure:**
1. `<header class="post-header">` — tags, h1, meta row (date + read time + GitHub link), italic lede paragraph
2. Opening prose — what problem this session addresses and why it matters
3. One `.exp-card` per experiment:
   - `.box.box-claim` for the testable claim
   - prose explaining setup and architecture
   - `.stat-row` with `.stat-pill` elements for key numbers (accuracy %, loss, etc.)
   - `.box.box-insight` for the "why this happened" explanation
4. "How It's Built" section with a code snippet and prose on the implementation choices
5. References section
6. `.demo-btn-wrap` → `.demo-btn` linking to the Netlify live demo

**Writing voice:** Human and personal. Technical specifics (real numbers, real architecture names). Explain the *why*, not just what happened. Write like you genuinely found this surprising or satisfying.

Then add a new post card to `index.html` (most recent first):
```html
<a href="posts/<slug>.html" class="post-card">
  <div class="post-card-meta">
    <span class="post-date">DD Mon YYYY</span>
    <div class="post-tags">
      <span class="tag tag-era">ERA v5 · SN</span>
      <span class="tag tag-ml">Deep Learning</span>
    </div>
  </div>
  <div class="post-card-title">Post title here</div>
  <div class="post-card-desc">One-sentence description of what the post proves.</div>
  <span class="post-card-arrow">→</span>
</a>
```

Commit and push:
```bash
cd /Users/harsha/dev/harshvardhanraju.github.io
git add posts/<slug>.html index.html
git commit -m "Add SN post: <title>"
git push origin main
```

---

## Step 4 · Draft LinkedIn post

Write a LinkedIn post with these rules:

**Must have:**
- First line: bold claim, surprising number, or counterintuitive statement. Never starts with "I".
- 3–5 specific things learned/proved, with real numbers from the experiments
- Personal and human voice — builder sharing what they discovered, not a product announcement
- Both links at the end:
  ```
  Live demo → <netlify URL>
  Blog post → <github.io URL>
  ```
- Mention ERA v5 coursework naturally in the text

**Must not have:**
- Hashtags (unless user asks)
- Corporate language ("excited to share", "thrilled to announce")
- Vague claims without numbers
- More than 350 words

**Length:** 180–350 words. Short paragraphs. One blank line between each.

Print the full post text so the user can copy-paste it directly into LinkedIn.

---

## Final checklist before reporting done

- [ ] `git log --oneline -1` on ERAv5 shows the new commit
- [ ] Netlify Production URL is live and loads
- [ ] Blog post file exists and `git log --oneline -1` on the blog repo shows the push
- [ ] LinkedIn post is printed in full

Report all four URLs: GitHub commit, Netlify live, blog post, and confirm LinkedIn draft is ready.
