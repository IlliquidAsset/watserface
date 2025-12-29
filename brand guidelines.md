# WatserFace Brand Guidelines

## 1. Identity
**Name:** WatserFace
**Tagline:** "Who's that again?"
**Version:** 1.0

**The Vibe:** WatserFace is the cheeky, chaotic cousin of traditional face-swapping tools. We aren't just processing pixels; we are remixing identity. The aesthetic is digital-native, slightly glitched, and high-contrast—designed to pop against modern Glassmorphism UIs.

---

## 2. Visual Language

### Color Palette
High-saturation neon tones designed to cut through dark mode and frosted glass overlays.

| Color Name | Hex | Usage |
| :--- | :--- | :--- |
| **Glitch Magenta** | `#FF00FF` | Primary Brand Color, Right-side face |
| **Deep Blurple** | `#4D4DFF` | Secondary Brand Color, Left-side face |
| **Electric Lime** | `#CCFF00` | Accents, Success States, CTAs |
| **Void Black** | `#0D0D0D` | Backgrounds, Contrast |
| **Ghost White** | `#F2F2F2` | Primary Text |

### Typography
**Display / Headings:** *Font:* **Righteous** (or similar tech-retro sans)
*Usage:* Logo text, H1, H2 headers.
*Style:* Uppercase or Title Case. Blocky, slightly futuristic.

**Body / UI:** *Font:* **JetBrains Mono** (or System Monospace)
*Usage:* General text, code snippets, logs, terminal outputs.
*Rationale:* Keeps the tool grounded in its developer roots.

---

## 3. The Logo

The logo consists of the **"Split Face" Mark** and the **Wordmark**.

* **The Mark:** A gender-neutral silhouette split down the center. The left side (Blue) is grounded; the right side (Magenta) is shifted vertically, representing the "swap" or glitch in identity.
* **The Wordmark:** Clean, bold typography. The "Watser" is standard weight, "Face" is bolded or accented.

### Official SVG Asset (Full Lockup)
Save this code as `logo_full.svg`. This vector is optimized for dark backgrounds and glassmorphism.

```xml
<svg width="800" height="200" viewBox="0 0 800 200" fill="none" xmlns="http://www.w3.org/2000/svg">
  <g id="Mark">
    <path d="M100 20C60 20 30 50 30 100C30 140 55 175 90 185V25C93 22 96 20 100 20Z" fill="#4D4DFF"/>

    <path d="M110 10C114 10 118 11 122 13V175C155 165 180 130 180 90C180 40 150 10 110 10Z" fill="#FF00FF"/>

    <rect x="95" y="40" width="20" height="5" fill="#CCFF00"/>
    <rect x="95" y="150" width="20" height="5" fill="#CCFF00"/>
  </g>

  <g id="Text" transform="translate(220, 135)">
    <text font-family="sans-serif" font-weight="800" font-size="120" fill="#F2F2F2" letter-spacing="-2">
      WATSER<tspan fill="#CCFF00">FACE</tspan>
    </text>
  </g>
</svg>
```
