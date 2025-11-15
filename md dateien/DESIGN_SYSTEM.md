# FrameTrain - Neues Design System 🎨

## ✨ Design Overview

Das neue FrameTrain Design ist inspiriert von:
- **iOS 18 / macOS 15** - Dynamic Island, Glassmorphism
- **Cyberpunk Aesthetic** - Neon, High Contrast
- **Modern Web** - Smooth Animations, Microinteractions

---

## 🎨 Farbschema

### Primary Colors

**Purple (Primary)**
```css
from-purple-400 to-purple-600
```
- Main Brand Color
- Buttons, Links, Highlights

**Pink (Accent)**
```css
from-pink-400 to-pink-600
```
- Secondary Actions
- Gradient Overlays

**Blue (Info)**
```css
from-blue-400 to-cyan-400
```
- Info States
- Alternative Gradients

**Green (Success)**
```css
from-green-400 to-emerald-400
```
- Success States
- Positive Feedback

### Background
```css
Background: #000000 (Pure Black)
Overlays: rgba(255, 255, 255, 0.05-0.15)
```

### Text
```css
Primary: #FFFFFF
Secondary: #9CA3AF (Gray-400)
Muted: #6B7280 (Gray-500)
```

---

## 🪟 Glassmorphism

### Classes

**`.glass`** - Standard Glass
```css
background: rgba(255, 255, 255, 0.05);
backdrop-filter: blur(20px) saturate(180%);
border: 1px solid rgba(255, 255, 255, 0.1);
```

**`.glass-strong`** - Stärkerer Effekt
```css
background: rgba(255, 255, 255, 0.08);
backdrop-filter: blur(24px) saturate(200%);
border: 1px solid rgba(255, 255, 255, 0.15);
```

**`.glass-dark`** - Dunkler Glass
```css
background: rgba(0, 0, 0, 0.4);
backdrop-filter: blur(20px) saturate(180%);
border: 1px solid rgba(255, 255, 255, 0.1);
```

### Verwendung
```jsx
<div className="glass rounded-2xl p-6">
  Content here
</div>
```

---

## 🌟 Glow Effects

### Text Glow

**`.text-glow-purple`**
```css
text-shadow: 0 0 20px rgba(168, 85, 247, 0.8),
             0 0 40px rgba(168, 85, 247, 0.4);
```

**`.text-glow-blue`**
```css
text-shadow: 0 0 20px rgba(59, 130, 246, 0.8),
             0 0 40px rgba(59, 130, 246, 0.4);
```

### Box Glow

**`.glow-purple`**
```css
box-shadow: 0 0 40px rgba(168, 85, 247, 0.4),
            0 0 80px rgba(168, 85, 247, 0.2);
```

Verfügbar auch: `.glow-blue`, `.glow-pink`, `.glow-green`

### Verwendung
```jsx
<h1 className="text-glow-purple">
  Glowing Text
</h1>

<div className="glass glow-purple">
  Glowing Box
</div>
```

---

## 🎭 Dynamic Island Header

### Features
- **Floating** - Nicht am Rand anliegend
- **Glassmorphism** - Transparenter Blur-Effekt
- **Responsive** - Wird kleiner beim Scrollen
- **Smooth Transitions** - Alle Übergänge animiert

### Struktur
```jsx
<header className="fixed top-4 left-1/2 -translate-x-1/2">
  <div className="glass-strong rounded-[2rem] px-6 py-3.5">
    {/* Content */}
  </div>
</header>
```

### Scroll-Effekt
```jsx
const [scrolled, setScrolled] = useState(false)

useEffect(() => {
  const handleScroll = () => {
    setScrolled(window.scrollY > 20)
  }
  window.addEventListener('scroll', handleScroll)
}, [])
```

---

## 🎬 Animationen

### Float Animation
```jsx
<div className="animate-float">
  Floating Element
</div>
```

### Pulse Slow
```jsx
<div className="animate-pulse-slow">
  Slow Pulsing
</div>
```

### Gradient Animation
```jsx
<div className="bg-gradient-to-r from-purple-600 to-pink-600 animate-gradient">
  Animated Gradient
</div>
```

### Custom Delays
```jsx
<div 
  className="animate-float" 
  style={{ animationDelay: '1s' }}
>
  Delayed Float
</div>
```

---

## 🎨 Gradient Buttons

### Primary CTA
```jsx
<button className="group relative px-8 py-4 rounded-2xl overflow-hidden">
  <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 animate-gradient" />
  <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 opacity-0 group-hover:opacity-100 blur-xl transition-opacity" />
  <span className="relative text-white font-bold">
    Button Text
  </span>
</button>
```

### Ghost Button
```jsx
<button className="glass-strong px-8 py-4 rounded-2xl hover:bg-white/10 transition-all">
  <span className="text-gray-200">Button Text</span>
</button>
```

---

## 🃏 Card Components

### Feature Card
```jsx
<div className="glass-strong rounded-2xl p-8 hover:scale-105 transition-all group">
  <div className="w-14 h-14 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center mb-6">
    <Icon className="w-6 h-6 text-white" />
  </div>
  <h3 className="text-xl font-bold text-white mb-3">Title</h3>
  <p className="text-gray-400">Description</p>
</div>
```

### Pricing Card
```jsx
<div className="glass-strong neon-border rounded-3xl p-12 hover:scale-105 transition-transform">
  {/* Content */}
</div>
```

### Stat Card
```jsx
<div className="glass-strong rounded-2xl p-6 hover:scale-105 transition-transform">
  <div className="text-4xl font-black bg-gradient-to-r from-purple-500 to-pink-500 bg-clip-text text-transparent">
    100%
  </div>
  <div className="text-sm text-gray-400">Label</div>
</div>
```

---

## 🎯 Interactive Elements

### Hover Effects

**Scale Up**
```jsx
className="hover:scale-105 transition-transform duration-300"
```

**Glow on Hover**
```jsx
<div className="group">
  <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600 opacity-0 group-hover:opacity-100 blur-xl transition-opacity" />
</div>
```

**Translate on Hover**
```jsx
<ArrowRight className="group-hover:translate-x-1 transition-transform" />
```

---

## 🌈 Gradient Text

### Usage
```jsx
<h1 className="bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
  Gradient Text
</h1>
```

### Variations
```css
/* Purple to Pink */
from-purple-400 to-pink-400

/* Blue to Cyan */
from-blue-400 to-cyan-400

/* Pink to Purple */
from-pink-400 to-purple-400

/* Multi-color */
from-purple-400 via-pink-400 to-blue-400
```

---

## 🔲 Neon Border

### Usage
```jsx
<div className="glass-strong neon-border rounded-3xl">
  Content with animated neon border
</div>
```

Die Border ist automatisch animiert und durchläuft alle Brand-Farben.

---

## 🎪 Background Effects

### Grid Background
```jsx
<div className="grid-bg opacity-50">
  Content
</div>
```

### Radial Gradient (Mouse-Following)
```jsx
const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })

useEffect(() => {
  const handleMouseMove = (e: MouseEvent) => {
    setMousePosition({ x: e.clientX, y: e.clientY })
  }
  window.addEventListener('mousemove', handleMouseMove)
}, [])

<div 
  style={{
    background: `radial-gradient(600px at ${mousePosition.x}px ${mousePosition.y}px, rgba(168, 85, 247, 0.15), transparent 80%)`
  }}
/>
```

### Floating Orbs
```jsx
<div className="absolute top-1/4 left-10 w-72 h-72 bg-purple-600 rounded-full blur-[128px] opacity-20 animate-pulse-slow" />
```

---

## 📱 Responsive Design

### Breakpoints
```jsx
<div className="text-6xl md:text-8xl">
  Responsive Text
</div>

<div className="grid md:grid-cols-2 lg:grid-cols-3">
  Responsive Grid
</div>

<div className="hidden sm:block">
  Hidden on Mobile
</div>
```

---

## ✨ Best Practices

### Do's ✅
- Verwende Glassmorphism für Cards und Overlays
- Nutze Gradient-Buttons für CTAs
- Animiere Hover-States
- High Contrast für Lesbarkeit
- Smooth Transitions (300ms+)

### Don'ts ❌
- Nicht zu viele Glows auf einmal
- Keine zu schnellen Animationen
- Nicht zu viel Blur (max 24px)
- Keine schlechten Kontraste auf Black
- Keine statischen Buttons

---

## 🎨 Component Examples

### Hero Section
```jsx
<section className="relative pt-20 pb-32">
  <div className="text-center">
    <h1 className="text-8xl font-black text-glow-purple bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
      Hero Title
    </h1>
    <p className="text-2xl text-gray-400 mt-8">
      Subtitle here
    </p>
  </div>
  
  {/* Floating orbs */}
  <div className="absolute top-1/4 left-10 w-72 h-72 bg-purple-600 rounded-full blur-[128px] opacity-20 animate-pulse-slow" />
</section>
```

### Feature Grid
```jsx
<div className="grid md:grid-cols-3 gap-6">
  {features.map((feature) => (
    <div className="glass-strong rounded-2xl p-8 hover:scale-105 transition-all">
      <div className="w-14 h-14 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center mb-6">
        <Icon />
      </div>
      <h3 className="text-xl font-bold text-white">{feature.title}</h3>
      <p className="text-gray-400">{feature.description}</p>
    </div>
  ))}
</div>
```

---

## 🚀 Performance Tips

1. **Backdrop-filter** kann performance-intensiv sein
   - Nutze es sparsam
   - Nicht auf großen Flächen

2. **Blur-Effekte** reduzieren
   - Max 24px blur
   - Nicht zu viele gleichzeitig

3. **Animationen** optimieren
   - `transform` und `opacity` sind am schnellsten
   - Vermeide `width`, `height`, `left`, `right`

4. **Will-change** für Animationen
   ```css
   will-change: transform, opacity;
   ```

---

## 📚 Resources

- **Glassmorphism Generator:** https://ui.glass/generator
- **Gradient Generator:** https://cssgradient.io
- **Animation Easing:** https://easings.net
- **Color Picker:** https://coolors.co

---

**Viel Spaß mit dem neuen Design! 🎉**
