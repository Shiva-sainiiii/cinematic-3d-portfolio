import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import Lenis from "lenis";

gsap.registerPlugin(ScrollTrigger);

/* ─── GLSL Vertex Shader ─────────────────────────────────────────────────── */
const VERTEX_SHADER = /* glsl */ `
  precision highp float;

  uniform float uTime;
  uniform vec2  uMouse;
  uniform float uScroll;
  uniform float uPixelRatio;

  attribute float aRandom;
  attribute vec3  aBasePosition;

  varying float vDistance;
  varying float vRandom;
  varying float vAlpha;

  /* ── Value-noise hash (no undefined vars) ── */
  float hash(vec3 p) {
    p = fract(p * vec3(443.8975, 397.2973, 491.1871));
    p += dot(p.zxy, p.yxz + 19.19);
    return fract(p.x * p.y * p.z);
  }

  float noise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
      mix(
        mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
        mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
      mix(
        mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
        mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z
    );
  }

  float fbm(vec3 p) {
    float v = 0.0;
    float a = 0.5;
    for(int i = 0; i < 5; i++) {
      v += a * noise3(p);
      p  = p * 2.07 + vec3(7.31, 3.17, 5.29);
      a *= 0.5;
    }
    return v;
  }

  /* ── Shape helpers ── */
  vec3 waveShape(vec3 base, float t) {
    float n  = fbm(base * 0.8 + vec3(uTime * 0.12));
    float n2 = fbm(base * 1.4 - vec3(uTime * 0.07));
    float y  = (n - 0.5) * 4.0 + (n2 - 0.5) * 1.5;
    return vec3(base.x, y, base.z);
  }

  vec3 vortexShape(vec3 base, float t) {
    float r     = length(base.xz);
    float angle = atan(base.z, base.x) + r * 1.2 + uTime * 0.15;
    float spread = 0.85 + aRandom * 0.3;
    float y = (base.y * 0.5) + sin(r * 2.0 - uTime * 0.4) * 0.8;
    return vec3(cos(angle) * r * spread, y, sin(angle) * r * spread);
  }

  vec3 sphereShape(vec3 base, float t) {
    float r     = 3.2 + aRandom * 0.4;
    float theta = acos(clamp(base.y / (length(base) + 0.001), -1.0, 1.0));
    float phi   = atan(base.z, base.x) + uTime * 0.05;
    float n     = fbm(vec3(phi, theta, uTime * 0.05)) * 0.5;
    float rr    = r + n;
    return vec3(
      rr * sin(theta) * cos(phi),
      rr * cos(theta),
      rr * sin(theta) * sin(phi)
    );
  }

  void main() {
    /* Phase 0→1: wave → vortex, Phase 1→2: vortex → sphere */
    float phase = uScroll * 2.0;
    float p0    = clamp(1.0 - phase,       0.0, 1.0);
    float p1    = clamp(phase,             0.0, 1.0) * clamp(2.0 - phase, 0.0, 1.0);
    float p2    = clamp(phase - 1.0,       0.0, 1.0);

    /* smooth weights */
    float w0 = smoothstep(0.0, 1.0, p0);
    float w1 = smoothstep(0.0, 1.0, p1);
    float w2 = smoothstep(0.0, 1.0, p2);

    vec3 pos = waveShape(aBasePosition, 0.0)   * w0
             + vortexShape(aBasePosition, 0.0) * w1
             + sphereShape(aBasePosition, 0.0) * w2;

    /* Mouse magnetic repulsion */
    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
    vec4 projected  = projectionMatrix * mvPosition;

    vec2 ndc     = projected.xy / projected.w;
    vec2 delta   = ndc - uMouse;
    float distM  = length(delta);
    float repel  = smoothstep(0.25, 0.0, distM) * 1.8;
    pos += vec3(normalize(vec3(delta, 0.0)) * repel * (0.5 + aRandom * 0.5));

    /* Reproject after repulsion */
    mvPosition = modelViewMatrix * vec4(pos, 1.0);

    /* Point size */
    float size = (2.5 + aRandom * 3.5) * uPixelRatio;
    gl_PointSize = size * (120.0 / -mvPosition.z);
    gl_Position  = projectionMatrix * mvPosition;

    vDistance = distM;
    vRandom   = aRandom;
    vAlpha    = 0.6 + aRandom * 0.4;
  }
`;

/* ─── GLSL Fragment Shader ───────────────────────────────────────────────── */
const FRAGMENT_SHADER = /* glsl */ `
  precision highp float;

  uniform float uTime;
  uniform float uScroll;

  varying float vDistance;
  varying float vRandom;
  varying float vAlpha;

  void main() {
    /* Soft-circle via distance from center of gl_PointCoord */
    vec2  coord = gl_PointCoord - 0.5;
    float d     = length(coord);
    if (d > 0.5) discard;

    float soft  = 1.0 - smoothstep(0.25, 0.5, d);
    float glow  = exp(-d * 6.0) * 0.7;
    float alpha = (soft + glow) * vAlpha;

    /* Color palette blended by scroll phase */
    float phase = uScroll * 2.0;

    /* Phase 0: cyan/teal cosmos */
    vec3 c0 = mix(vec3(0.12, 0.95, 1.00), vec3(0.50, 0.20, 1.00), vRandom);
    /* Phase 1: violet/magenta vortex */
    vec3 c1 = mix(vec3(0.75, 0.10, 1.00), vec3(1.00, 0.40, 0.10), vRandom);
    /* Phase 2: golden/amber sphere */
    vec3 c2 = mix(vec3(1.00, 0.60, 0.10), vec3(0.20, 0.90, 0.60), vRandom);

    float w0 = clamp(1.0 - phase,       0.0, 1.0);
    float w1 = clamp(phase,             0.0, 1.0) * clamp(2.0 - phase, 0.0, 1.0);
    float w2 = clamp(phase - 1.0,       0.0, 1.0);

    vec3 col = c0 * w0 + c1 * w1 + c2 * w2;

    /* Near-cursor brightening */
    col += vec3(0.3) * smoothstep(0.25, 0.0, vDistance);

    gl_FragColor = vec4(col * (0.85 + glow * 0.5), alpha);
  }
`;

/* ─── Film-grain + Chromatic-aberration pass ─────────────────────────────── */
const POST_SHADER = {
  uniforms: {
    tDiffuse: { value: null as THREE.Texture | null },
    uTime:    { value: 0.0 },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    precision highp float;
    uniform sampler2D tDiffuse;
    uniform float     uTime;
    varying vec2 vUv;

    float rand(vec2 co) {
      return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
    }

    void main() {
      /* Subtle chromatic aberration */
      float ca = 0.0008;
      vec4 r = texture2D(tDiffuse, vUv + vec2( ca, 0.0));
      vec4 g = texture2D(tDiffuse, vUv);
      vec4 b = texture2D(tDiffuse, vUv + vec2(-ca, 0.0));
      vec4 col = vec4(r.r, g.g, b.b, g.a);

      /* Film grain */
      float grain = rand(vUv + uTime * 0.07) * 0.04 - 0.02;
      col.rgb += grain;

      /* Vignette */
      vec2  uv2 = vUv * 2.0 - 1.0;
      float vig = 1.0 - dot(uv2 * 0.5, uv2 * 0.5);
      col.rgb *= smoothstep(0.0, 0.8, vig);

      gl_FragColor = col;
    }
  `,
};

/* ─── Cursor ──────────────────────────────────────────────────────────────── */
function CustomCursor() {
  const dotRef  = useRef<HTMLDivElement>(null);
  const ringRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const dot  = dotRef.current;
    const ring = ringRef.current;
    if (!dot || !ring) return;

    let mx = 0, my = 0, rx = 0, ry = 0;

    const onMove = (e: MouseEvent) => { mx = e.clientX; my = e.clientY; };
    window.addEventListener("mousemove", onMove);

    let raf: number;
    const tick = () => {
      rx += (mx - rx) * 0.12;
      ry += (my - ry) * 0.12;
      dot.style.transform  = `translate(${mx - 4}px, ${my - 4}px)`;
      ring.style.transform = `translate(${rx - 20}px, ${ry - 20}px)`;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      window.removeEventListener("mousemove", onMove);
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <>
      <div ref={dotRef}  className="cursor-dot"  />
      <div ref={ringRef} className="cursor-ring" />
    </>
  );
}

/* ─── Section text content ─────────────────────────────────────────────────── */
const SECTIONS = [
  {
    label:    "001 — WAVE",
    title:    ["Creative", "Digital", "Experiences"],
    sub:      "Immersive WebGL journeys that redefine the boundary between art and interface.",
    accent:   "#6efaff",
  },
  {
    label:    "002 — VORTEX",
    title:    ["Discover", "Your", "Patronus"],
    sub:      "Every interaction is a ritual. We craft digital identities that resonate at a cellular level.",
    accent:   "#a855f7",
  },
  {
    label:    "003 — SPHERE",
    title:    ["Sustainable", "Horizons"],
    sub:      "Technology with conscience. Building the ecosystems of tomorrow — today.",
    accent:   "#f97316",
  },
];

/* ─── CSS-only animated cosmos fallback (shown when WebGL is unavailable) ── */
function CosmosFallback() {
  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 0,
      background: "radial-gradient(ellipse at 50% 50%, #0d0d1a 0%, #050505 70%)",
      overflow: "hidden",
    }}>
      {/* Animated gradient orbs */}
      <div style={{
        position: "absolute", width: "60vw", height: "60vw",
        top: "10%", left: "20%",
        background: "radial-gradient(circle, rgba(110,250,255,0.08) 0%, transparent 70%)",
        animation: "orb1 8s ease-in-out infinite alternate",
        borderRadius: "50%",
      }} />
      <div style={{
        position: "absolute", width: "50vw", height: "50vw",
        top: "30%", left: "40%",
        background: "radial-gradient(circle, rgba(168,85,247,0.10) 0%, transparent 70%)",
        animation: "orb2 11s ease-in-out infinite alternate",
        borderRadius: "50%",
      }} />
      <div style={{
        position: "absolute", width: "40vw", height: "40vw",
        top: "50%", left: "10%",
        background: "radial-gradient(circle, rgba(249,115,22,0.07) 0%, transparent 70%)",
        animation: "orb3 14s ease-in-out infinite alternate",
        borderRadius: "50%",
      }} />
      {/* Dot grid */}
      <div style={{
        position: "absolute", inset: 0,
        backgroundImage: "radial-gradient(rgba(110,250,255,0.15) 1px, transparent 1px)",
        backgroundSize: "40px 40px",
        opacity: 0.3,
        animation: "gridPulse 6s ease-in-out infinite",
      }} />
      <style>{`
        @keyframes orb1 { from { transform: translate(0,0) scale(1); } to { transform: translate(5%,8%) scale(1.15); } }
        @keyframes orb2 { from { transform: translate(0,0) scale(1.1); } to { transform: translate(-8%,5%) scale(0.9); } }
        @keyframes orb3 { from { transform: translate(0,0) scale(1); } to { transform: translate(10%,-6%) scale(1.2); } }
        @keyframes gridPulse { 0%,100% { opacity: 0.15; } 50% { opacity: 0.35; } }
      `}</style>
    </div>
  );
}

/* ─── Main Experience ────────────────────────────────────────────────────── */
export default function Experience() {
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const wrapperRef   = useRef<HTMLDivElement>(null);
  const webglFailed  = useRef(false);
  const [noWebGL, setNoWebGL] = React.useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    /* ── Renderer ── */
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: false,
        alpha: false,
        powerPreference: "high-performance",
        failIfMajorPerformanceCaveat: false,
      });
    } catch (err) {
      console.warn("WebGL unavailable — CSS fallback active.", err);
      webglFailed.current = true;
      setNoWebGL(true);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;

    /* ── Scene / Camera ── */
    const scene  = new THREE.Scene();
    scene.background = new THREE.Color(0x050505);
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 0, 8);

    /* ── Particle geometry ── */
    const COUNT = 45000;
    const positions  = new Float32Array(COUNT * 3);
    const randoms    = new Float32Array(COUNT);
    const basePos    = new Float32Array(COUNT * 3);

    for (let i = 0; i < COUNT; i++) {
      /* Distribute in XZ plane initially with slight Y spread */
      const r   = Math.random() * 6.0;
      const phi = Math.random() * Math.PI * 2;
      const x   = Math.cos(phi) * r;
      const z   = Math.sin(phi) * r;
      const y   = (Math.random() - 0.5) * 1.5;

      positions[i * 3]     = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;

      basePos[i * 3]     = x;
      basePos[i * 3 + 1] = y;
      basePos[i * 3 + 2] = z;

      randoms[i] = Math.random();
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position",      new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("aBasePosition", new THREE.BufferAttribute(basePos,   3));
    geo.setAttribute("aRandom",       new THREE.BufferAttribute(randoms,   1));

    /* ── Material ── */
    const uniforms = {
      uTime:       { value: 0.0 },
      uMouse:      { value: new THREE.Vector2(0, 0) },
      uScroll:     { value: 0.0 },
      uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
    };

    const mat = new THREE.ShaderMaterial({
      vertexShader:   VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      uniforms,
      transparent:    true,
      depthWrite:     false,
      blending:       THREE.AdditiveBlending,
    });

    const points = new THREE.Points(geo, mat);
    scene.add(points);

    /* ── Post-processing ── */
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));

    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      0.85,   /* strength  */
      0.55,   /* radius    */
      0.1,    /* threshold */
    );
    composer.addPass(bloomPass);

    const postPass = new ShaderPass(POST_SHADER);
    composer.addPass(postPass);

    /* ── Lenis smooth scroll ── */
    const lenis = new Lenis({ lerp: 0.08, smoothWheel: true });

    lenis.on("scroll", () => ScrollTrigger.update());

    function lenisRaf(time: number) {
      lenis.raf(time);
      requestAnimationFrame(lenisRaf);
    }
    requestAnimationFrame(lenisRaf);

    /* Sync lenis with GSAP ticker */
    gsap.ticker.add((time) => lenis.raf(time * 1000));
    gsap.ticker.lagSmoothing(0);

    /* ── GSAP ScrollTrigger – scroll uniforms ── */
    const totalH = (SECTIONS.length - 1) * window.innerHeight;

    ScrollTrigger.create({
      trigger: "#scroll-root",
      start:   "top top",
      end:     `+=${totalH}`,
      scrub:   1.5,
      onUpdate(self) {
        uniforms.uScroll.value = self.progress;

        /* Camera drift */
        gsap.to(camera.position, {
          y: -self.progress * 2.5,
          duration: 1.2,
          overwrite: "auto",
        });
      },
    });

    /* ── Section text reveals ── */
    document.querySelectorAll<HTMLElement>(".section-title span").forEach((el) => {
      gsap.fromTo(el,
        { y: "110%", opacity: 0 },
        {
          y: "0%", opacity: 1, duration: 1.1, ease: "power3.out",
          scrollTrigger: {
            trigger: el.closest(".section"),
            start:   "top 75%",
            toggleActions: "play none none reverse",
          },
        }
      );
    });

    document.querySelectorAll<HTMLElement>(".section-sub").forEach((el) => {
      gsap.fromTo(el,
        { y: 30, opacity: 0 },
        {
          y: 0, opacity: 1, duration: 0.9, ease: "power2.out", delay: 0.3,
          scrollTrigger: {
            trigger: el.closest(".section"),
            start:   "top 70%",
            toggleActions: "play none none reverse",
          },
        }
      );
    });

    /* ── Mouse tracking (lerped) ── */
    let targetMX = 0, targetMY = 0;
    let currentMX = 0, currentMY = 0;

    const onMouseMove = (e: MouseEvent) => {
      targetMX = (e.clientX / window.innerWidth)  * 2 - 1;
      targetMY = (e.clientY / window.innerHeight) * 2 - 1;
    };
    window.addEventListener("mousemove", onMouseMove);

    /* ── Resize ── */
    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
      composer.setSize(window.innerWidth, window.innerHeight);
      uniforms.uPixelRatio.value = Math.min(window.devicePixelRatio, 2);
    };
    window.addEventListener("resize", onResize);

    /* ── Animation loop ── */
    let animId: number;
    const clock = new THREE.Clock();

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      /* Lerp mouse */
      currentMX += (targetMX - currentMX) * 0.06;
      currentMY += (targetMY - currentMY) * 0.06;

      uniforms.uTime.value        = elapsed;
      uniforms.uMouse.value.set(currentMX, -currentMY);
      postPass.uniforms.uTime.value = elapsed;

      /* Slow auto-rotation */
      points.rotation.y = elapsed * 0.025;

      composer.render();
    };
    animate();

    /* ── Cleanup ── */
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("resize", onResize);
      ScrollTrigger.getAll().forEach(t => t.kill());
      gsap.ticker.remove((time) => lenis.raf(time * 1000));
      lenis.destroy();
      geo.dispose();
      mat.dispose();
      composer.dispose();
      renderer.dispose();
    };
  }, []);

  return (
    <div ref={wrapperRef} id="scroll-root" style={{ height: `${SECTIONS.length * 100}vh`, background: "#050505" }}>

      {/* Fixed canvas — hidden when WebGL unavailable */}
      <canvas
        ref={canvasRef}
        style={{
          position: "fixed", top: 0, left: 0, width: "100vw", height: "100vh", zIndex: 0,
          display: noWebGL ? "none" : "block",
        }}
      />
      {/* CSS cosmos fallback when WebGL is unavailable */}
      {noWebGL && <CosmosFallback />}

      {/* Cursor */}
      <CustomCursor />

      {/* Navbar */}
      <nav style={{
        position: "fixed", top: 0, left: 0, right: 0, zIndex: 50,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "1.5rem 2.5rem",
        backdropFilter: "blur(12px)",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        background: "rgba(5,5,5,0.5)",
      }}>
        <span className="font-display" style={{ fontSize: "0.75rem", letterSpacing: "0.3em", color: "var(--accent)" }}>
          AXIOM.STUDIO
        </span>
        <div style={{ display: "flex", gap: "2.5rem" }}>
          {["WORK", "ABOUT", "LABS", "CONTACT"].map(item => (
            <button key={item} style={{
              background: "none", border: "none", cursor: "none",
              fontFamily: "Space Mono, monospace",
              fontSize: "0.65rem", letterSpacing: "0.2em",
              color: "rgba(240,240,240,0.5)",
              transition: "color 0.3s",
            }}
              onMouseEnter={e => (e.currentTarget.style.color = "var(--fg)")}
              onMouseLeave={e => (e.currentTarget.style.color = "rgba(240,240,240,0.5)")}
            >{item}</button>
          ))}
        </div>
      </nav>

      {/* Side stats */}
      <div style={{
        position: "fixed", left: "1.5rem", bottom: "3rem", zIndex: 50,
        display: "flex", flexDirection: "column", gap: "0.5rem",
        writingMode: "vertical-lr", textOrientation: "mixed",
        fontSize: "0.6rem", letterSpacing: "0.2em",
        color: "rgba(255,255,255,0.25)",
        fontFamily: "Space Mono, monospace",
      }}>
        45K PARTICLES&nbsp;·&nbsp;WebGL 2.0&nbsp;·&nbsp;60fps
      </div>

      {/* Scroll indicator */}
      <div style={{
        position: "fixed", right: "2rem", bottom: "3rem", zIndex: 50,
        display: "flex", flexDirection: "column", alignItems: "center", gap: "0.75rem",
      }}>
        <span style={{ fontSize: "0.55rem", letterSpacing: "0.25em", color: "rgba(255,255,255,0.3)", fontFamily: "Space Mono, monospace" }}>
          SCROLL
        </span>
        <div style={{ width: "1px", height: "48px", background: "rgba(255,255,255,0.15)", position: "relative", overflow: "hidden" }}>
          <div style={{
            position: "absolute", top: 0, left: 0, width: "100%",
            height: "40%", background: "var(--accent)",
            animation: "scrollLine 2s ease-in-out infinite",
          }} />
        </div>
      </div>

      {/* Sections */}
      {SECTIONS.map((sec, i) => (
        <section
          key={i}
          className="section"
          style={{
            position: "relative", zIndex: 10,
            height: "100vh",
            display: "flex", flexDirection: "column", justifyContent: "center",
            padding: "0 clamp(2rem, 8vw, 8rem)",
            pointerEvents: "none",
          }}
        >
          {/* Section label */}
          <div style={{
            fontSize: "0.6rem", letterSpacing: "0.35em",
            color: sec.accent, marginBottom: "1.5rem",
            fontFamily: "Space Mono, monospace",
            overflow: "hidden",
          }}>
            <span style={{ display: "inline-block" }}>{sec.label}</span>
          </div>

          {/* Main title */}
          <h1
            className="section-title"
            style={{ overflow: "hidden", lineHeight: 0.9 }}
          >
            {sec.title.map((word, wi) => (
              <div key={wi} style={{ overflow: "hidden" }}>
                <span style={{
                  display: "inline-block",
                  fontFamily: "Syncopate, sans-serif",
                  fontWeight: 700,
                  fontSize: "clamp(3rem, 9vw, 10rem)",
                  letterSpacing: "0.02em",
                  textTransform: "uppercase",
                  color: "#f0f0f0",
                  lineHeight: 1.0,
                }}>
                  {word}
                </span>
              </div>
            ))}
          </h1>

          {/* Accent line */}
          <div style={{
            width: "4rem", height: "2px",
            background: sec.accent,
            margin: "2rem 0 1.5rem",
          }} />

          {/* Subtitle */}
          <p
            className="section-sub"
            style={{
              maxWidth: "36rem",
              fontSize: "0.8rem", lineHeight: 1.8,
              letterSpacing: "0.04em",
              color: "rgba(240,240,240,0.5)",
              fontFamily: "Space Mono, monospace",
            }}
          >
            {sec.sub}
          </p>

          {/* CTA button (first section only) */}
          {i === 0 && (
            <button style={{
              marginTop: "3rem",
              width: "fit-content",
              padding: "0.85rem 2.5rem",
              border: `1px solid ${sec.accent}`,
              background: "transparent",
              color: sec.accent,
              fontFamily: "Space Mono, monospace",
              fontSize: "0.65rem", letterSpacing: "0.3em",
              cursor: "none",
              backdropFilter: "blur(8px)",
              transition: "background 0.3s, color 0.3s",
              pointerEvents: "all",
            }}
              onMouseEnter={e => { e.currentTarget.style.background = sec.accent; e.currentTarget.style.color = "#050505"; }}
              onMouseLeave={e => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = sec.accent; }}
            >
              EXPLORE WORK →
            </button>
          )}
        </section>
      ))}

      {/* Footer */}
      <footer style={{
        position: "fixed", bottom: 0, left: 0, right: 0, zIndex: 50,
        padding: "1rem 2.5rem",
        display: "flex", justifyContent: "space-between", alignItems: "center",
        borderTop: "1px solid rgba(255,255,255,0.05)",
        background: "rgba(5,5,5,0.4)",
        backdropFilter: "blur(12px)",
      }}>
        <span style={{ fontSize: "0.55rem", letterSpacing: "0.2em", color: "rgba(255,255,255,0.2)", fontFamily: "Space Mono, monospace" }}>
          © 2025 AXIOM.STUDIO
        </span>
        <span style={{ fontSize: "0.55rem", letterSpacing: "0.2em", color: "rgba(255,255,255,0.2)", fontFamily: "Space Mono, monospace" }}>
          CRAFTED WITH WEBGL
        </span>
      </footer>

      {/* Keyframe for scroll indicator animation */}
      <style>{`
        @keyframes scrollLine {
          0%   { transform: translateY(-100%); opacity: 1; }
          100% { transform: translateY(300%);  opacity: 0; }
        }

        .cursor-dot {
          position: fixed; top: 0; left: 0; z-index: 9999;
          width: 8px; height: 8px; border-radius: 50%;
          background: var(--accent);
          pointer-events: none;
          transition: background 0.3s;
          mix-blend-mode: difference;
        }

        .cursor-ring {
          position: fixed; top: 0; left: 0; z-index: 9998;
          width: 40px; height: 40px; border-radius: 50%;
          border: 1px solid rgba(110, 250, 255, 0.4);
          pointer-events: none;
          transition: border-color 0.3s;
        }
      `}</style>
    </div>
  );
}
