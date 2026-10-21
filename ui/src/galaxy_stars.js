// =============================================================================
// AirDrop-LLM — Deep Space Galaxy & Dynamic Cosmic Starfield Engine
// =============================================================================
// High-performance, zero-allocation 60fps canvas animation featuring:
//   - Multi-layer 3D parallax starfield with variable magnitude & color spectra
//   - Sine-wave scintillation (twinkling) with independent phase offsets
//   - Glowing star flares with radial gradients for prominent celestial bodies
//   - Random atmospheric meteor / shooting star streaks with trailing tails
//   - Mouse parallax easing with dampened momentum (subtle 3D depth)
//   - Automatic Retina / High-DPI scaling & lifecycle tab visibility throttling
// =============================================================================

"use strict";

(function () {
    const STAR_COUNT = 280;
    const DUST_COUNT = 30;
    const METEOR_MIN_INTERVAL = 3200; // ms between shooting stars
    const METEOR_MAX_INTERVAL = 7500;

    // Celestial Color Spectra (Hot O-type blue to warm M-type gold & white)
    const STAR_PALETTES = [
        { r: 255, g: 255, b: 255 }, // Pure Diamond White
        { r: 215, g: 235, b: 255 }, // Sirius Ice Blue
        { r: 190, g: 215, b: 255 }, // Rigel Deep Blue
        { r: 255, g: 240, b: 210 }, // Capella Soft Gold
        { r: 235, g: 210, b: 255 }, // Interstellar Lavender
    ];

    let canvas = null;
    let ctx = null;
    let width = 0;
    let height = 0;
    let dpr = 1;

    let stars = [];
    let dustMotes = [];
    let meteors = [];

    let mouseX = 0;
    let mouseY = 0;
    let targetMouseX = 0;
    let targetMouseY = 0;

    let animFrameId = null;
    let isVisible = true;
    let nextMeteorTime = 0;

    class Star {
        constructor() {
            this.reset(true);
        }

        reset(initial = false) {
            this.x = Math.random() * width;
            this.y = initial ? Math.random() * height : -10;
            this.layer = Math.random(); // 0 = far, 1 = near

            if (this.layer < 0.55) {
                // Distant Stars (subtle, tiny)
                this.size = 0.5 + Math.random() * 0.7;
                this.baseAlpha = 0.25 + Math.random() * 0.35;
                this.speed = 0.04 + Math.random() * 0.06;
                this.twinkleSpeed = 0.008 + Math.random() * 0.015;
            } else if (this.layer < 0.88) {
                // Mid-field Stars (moderate brightness & scintillation)
                this.size = 0.9 + Math.random() * 0.9;
                this.baseAlpha = 0.45 + Math.random() * 0.4;
                this.speed = 0.08 + Math.random() * 0.12;
                this.twinkleSpeed = 0.015 + Math.random() * 0.03;
            } else {
                // Foreground Bright Stars (with radiant flare)
                this.size = 1.6 + Math.random() * 1.3;
                this.baseAlpha = 0.75 + Math.random() * 0.25;
                this.speed = 0.14 + Math.random() * 0.18;
                this.twinkleSpeed = 0.02 + Math.random() * 0.045;
                this.hasFlare = true;
            }

            this.twinklePhase = Math.random() * Math.PI * 2;
            const colorObj = STAR_PALETTES[Math.floor(Math.random() * STAR_PALETTES.length)];
            this.r = colorObj.r;
            this.g = colorObj.g;
            this.b = colorObj.b;
        }

        update(deltaSpeed, mouseOffsetX, mouseOffsetY) {
            // Downward and gentle lateral drift
            this.y += this.speed * deltaSpeed;
            this.twinklePhase += this.twinkleSpeed;

            // Parallax shift based on depth layer
            const px = this.x + mouseOffsetX * (this.layer * 22);
            const py = this.y + mouseOffsetY * (this.layer * 22);

            // Wrap boundaries
            if (this.y > height + 10) {
                this.reset(false);
            }
            if (this.x < -20) this.x = width + 20;
            if (this.x > width + 20) this.x = -20;

            return { x: px, y: py };
        }

        draw(pos) {
            const alphaPulse = Math.sin(this.twinklePhase);
            const alpha = Math.max(0.1, Math.min(1.0, this.baseAlpha + alphaPulse * 0.3));

            ctx.fillStyle = `rgba(${this.r}, ${this.g}, ${this.b}, ${alpha.toFixed(3)})`;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, this.size, 0, Math.PI * 2);
            ctx.fill();

            // Subtle lens flare glow for prominent celestial bodies
            if (this.hasFlare && alpha > 0.6) {
                const glowRadius = this.size * 3.8;
                const glow = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, glowRadius);
                glow.addColorStop(0, `rgba(${this.r}, ${this.g}, ${this.b}, ${(alpha * 0.45).toFixed(3)})`);
                glow.addColorStop(0.5, `rgba(${this.r}, ${this.g}, ${this.b}, ${(alpha * 0.12).toFixed(3)})`);
                glow.addColorStop(1, "rgba(0, 0, 0, 0)");

                ctx.fillStyle = glow;
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, glowRadius, 0, Math.PI * 2);
                ctx.fill();
            }
        }
    }

    class CosmicDust {
        constructor() {
            this.reset(true);
        }

        reset(initial = false) {
            this.x = Math.random() * width;
            this.y = initial ? Math.random() * height : -20;
            this.size = 1.2 + Math.random() * 2.5;
            this.alpha = 0.04 + Math.random() * 0.09;
            this.vx = (Math.random() - 0.5) * 0.15;
            this.vy = 0.05 + Math.random() * 0.08;
            this.hue = 220 + Math.random() * 60; // Violet / Cyan spectrum
        }

        update() {
            this.x += this.vx;
            this.y += this.vy;

            if (this.y > height + 20 || this.x < -20 || this.x > width + 20) {
                this.reset(false);
            }
        }

        draw() {
            ctx.fillStyle = `hsla(${this.hue}, 80%, 75%, ${this.alpha})`;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    class ShootingStar {
        constructor() {
            this.active = false;
        }

        spawn() {
            this.active = true;
            this.length = 70 + Math.random() * 90;
            this.speed = 14 + Math.random() * 10;
            this.angle = (Math.PI / 4) + (Math.random() - 0.5) * 0.35; // ~45 degree trajectory

            // Start from top-left quadrant
            this.x = Math.random() * (width * 0.85);
            this.y = Math.random() * (height * 0.45);
            this.dx = Math.cos(this.angle) * this.speed;
            this.dy = Math.sin(this.angle) * this.speed;

            this.progress = 0;
            this.maxDistance = 250 + Math.random() * 200;
            this.traveled = 0;
            this.alpha = 1;
        }

        update() {
            if (!this.active) return;

            this.x += this.dx;
            this.y += this.dy;
            this.traveled += this.speed;

            const halfDist = this.maxDistance * 0.5;
            if (this.traveled < halfDist) {
                this.alpha = this.traveled / halfDist;
            } else {
                this.alpha = 1 - (this.traveled - halfDist) / halfDist;
            }

            if (this.traveled >= this.maxDistance || this.x > width + 100 || this.y > height + 100) {
                this.active = false;
            }
        }

        draw() {
            if (!this.active || this.alpha <= 0) return;

            const tailX = this.x - Math.cos(this.angle) * this.length;
            const tailY = this.y - Math.sin(this.angle) * this.length;

            const grad = ctx.createLinearGradient(tailX, tailY, this.x, this.y);
            grad.addColorStop(0, "rgba(255, 255, 255, 0)");
            grad.addColorStop(0.65, `rgba(180, 220, 255, ${(this.alpha * 0.4).toFixed(3)})`);
            grad.addColorStop(1, `rgba(255, 255, 255, ${this.alpha.toFixed(3)})`);

            ctx.strokeStyle = grad;
            ctx.lineWidth = 1.6;
            ctx.lineCap = "round";

            ctx.beginPath();
            ctx.moveTo(tailX, tailY);
            ctx.lineTo(this.x, this.y);
            ctx.stroke();

            // Head orb glow
            ctx.fillStyle = `rgba(255, 255, 255, ${this.alpha.toFixed(3)})`;
            ctx.beginPath();
            ctx.arc(this.x, this.y, 1.4, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    // -------------------------------------------------------------------------
    // Engine Initialization & Lifecycle
    // -------------------------------------------------------------------------

    function scheduleNextMeteor() {
        const delay = METEOR_MIN_INTERVAL + Math.random() * (METEOR_MAX_INTERVAL - METEOR_MIN_INTERVAL);
        nextMeteorTime = performance.now() + delay;
    }

    function initGalaxy() {
        canvas = document.getElementById("galaxyCanvas");
        if (!canvas) return;

        ctx = canvas.getContext("2d", { alpha: true });
        if (!ctx) return;

        resize();
        window.addEventListener("resize", resize, { passive: true });

        // Initialize Star Populations
        stars = [];
        for (let i = 0; i < STAR_COUNT; i++) {
            stars.push(new Star());
        }

        // Initialize Dust Particles
        dustMotes = [];
        for (let i = 0; i < DUST_COUNT; i++) {
            dustMotes.push(new CosmicDust());
        }

        // Initialize Meteor Pool
        meteors = [new ShootingStar(), new ShootingStar()];
        scheduleNextMeteor();

        // Interactive Parallax Movement Listener
        window.addEventListener("mousemove", (e) => {
            targetMouseX = (e.clientX / width - 0.5) * 2; // -1 to 1
            targetMouseY = (e.clientY / height - 0.5) * 2;
        }, { passive: true });

        // Visibility Change (battery & GPU saver)
        document.addEventListener("visibilitychange", () => {
            isVisible = !document.hidden;
            if (isVisible && !animFrameId) {
                lastTime = performance.now();
                renderLoop(performance.now());
            }
        });

        lastTime = performance.now();
        renderLoop(performance.now());
    }

    function resize() {
        if (!canvas) return;
        dpr = Math.min(window.devicePixelRatio || 1, 2);
        width = window.innerWidth;
        height = window.innerHeight;

        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;

        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
    }

    let lastTime = 0;

    function renderLoop(currentTime) {
        if (!isVisible) {
            animFrameId = null;
            return;
        }

        animFrameId = requestAnimationFrame(renderLoop);

        const delta = Math.min((currentTime - lastTime) / 16.666, 2.5); // Normalized to 60fps
        lastTime = currentTime;

        // Smooth mouse parallax damping (lerp)
        mouseX += (targetMouseX - mouseX) * 0.04;
        mouseY += (targetMouseY - mouseY) * 0.04;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // 1. Render Cosmic Dust
        for (let i = 0; i < dustMotes.length; i++) {
            dustMotes[i].update();
            dustMotes[i].draw();
        }

        // 2. Render Stars
        for (let i = 0; i < stars.length; i++) {
            const star = stars[i];
            const pos = star.update(delta, mouseX, mouseY);
            star.draw(pos);
        }

        // 3. Meteor Shower Spawning & Render
        if (currentTime >= nextMeteorTime) {
            const idleMeteor = meteors.find((m) => !m.active);
            if (idleMeteor) {
                idleMeteor.spawn();
            }
            scheduleNextMeteor();
        }

        for (let i = 0; i < meteors.length; i++) {
            meteors[i].update();
            meteors[i].draw();
        }
    }

    // Auto-launch once DOM is ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initGalaxy);
    } else {
        initGalaxy();
    }
})();
