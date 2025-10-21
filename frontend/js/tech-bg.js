// frontend/js/tech-bg.js - 科技风全屏粒子网络 + 渐变流光背景
(function(){
  const cfg = {
    baseParticles: 100,
    maxConnectionDistance: 150,
    speed: 0.28,
    hueBase: 200, // 初始色相（蓝青）
    hueDrift: 40  // 色相漂移范围
  };

  let canvas, ctx, gradientLayer, particles = [], width = 0, height = 0, dpr = Math.max(1, window.devicePixelRatio || 1), t = 0;
  const mouse = { x: -9999, y: -9999 };

  function createCanvas(){
    // 渐变流光层
    injectStyles();
    gradientLayer = document.createElement('div');
    gradientLayer.className = 'tech-gradient-flow';
    document.body.appendChild(gradientLayer);

    // 粒子画布
    canvas = document.createElement('canvas');
    canvas.className = 'tech-bg-canvas';
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.zIndex = '1';
    canvas.style.pointerEvents = 'none';
    document.body.appendChild(canvas);
    ctx = canvas.getContext('2d');
    onResize();
  }

  function injectStyles(){
    if (document.getElementById('tech-bg-style')) return;
    const css = `
      @keyframes flowGradient {
        0% { background-position: 0% 50%; filter: hue-rotate(0deg) saturate(1.1); }
        50% { background-position: 100% 50%; filter: hue-rotate(18deg) saturate(1.25); }
        100% { background-position: 0% 50%; filter: hue-rotate(0deg) saturate(1.1); }
      }
      .tech-gradient-flow {
        position: fixed; inset: 0; z-index: 0; pointer-events: none;
        background: radial-gradient(1200px 700px at 15% 20%, rgba(23,55,94,0.85) 0%, rgba(10,30,54,0.9) 35%, rgba(6,18,36,0.95) 70%),
                    linear-gradient(120deg, #0a2a4a 0%, #0d446a 40%, #0f6b85 70%, #0a223f 100%);
        background-blend-mode: overlay, normal;
        background-size: 180% 180%, 200% 200%;
        animation: flowGradient 14s ease-in-out infinite;
      }
    `;
    const style = document.createElement('style');
    style.type = 'text/css';
    style.id = 'tech-bg-style';
    style.appendChild(document.createTextNode(css));
    document.head.appendChild(style);
  }

  function onResize(){
    width = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
    height = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    initParticles();
  }

  function rnd(min, max){ return Math.random() * (max - min) + min; }

  function initParticles(){
    const area = width * height;
    const density = cfg.baseParticles / (1920*1080); // 基于 1080p 的密度
    const count = Math.min(200, Math.max(40, Math.floor(area * density)));
    particles = [];
    for (let i = 0; i < count; i++) {
      particles.push({
        x: rnd(0, width),
        y: rnd(0, height),
        vx: rnd(-cfg.speed, cfg.speed),
        vy: rnd(-cfg.speed, cfg.speed),
        r: rnd(1.0, 2.2)
      });
    }
  }

  function step(){
    t += 0.005;
    ctx.clearRect(0, 0, width, height);

    // 粒子移动
    for (const p of particles){
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > width) p.vx *= -1;
      if (p.y < 0 || p.y > height) p.vy *= -1;
      ctx.beginPath();
      const hue = cfg.hueBase + Math.sin((p.x/width)*Math.PI*2 + t*6) * cfg.hueDrift;
      ctx.fillStyle = `hsla(${hue}, 78%, 62%, 0.95)`;
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }

    // 连接线（距离越近越亮）
    const md = cfg.maxConnectionDistance;
    for (let i = 0; i < particles.length; i++){
      for (let j = i + 1; j < particles.length; j++){
        const a = particles[i], b = particles[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const dist = Math.hypot(dx, dy);
        if (dist < md){
          const alpha = 1 - dist / md;
          const hue = cfg.hueBase + (Math.sin(t*4 + (a.x+b.x)/width*3) * cfg.hueDrift);
          ctx.strokeStyle = `hsla(${hue}, 90%, 60%, ${0.15 + alpha * 0.55})`;
          ctx.lineWidth = Math.max(0.5, 1.3 * alpha);
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    // 鼠标引力（微弱吸引）
    if (mouse.x > -999 && mouse.y > -999){
      for (const p of particles){
        const dx = mouse.x - p.x, dy = mouse.y - p.y;
        const dist = Math.hypot(dx, dy) || 1;
        if (dist < 160){
          const force = (160 - dist) / 160 * 0.02;
          p.vx += (dx / dist) * force;
          p.vy += (dy / dist) * force;
        }
      }
    }

    requestAnimationFrame(step);
  }

  function onMouseMove(e){ mouse.x = e.clientX; mouse.y = e.clientY; }
  function onMouseLeave(){ mouse.x = mouse.y = -9999; }

  function init(){
    createCanvas();
    window.addEventListener('resize', onResize);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseleave', onMouseLeave);
    requestAnimationFrame(step);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else { init(); }
})();


