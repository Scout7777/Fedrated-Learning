// frontend/js/app.js - 完整替换文件（flex 布局，不依赖 bootstrap row/col）
const { createApp } = Vue;

const api = axios.create({
  baseURL: "/",
  timeout: 60000
});
api.interceptors.request.use(cfg => {
  const t = localStorage.getItem("token");
  if (t) cfg.headers["Authorization"] = "Bearer " + t;
  return cfg;
});

async function fetchTemplate(name) {
  const url = `/components/${name}.html`;
  try {
    const r = await fetch(url);
    if (!r.ok) return `<div class="alert alert-danger p-3">组件 ${name} 加载失败 (HTTP ${r.status})</div>`;
    return await r.text();
  } catch (err) {
    return `<div class="alert alert-danger p-3">组件 ${name} 加载出错</div>`;
  }
}

(async () => {
  const names = ["dashboard","predict","train","history","about"];
  const templates = {};
  for (const n of names) templates[n] = await fetchTemplate(n);

  const rootTemplate = `
  <div class="container-fluid app-wrap">
    <header class="app-header mb-3" v-if="loggedIn">
      <div>
        <div class="app-title">同态加密的分布式协作医学影像异常检测系统</div>
        <div class="app-sub">隐私保护 · 多机构协作 · 可视化监控</div>
      </div>
      <div>
        <div v-if="loggedIn" style="text-align:right;">
          <div class="small-muted">你好，<strong>{{ currentUser }}</strong></div>
          <div style="margin-top:8px;"><button class="btn btn-outline-light btn-sm" @click="logout">登出</button></div>
        </div>
        <div v-else></div>
      </div>
    </header>

    <div class="main-grid">
      <aside class="left-col sidebar" v-if="loggedIn">
        <div class="mb-3"><strong>菜单</strong><div class="small-muted">功能入口</div></div>
        <nav>
          <ul class="nav flex-column" style="padding-left:0; list-style:none;">
            <li class="nav-item mb-1"><a href="#" class="nav-link" :class="{active: view==='about'}" @click.prevent="showView('about')">系统简介</a></li>
            <li class="nav-item mb-1"><a href="#" class="nav-link" :class="{active: view==='dashboard'}" @click.prevent="showView('dashboard')">医学监控中心</a></li>
            <li class="nav-item mb-1" v-if="loggedIn"><a href="#" class="nav-link" :class="{active: view==='train'}" @click.prevent="showView('train')">安全模型训练</a></li>
            <li class="nav-item mb-1" v-if="loggedIn"><a href="#" class="nav-link" :class="{active: view==='predict'}" @click.prevent="showView('predict')">隐私检测推理</a></li>
            <li class="nav-item mb-1" v-if="loggedIn"><a href="#" class="nav-link" :class="{active: view==='history'}" @click.prevent="showView('history')">诊断记录回顾</a></li>
          </ul>
        </nav>

        <!-- 注意：这里留空，删除原 footer 后不再需要占位文字 -->
      </aside>

      <main class="right-col">
        <section v-if="view === 'dashboard'">${templates["dashboard"]}</section>
        <section v-if="view === 'predict'">${templates["predict"]}</section>
        <section v-if="view === 'train'">${templates["train"]}</section>
        <section v-if="view === 'history'">${templates["history"]}</section>
        <section v-if="view === 'about'">${templates["about"]}</section>
      </main>
    </div>
  </div>
  `;

  const app = createApp({
    template: rootTemplate,
    data() {
      return {
        view: "about",
        loggedIn: false,
        currentUser: null,
        authLogin: { username:"", password:"" },
        authReg: { username:"", password:"" },
        authLoading: false, regLoading:false, authError:null, regError:null,
        metrics: {},
        history: [],
        lastPrediction: null,
        preview: null,
        predictResult: null,
        predictLoading: false,
      // feature extraction
      featureImage: null,
      featurePreview: null,
      featureVector: [],
      ckks: { poly: 8192, modBits: "50,30,30,30,50" },
      decryptedProbs: [],
      ckksInputCt: null,
      ckksResultCt: null,
        trainLoading: false,
        trainResult: null,
        trainForm: { epochs: 40, lr: 0.001, dp_noise_type: null, dp_epsilon: null },
        noiseLogs: [],
        userCount: 0,
        chartInstances: {}
      };
    },
    mounted() {
      const token = localStorage.getItem("token");
      if (token) {
        api.get("/api/profile").then(p => {
          if (p.data && p.data.status === "success") {
            this.loggedIn = true; this.currentUser = p.data.username; this.showView("about"); this.refreshAll();
          } else { localStorage.removeItem("token"); window.location.replace("/auth.html"); }
        }).catch(() => { localStorage.removeItem("token"); window.location.replace("/auth.html"); });
      } else { window.location.replace("/auth.html"); }
    },
    methods: {
      showView(name) {
        // dispose any existing charts to avoid duplicate instances
        for (const id of Object.keys(this.chartInstances)) {
          try { if (this.chartInstances[id] && this.chartInstances[id].dispose) this.chartInstances[id].dispose(); } catch(e){}
        }
        this.chartInstances = {};
        this.view = name;
        if (name === "dashboard") this.refreshAll();
        if (name === "history") this.fetchHistory();
        if (name === "train") this.refreshMetrics();
      },
      fillDemo() { this.authLogin.username = "demo"; this.authLogin.password = "demo"; },
      logout() { localStorage.removeItem("token"); this.loggedIn = false; this.currentUser = null; window.location.replace("/auth.html"); },

      /* data fetchers */
      async refreshAll() { await this.refreshUsersAndHistory(); await this.refreshMetrics(); },
      async refreshUsersAndHistory() {
        try { const hist = await api.get("/api/history"); if (hist.data && hist.data.status === "success") { this.history = hist.data.predictions || []; this.lastPrediction = this.history.length ? this.history[this.history.length - 1] : null; } else { this.history = []; this.lastPrediction = null; } }
        catch(e){ this.history = []; this.lastPrediction = null; }

        try { const r = await api.get("/api/users"); if (r.data && r.data.status === "success") { this.userCount = Array.isArray(r.data.users) ? r.data.users.length : 0; return; } } catch(e){}

        try { const resp = await fetch("/users.json"); if (resp.ok){ const ud = await resp.json(); this.userCount = Array.isArray(ud.users) ? ud.users.length : 0; } else this.userCount = 0; } catch(e){ this.userCount = 0; }
      },
      async fetchHistory() { try { const hist = await api.get("/api/history"); if (hist.data && hist.data.status==="success") this.history = hist.data.predictions || []; else this.history = []; } catch(e){ this.history = []; } },

      /* prediction handlers (unchanged) */
      async onFileChange(e) {
        const f = e.target.files[0]; if (!f) return;
        this.preview = URL.createObjectURL(f);
        const fd = new FormData(); fd.append("image", f);
        this.predictLoading = true; this.predictResult = null;
        try {
          const res = await api.post("/api/predict", fd, { headers: {"Content-Type":"multipart/form-data"} });
          if (res.data && res.data.status==="success") { this.predictResult = res.data; this.preview = res.data.preview_url || this.preview; this.refreshAll(); } else alert(res.data.message || "预测失败");
        } catch(err) { alert((err.response && err.response.data && err.response.data.message) || err.message || "请求失败"); }
        finally { this.predictLoading = false; }
      },
      async useDemoImage() {
        this.predictLoading = true; this.predictResult = null;
        try {
          const res = await api.post("/api/predict", { use_demo: true });
          if (res.data && res.data.status==="success") { this.predictResult = res.data; this.preview = res.data.preview_url || this.preview; this.refreshAll(); } else alert(res.data.message || "预测失败");
        } catch(err) { alert((err.response && err.response.data && err.response.data.message) || err.message || "请求失败"); }
        finally { this.predictLoading = false; }
      },

      // feature extraction (16 dims)
      onFeatFileChange(e){
        const f = e.target.files[0]; if (!f) return;
        this.featureImage = f; this.featurePreview = URL.createObjectURL(f);
      },
      async extractFeatures(){
        if (!this.featureImage){ alert('请先选择图片'); return; }
        try{
          const fd = new FormData(); fd.append('image', this.featureImage);
          const r = await api.post('/api/features', fd, { headers: { 'Content-Type': 'multipart/form-data' } });
          if (r.data && r.data.status==='success'){
            this.featureVector = r.data.features || [];
          } else {
            alert(r.data.message || '提取特征失败');
          }
        }catch(err){
          alert((err.response && err.response.data && err.response.data.message) || err.message || '请求失败');
        }
      },

      async encryptFeatures(){
        if (!this.featureVector || !this.featureVector.length){ alert('请先完成特征提取'); return; }
        try{
          const payload = {
            features: this.featureVector,
            poly_modulus_degree: this.ckks.poly,
            coeff_mod_bit_sizes: this.ckks.modBits
          };
          // 先在客户端本地根据参数创建上下文并加密，序列化 ciphertext 展示
          try {
            const poly = this.ckks.poly;
            const modBits = this.ckks.modBits;
            if (window.tenseal) {
              // 如果前端引入了 TenSEAL WebAssembly，可在此演示（当前未引入，保留占位）
            }
          } catch(e) {}
          const r = await api.post('/api/ckks_demo', payload);
          if (r.data && r.data.status==='success'){
            this.decryptedProbs = r.data.decrypted_probs || [];
            this.ckksInputCt = r.data.client_ct || null;
            this.ckksResultCt = r.data.server_ct || null;
          } else {
            alert(r.data.message || '加密演示失败');
          }
        }catch(err){
          alert((err.response && err.response.data && err.response.data.message) || err.message || '请求失败');
        }
      },

      /* training */
      async startTrain() {
        this.trainLoading = true; this.trainResult = null;
        try {
          const payload = {
            epochs: this.trainForm.epochs || 40,
            lr: this.trainForm.lr || 0.001,
            sleep: 0.2,
            dp_noise_type: this.trainForm.dp_noise_type || null,
            dp_epsilon: this.trainForm.dp_epsilon == null ? null : this.trainForm.dp_epsilon
          };
          const res = await api.post("/api/train", payload);
          if (res.data && res.data.status==="success") this.pollTrainStatus(); else alert(res.data.message || "启动训练失败");
        } catch (err) { alert((err.response && err.response.data && err.response.data.message) || err.message || "请求失败"); }
        finally { this.trainLoading = false; }
      },
      async pollTrainStatus() {
        const pollInterval = 1000;
        const timer = setInterval(async () => {
          try {
            const st = await api.get("/api/train_status");
            if (st.data && st.data.status === "success") {
              const s = st.data.train_status || {};
              this.trainResult = s;
              if (Array.isArray(s.noises)) {
                this.noiseLogs = s.noises.map(v => (v === null || v === undefined) ? null : Number(v).toFixed(6));
              }
              try { await this.refreshMetrics(); } catch(e){}
              if (s.status === "done" || s.status === "error") clearInterval(timer);
            } else clearInterval(timer);
          } catch (e) { clearInterval(timer); }
        }, pollInterval);
      },

      /* metrics & charts - simple safe render */
      async refreshMetrics() {
        try {
          const r = await api.get("/api/metrics");
          if (r.data && r.data.status==="success") this.metrics = r.data.metrics || {};
          else this.metrics = {};
        } catch(e) { this.metrics = {}; }

        this.$nextTick(() => {
          this.renderMetricsChart("chart-metrics");
          this.renderMetricsChart("chart-metrics-2");
          this.renderPie();
          this.renderConfusion();
        });
      },

      renderMetricsChart(domId) {
        const dom = document.getElementById(domId);
        if (!dom) return;
        try { if (this.chartInstances[domId] && this.chartInstances[domId].dispose) this.chartInstances[domId].dispose(); } catch(e){}
        const myChart = echarts.init(dom);
        this.chartInstances[domId] = myChart;
        const metrics = this.metrics || {};
        const option = {
          tooltip: { trigger: 'axis' },
          legend: { data: ['loss', 'accuracy'] },
          xAxis: { type: 'category', data: metrics.epochs || [] },
          yAxis: [{ type: 'value', name: 'loss' }, { type: 'value', name: 'accuracy', min: 0, max: 1 }],
          series: [
            { name: 'loss', type: 'line', data: metrics.loss || [], yAxisIndex: 0, smooth: true },
            { name: 'accuracy', type: 'line', data: metrics.accuracy || [], yAxisIndex: 1, smooth: true }
          ]
        };
        try { myChart.setOption(option); } catch(e){}
      },

      renderPie() {
        const dom = document.getElementById("chart-pie"); if (!dom) return;
        try { if (this.chartInstances["chart-pie"] && this.chartInstances["chart-pie"].dispose) this.chartInstances["chart-pie"].dispose(); } catch(e){}
        const myChart = echarts.init(dom);
        this.chartInstances["chart-pie"] = myChart;
        const dist = (this.metrics && this.metrics.class_distribution) ? this.metrics.class_distribution : [0.6, 0.4];
        const option = { tooltip: { trigger: 'item' }, legend: { bottom: 0 }, series: [{ name: 'class', type: 'pie', radius: '60%', data: [{ value: dist[0], name: '良性' }, { value: dist[1], name: '恶性' }], label: { formatter: '{b}: {d}%' } }] };
        myChart.setOption(option);
      },

      renderConfusion() {
        const dom = document.getElementById("chart-confusion"); if (!dom) return;
        try { if (this.chartInstances["chart-confusion"] && this.chartInstances["chart-confusion"].dispose) this.chartInstances["chart-confusion"].dispose(); } catch(e){}
        const myChart = echarts.init(dom);
        this.chartInstances["chart-confusion"] = myChart;
        const matrix = (this.metrics && this.metrics.confusion_matrix) ? this.metrics.confusion_matrix : [[50,5],[6,30]];
        const data = [];
        for (let i=0;i<matrix.length;i++){ for (let j=0;j<matrix[i].length;j++){ data.push([j,i,matrix[i][j]]); } }
        const option = { tooltip: {}, xAxis: { type: 'category', data: ['Pred 0','Pred 1'] }, yAxis: { type: 'category', data: ['True 0','True 1'] }, visualMap: { min: 0, max: Math.max(...matrix.flat()) }, series: [{ name: 'confusion', type: 'heatmap', data: data, label: { show: true } }] };
        myChart.setOption(option);
      },

      probText(item) { if (!item || !item.probabilities) return ""; return item.probabilities.map(p => (Math.round(p*100)/100)).join(" , "); },
      pct(item, idx) { if (!item || !item.probabilities) return 0; return Math.round((item.probabilities[idx] || 0) * 100); }
    }
  });

  app.mount("#root");
})();
