// frontend/js/register.js
const { createApp } = Vue;

const api = axios.create({ baseURL: "/", timeout: 60000 });

createApp({
  data() {
    return {
      authReg: { username: "", password: "" },
      regLoading: false,
      regError: null,
      showFaceCapture: true,
      cameraActive: false,
      capturedFace: null,
      stream: null
    };
  },
  mounted() {
    const token = localStorage.getItem("token");
    if (token) {
      api.get("/api/profile").then(r => {
        if (r.data && r.data.status === "success") {
          window.location.replace("/index.html");
        } else {
          localStorage.removeItem("token");
        }
      }).catch(() => localStorage.removeItem("token"));
    }
  },
  methods: {
    // 人脸为必选，去掉可见性切换
    
    async startCamera() {
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user'
          } 
        });
        this.$refs.videoElement.srcObject = this.stream;
        this.cameraActive = true;
      } catch (error) {
        this.regError = "无法访问摄像头，请检查权限设置";
        console.error("摄像头访问失败:", error);
      }
    },
    
    stopCamera() {
      if (this.stream) {
        this.stream.getTracks().forEach(track => track.stop());
        this.stream = null;
      }
      this.cameraActive = false;
    },
    
    captureFace() {
      if (!this.cameraActive) return;
      
      const video = this.$refs.videoElement;
      const canvas = this.$refs.canvasElement;
      const ctx = canvas.getContext('2d');
      
      // 设置canvas尺寸
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // 绘制当前帧
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // 转换为base64
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      this.capturedFace = imageData;
      
      // 停止摄像头
      this.stopCamera();
    },
    
    async register() {
      this.regError = null; 
      this.regLoading = true;
      
      try {
        const payload = {
          username: this.authReg.username,
          password: this.authReg.password
        };
        // 强制必须有人脸
        if (!this.capturedFace) {
          this.regError = "注册需要采集人脸，请先进行人脸采集";
          this.regLoading = false;
          return;
        }
        payload.face_image = this.capturedFace;
        
        const res = await api.post("/api/register", payload);
        if (res.data && res.data.status === "success") {
          const hasFace = res.data.has_face;
          alert(`注册成功！请前往登录。${hasFace ? '' : ''}`);
          window.location.replace("/auth.html");
        } else {
          this.regError = res.data.message || "注册失败";
        }
      } catch (e) {
        this.regError = (e.response && e.response.data && e.response.data.message) || e.message;
      } finally {
        this.regLoading = false;
      }
    }
  },
  
  beforeUnmount() {
    // 组件销毁时关闭摄像头
    this.stopCamera();
  }
}).mount("#register-root");



