// frontend/js/auth.js
const { createApp } = Vue;

const api = axios.create({ baseURL: "/", timeout: 60000 });

createApp({
  data() {
    return {
      authLogin: { username: "", password: "" },
      authLoading: false,
      authError: null,
      loginMode: 'face', // 默认先做人脸
      cameraActive: false,
      faceLoginStatus: null,
      stream: null,
      faceToken: null
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
    // 默认聚焦用户名输入
    this.$nextTick(() => {
      try { document.querySelector('input[placeholder="用户名"]').focus(); } catch(e){}
    });
  },
  methods: {
    fillDemo() { this.authLogin.username = "demo"; this.authLogin.password = "demo"; },
    
    async startCamera() {
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user'
          } 
        });
        this.cameraActive = true;
        this.faceLoginStatus = null;
        
        // 确保视频元素正确显示
        this.$nextTick(() => {
          if (this.$refs.videoElement && this.stream) {
            this.$refs.videoElement.srcObject = this.stream;
            this.$refs.videoElement.play().catch(e => {
              console.log('视频播放失败:', e);
            });
          }
        });
      } catch (error) {
        this.authError = "无法访问摄像头，请检查权限设置";
        console.error("摄像头访问失败:", error);
      }
    },

    // 监听登录模式切换，确保摄像头状态正确
    watch: {
      loginMode(newMode) {
        if (newMode === 'face' && this.cameraActive) {
          // 切换到人脸模式时，如果摄像头已开启，确保视频元素正确显示
          this.$nextTick(() => {
            setTimeout(() => {
              if (this.stream && this.$refs.videoElement) {
                // 重新设置视频流并确保播放
                this.$refs.videoElement.srcObject = this.stream;
                this.$refs.videoElement.load(); // 重新加载视频元素
                this.$refs.videoElement.play().catch(e => {
                  console.log('视频播放失败:', e);
                });
              }
            }, 100); // 延迟100ms确保DOM完全更新
          });
        }
      },
      // 监听人脸识别状态，确保成功时摄像头画面保持
      faceLoginStatus(newStatus) {
        if (newStatus === 'success' && this.cameraActive) {
          // 人脸识别成功时，确保摄像头画面继续显示
          this.$nextTick(() => {
            setTimeout(() => {
              if (this.stream && this.$refs.videoElement) {
                this.$refs.videoElement.srcObject = this.stream;
                this.$refs.videoElement.play().catch(e => {
                  console.log('视频播放失败:', e);
                });
              }
            }, 50);
          });
        }
      }
    },
    
    stopCamera() {
      if (this.stream) {
        this.stream.getTracks().forEach(track => track.stop());
        this.stream = null;
      }
      this.cameraActive = false;
      this.faceLoginStatus = null;
    },
    
    async faceLogin() {
      if (!this.cameraActive || !this.authLogin.username) return;
      
      this.authError = null;
      this.authLoading = true;
      this.faceLoginStatus = null;
      
      try {
        // 捕获当前帧
        const video = this.$refs.videoElement;
        const canvas = this.$refs.canvasElement;
        const ctx = canvas.getContext('2d');
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        
        // 发送人脸识别请求（第一步）
        const res = await api.post("/api/login_face", {
          username: this.authLogin.username,
          face_image: imageData
        });
        
        if (res.data && res.data.status === "success" && res.data.face_token) {
          // 人脸通过，保存临时 face_token，提示输入密码
          this.faceToken = res.data.face_token;
          this.faceLoginStatus = 'success';
          // 保持在人脸登录模式，显示成功提示和摄像头画面
          // 不切换到密码模式，让用户看到成功状态
          setTimeout(() => {
            this.loginMode = 'password';
            this.$nextTick(() => {
              try { document.querySelector('input[type="password"]').focus(); } catch(e){}
            });
          }, 2000); // 2秒后切换到密码输入
        } else {
          this.authError = res.data.message || "人脸识别失败";
          this.faceLoginStatus = 'failed';
        }
      } catch (e) {
        this.authError = (e.response && e.response.data && e.response.data.message) || e.message;
        this.faceLoginStatus = 'failed';
      } finally {
        this.authLoading = false;
      }
    },
    
    async login() {
      this.authError = null; 
      this.authLoading = true;
      try {
        // 第二步：带上 face_token 完成 2FA 登录
        const res = await api.post("/api/login", {
          username: this.authLogin.username,
          password: this.authLogin.password,
          face_token: this.faceToken
        });
        if (res.data && res.data.status === "success") {
          localStorage.setItem("token", res.data.token);
          window.location.replace("/index.html");
        } else {
          this.authError = res.data.message || "登录失败";
        }
      } catch (e) {
        this.authError = (e.response && e.response.data && e.response.data.message) || e.message;
      } finally {
        this.authLoading = false;
      }
    }
  },
  
  beforeUnmount() {
    // 组件销毁时关闭摄像头
    this.stopCamera();
  }
}).mount("#auth-root");
