# 问题排查指南

## 🚨 症状：点击按钮没有反应，图表不显示

### 立即排查步骤：

#### 1. 打开浏览器开发者工具（非常重要！）
- **Chrome/Edge**: 按 `F12` 或右键点击页面选择"检查"
- **Firefox**: 按 `F12` 或右键点击页面选择"检查元素"

#### 2. 查看Console（控制台）标签页
- 点击Console标签
- 查看是否有红色错误信息
- **特别注意**这些常见错误：
  - `Uncaught ReferenceError: echarts is not defined`
  - `Uncaught SyntaxError`
  - `404 Not Found` (针对js/charts.js)
  - `500 Internal Server Error`

#### 3. 查看Network（网络）标签页
- 点击Network标签
- 点击页面上的按钮
- 查看是否有新的请求出现
- 检查请求的状态码（200成功，404未找到，500服务器错误）

---

## 🛠️ 常见问题和解决方案

### 问题1: ECharts未加载
**症状**: Console显示 `echarts is not defined`

**解决方案**:
检查 `base.html` 中ECharts是否正确加载：
```html
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
```

### 问题2: charts.js未加载或路径错误
**症状**: Network中 `js/charts.js` 返回404

**解决方案**:
确保静态文件配置正确，访问：
```
http://127.0.0.1:8000/static/js/charts.js
```
如果无法访问，说明静态文件配置有问题。

### 问题3: 按钮事件未绑定
**症状**: 点击按钮无反应，Console无错误

**解决方案**:
JavaScript代码在DOM元素加载前执行，需要添加defer或放在页面底部。

### 问题4: API路由错误
**症状**: Network中 `/api/run-diagnosis/` 返回404或500

**解决方案**:
检查Django路由配置和视图函数是否正确。

---

## 🔧 快速修复脚本

### 步骤1: 检查静态文件
```bash
# 确认静态文件存在
dir static\js\charts.js
dir static\css\style.css
```

### 步骤2: 重新收集静态文件
```bash
python manage.py collectstatic --noinput
```

### 步骤3: 重启Django服务器
```bash
# 停止当前服务器 (Ctrl+C)
# 重新启动
python manage.py runserver
```

### 步骤4: 清除浏览器缓存
- 按 `Ctrl+Shift+Delete`
- 选择清除缓存
- 重新访问页面

---

## 📋 诊断检查清单

请按顺序检查以下项目：

- [ ] 浏览器Console是否有红色错误？
- [ ] Network标签中点击按钮是否有新请求？
- [ ] 访问 `http://127.0.0.1:8000/static/js/charts.js` 是否能下载文件？
- [ ] Django服务器终端是否有错误日志？
- [ ] 是否已登录系统？
- [ ] 静态文件文件夹是否存在？

---

## 🆘 紧急修复

如果以上都不行，尝试这个临时修复：

### 1. 修改base.html，将脚本移到head标签中
### 2. 使用内联JavaScript而不是外部文件
### 3. 检查Django的DEBUG设置是否为True

---

## 📞 获取更多信息

进行以上检查后，请记录：
1. Console中的具体错误信息
2. Network中的请求状态码
3. Django终端的日志输出
4. 您使用的浏览器类型和版本

这些信息对解决问题至关重要！