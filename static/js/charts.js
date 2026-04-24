function renderHomeTrendChart(id) {
    const chart = echarts.init(document.getElementById(id));
    const option = {
        tooltip: { trigger: 'axis' },
        xAxis: {
            type: 'category',
            data: ['13:00', '13:10', '13:20', '13:30', '13:40', '13:50', '14:00', '14:10', '14:20']
        },
        yAxis: { type: 'value' },
        series: [{
            data: [20, 35, 28, 42, 30, 55, 38, 48, 62],
            type: 'line',
            smooth: true,
            areaStyle: {}
        }]
    };
    chart.setOption(option);
}

function renderPRPDInputChart(id) {
    const chart = echarts.init(document.getElementById(id));
    const data = [];
    for (let i = 0; i < 120; i++) {
        data.push([i * 3, Math.random() * 80 + 10]);
    }
    const option = {
        tooltip: {},
        xAxis: { type: 'value', name: '相位' },
        yAxis: { type: 'value', name: '幅值' },
        series: [{
            symbolSize: 8,
            data: data,
            type: 'scatter'
        }]
    };
    chart.setOption(option);
}

function renderProbabilityChart(id) {
    const chart = echarts.init(document.getElementById(id));
    const option = {
        tooltip: { trigger: 'axis' },
        xAxis: {
            type: 'category',
            data: ['金属颗粒', '悬浮放电', '沿面放电', '气隙放电', '局部放电']
        },
        yAxis: { type: 'value', max: 100 },
        series: [{
            type: 'bar',
            data: [94, 61, 48, 32, 27],
            barWidth: '45%'
        }]
    };
    chart.setOption(option);
}

function renderGaugeChart(id, score) {
    const chart = echarts.init(document.getElementById(id));
    const option = {
        series: [{
            type: 'gauge',
            startAngle: 200,
            endAngle: -20,
            min: 0,
            max: 100,
            progress: { show: true, width: 18 },
            axisLine: { lineStyle: { width: 18 } },
            pointer: { show: true },
            detail: {
                valueAnimation: true,
                formatter: '{value}',
                fontSize: 30,
                offsetCenter: [0, '45%']
            },
            data: [{ value: score }]
        }]
    };
    chart.setOption(option);
}

function renderFusionTrendChart(id) {
    const chart = echarts.init(document.getElementById(id));
    const option = {
        tooltip: { trigger: 'axis' },
        legend: { data: ['局放', '温度', '电流振动'] },
        xAxis: {
            type: 'category',
            data: ['1', '50', '100', '150', '200', '250', '300', '350', '400', '450', '500', '550', '600']
        },
        yAxis: { type: 'value' },
        series: [
            { name: '局放', type: 'line', smooth: true, data: [15, 20, 35, 25, 28, 32, 22, 26, 38, 30, 24, 29, 31] },
            { name: '温度', type: 'line', smooth: true, data: [22, 24, 30, 29, 26, 31, 28, 30, 33, 27, 25, 31, 29] },
            { name: '电流振动', type: 'line', smooth: true, data: [30, 18, 48, 36, 21, 44, 29, 19, 50, 31, 23, 47, 35] }
        ]
    };
    chart.setOption(option);
}

function renderAnomalyTrendChart(id) {
    const chart = echarts.init(document.getElementById(id));
    const option = {
        tooltip: { trigger: 'axis' },
        xAxis: {
            type: 'category',
            data: ['0', '20', '40', '60', '80', '100', '120']
        },
        yAxis: { type: 'value' },
        series: [{
            type: 'line',
            smooth: true,
            areaStyle: {},
            data: [15, 40, 92, 38, 30, 26, 24]
        }]
    };
    chart.setOption(option);
}

function renderRULChart(id) {
    const chart = echarts.init(document.getElementById(id));
    const option = {
        tooltip: { trigger: 'axis' },
        legend: { data: ['实际值', '预测值', '置信区间上界', '置信区间下界'] },
        xAxis: {
            type: 'category',
            data: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        },
        yAxis: { type: 'value' },
        series: [
            {
                name: '实际值',
                type: 'line',
                data: [220, 214, 208, 199, 193, 188, 181, 176, 170, 164]
            },
            {
                name: '预测值',
                type: 'line',
                smooth: true,
                data: [222, 216, 209, 202, 196, 189, 184, 178, 172, 166]
            },
            {
                name: '置信区间上界',
                type: 'line',
                lineStyle: { opacity: 0 },
                stack: 'confidence',
                data: [10, 11, 10, 12, 9, 10, 11, 10, 9, 8]
            },
            {
                name: '置信区间下界',
                type: 'line',
                lineStyle: { opacity: 0 },
                areaStyle: {},
                stack: 'confidence',
                data: [18, 16, 15, 17, 14, 15, 13, 12, 11, 10]
            }
        ]
    };
    chart.setOption(option);
}

function renderHistoryTrendChart(id) {
    const chart = echarts.init(document.getElementById(id));
    const option = {
        tooltip: { trigger: 'axis' },
        legend: { data: ['HI指数', '局放幅值', '温度', 'RUL预测值'] },
        xAxis: {
            type: 'category',
            data: ['05-01', '05-05', '05-10', '05-15', '05-20', '05-25', '06-01']
        },
        yAxis: { type: 'value' },
        series: [
            { name: 'HI指数', type: 'line', smooth: true, data: [90, 91, 92, 90, 89, 88, 92] },
            { name: '局放幅值', type: 'line', smooth: true, data: [22, 28, 30, 35, 32, 40, 38] },
            { name: '温度', type: 'line', smooth: true, data: [48, 50, 49, 52, 54, 53, 51] },
            { name: 'RUL预测值', type: 'line', smooth: true, data: [210, 205, 201, 196, 192, 188, 186] }
        ]
    };
    chart.setOption(option);
}

// ==================== 模型调用与数据更新 ====================

// 显示加载提示
function showLoading(message) {
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'loadingOverlay';
    loadingDiv.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); display: flex; justify-content: center; align-items: center; z-index: 9999;';
    loadingDiv.innerHTML = `<div style="background: white; padding: 30px; border-radius: 10px; text-align: center;"><p style="font-size: 16px; color: #333;">${message}</p></div>`;
    document.body.appendChild(loadingDiv);
}

// 隐藏加载提示
function hideLoading() {
    const loadingDiv = document.getElementById('loadingOverlay');
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

function fetchApiJson(url) {
    return fetch(url, {
        headers: {
            'X-Requested-With': 'XMLHttpRequest',
            'Content-Type': 'application/json',
        }
    }).then(async (response) => {
        const contentType = response.headers.get('content-type') || '';
        const isJson = contentType.includes('application/json');
        const payload = isJson ? await response.json() : await response.text();

        if (!response.ok) {
            const message = isJson && payload.error
                ? payload.error
                : `请求失败（HTTP ${response.status}）`;
            throw new Error(message);
        }

        if (!isJson) {
            throw new Error('服务返回了非 JSON 数据，请重新登录后重试。');
        }

        return payload;
    });
}

// 智能诊断 - 运行模型按钮
document.addEventListener('DOMContentLoaded', function() {
    const runDiagnosisBtn = document.getElementById('runDiagnosisBtn');
    if (runDiagnosisBtn) {
        runDiagnosisBtn.addEventListener('click', function() {
            runDiagnosisBtn.disabled = true;
            showLoading('正在运行诊断模型，请稍候...');

            fetchApiJson('/api/run-diagnosis/')
                .then(data => {
                    hideLoading();

                    if (data.error) {
                        alert('模型运行出错: ' + data.error);
                        return;
                    }

                    // 更新诊断结果文本
                    const resultElements = document.querySelectorAll('.result-line strong');
                    if (resultElements.length >= 3) {
                        resultElements[0].textContent = data.discharge_type;
                        resultElements[1].textContent = data.confidence + '%';
                        resultElements[2].textContent = data.severity;
                    }

                    // 更新PRPD图谱
                    const prpdChart = echarts.getInstanceByDom(document.getElementById('prpdInputChart'));
                    if (prpdChart) {
                        prpdChart.setOption({
                            series: [{
                                data: data.prpd_data
                            }]
                        });
                    }

                    // 更新概率分布柱状图
                    const probChart = echarts.getInstanceByDom(document.getElementById('probabilityChart'));
                    if (probChart) {
                        probChart.setOption({
                            xAxis: {
                                data: data.probabilities.types
                            },
                            series: [{
                                data: data.probabilities.values
                            }]
                        });
                    }

                    // 更新特征重要性
                    const featureContainer = document.querySelector('.feature-bars');
                    if (featureContainer && data.feature_importance) {
                        featureContainer.innerHTML = data.feature_importance.map(item => `
                            <div class="feature-item">
                                <div class="feature-label">${item.name}</div>
                                <div class="feature-progress">
                                    <div class="feature-fill" style="width: ${item.value}%;"></div>
                                </div>
                                <div class="feature-value">${item.value}%</div>
                            </div>
                        `).join('');
                    }

                    // 更新相似案例
                    const caseList = document.querySelector('.case-list');
                    if (caseList && data.similar_cases) {
                        caseList.innerHTML = data.similar_cases.map(c =>
                            `<div class="case-item">${c}</div>`
                        ).join('');
                    }

                    alert('模型运行完成！诊断结果已更新。');
                })
                .catch(error => {
                    hideLoading();
                    alert('请求失败: ' + error.message);
                })
                .finally(() => {
                    runDiagnosisBtn.disabled = false;
                });
        });
    }

    // 健康评估 - 运行模型按钮
    const runHealthBtn = document.getElementById('runHealthBtn');
    if (runHealthBtn) {
        runHealthBtn.addEventListener('click', function() {
            runHealthBtn.disabled = true;
            showLoading('正在运行健康评估模型，请稍候...');

            fetchApiJson('/api/run-health/')
                .then(data => {
                    hideLoading();

                    if (data.error) {
                        alert('模型运行出错: ' + data.error);
                        return;
                    }

                    // 更新HI仪表盘
                    const gaugeChart = echarts.getInstanceByDom(document.getElementById('gaugeChart'));
                    if (gaugeChart) {
                        gaugeChart.setOption({
                            series: [{
                                data: [{ value: data.hi_score }]
                            }]
                        });
                    }

                    // 更新健康状态文本
                    const healthMeta = document.querySelector('.health-meta');
                    if (healthMeta) {
                        healthMeta.innerHTML = `
                            <div class="meta-big-line">状态：${data.status}</div>
                            <div class="meta-big-line">阈值：${data.threshold}（注意）</div>
                            <div class="meta-small-line">范围：0-100</div>
                            <div class="meta-small-line">绿色区域</div>
                        `;
                    }

                    // 更新RUL摘要
                    const rulSummary = document.querySelector('.rul-summary');
                    if (rulSummary) {
                        rulSummary.innerHTML = `预测值：<strong>${data.rul_days} 天</strong>（约 ${data.rul_months} 个月）`;
                    }

                    // 更新多指标融合趋势图
                    const fusionChart = echarts.getInstanceByDom(document.getElementById('fusionTrendChart'));
                    if (fusionChart) {
                        fusionChart.setOption({
                            xAxis: {
                                data: data.fusion_trend.labels
                            },
                            series: data.fusion_trend.series.map(s => ({
                                name: s.name,
                                type: 'line',
                                smooth: true,
                                data: s.data
                            }))
                        });
                    }

                    // 更新异常检测趋势图
                    const anomalyChart = echarts.getInstanceByDom(document.getElementById('anomalyTrendChart'));
                    if (anomalyChart) {
                        anomalyChart.setOption({
                            xAxis: {
                                data: data.anomaly_trend.labels
                            },
                            series: [{
                                data: data.anomaly_trend.values
                            }]
                        });
                    }

                    // 更新RUL预测图
                    const rulChart = echarts.getInstanceByDom(document.getElementById('rulChart'));
                    if (rulChart) {
                        rulChart.setOption({
                            xAxis: {
                                data: data.rul_prediction.labels
                            },
                            series: [
                                {
                                    name: '实际值',
                                    data: data.rul_prediction.true_values
                                },
                                {
                                    name: '预测值',
                                    data: data.rul_prediction.predicted_values
                                },
                                {
                                    name: '置信区间上界',
                                    data: data.rul_prediction.confidence_upper
                                },
                                {
                                    name: '置信区间下界',
                                    data: data.rul_prediction.confidence_lower
                                }
                            ]
                        });
                    }

                    alert('模型运行完成！健康评估结果已更新。');
                })
                .catch(error => {
                    hideLoading();
                    alert('请求失败: ' + error.message);
                })
                .finally(() => {
                    runHealthBtn.disabled = false;
                });
        });
    }
});

window.addEventListener('resize', function () {
    const chartDoms = document.querySelectorAll('.chart-box, .gauge-chart');
    chartDoms.forEach(dom => {
        const instance = echarts.getInstanceByDom(dom);
        if (instance) {
            instance.resize();
        }
    });
});
