# Time-Series Transformer Deployment

Complete deployment infrastructure for the time-series transformer ML model following deployment-standards.md specifications.

## 📁 Directory Structure

```
deployment/
├── docker/                 # Docker containerization
│   ├── Dockerfile          # Multi-stage build (builder, tester, training, inference, dev)
│   ├── docker-compose.yml  # Local development stack
│   ├── nginx.conf          # Load balancer configuration
│   └── health_check.py     # Container health checks
├── k8s/                    # Kubernetes manifests
│   ├── namespace.yaml      # Namespace definition
│   ├── deployment.yaml     # Model server deployment
│   ├── service.yaml        # Service definitions (LoadBalancer, ClusterIP, Headless)
│   ├── ingress.yaml        # Ingress with TLS and rate limiting
│   ├── hpa.yaml            # Horizontal Pod Autoscaler + VPA
│   ├── configmap.yaml      # Configuration management
│   ├── secrets.yaml        # Secrets management
│   ├── pvc.yaml            # Persistent storage
│   └── rbac.yaml           # Service accounts and RBAC
├── helm/                   # Helm charts
│   ├── Chart.yaml          # Chart metadata with dependencies
│   ├── values.yaml         # Default values
│   ├── values-dev.yaml     # Development environment
│   ├── values-staging.yaml # Staging environment
│   ├── values-prod.yaml    # Production environment
│   └── templates/          # Helm templates
│       ├── _helpers.tpl    # Template helpers
│       ├── deployment.yaml # Deployment template
│       ├── service.yaml    # Service template
│       ├── ingress.yaml    # Ingress template
│       └── hpa.yaml        # HPA template
├── monitoring/             # Observability stack
│   ├── prometheus.yml      # Prometheus configuration
│   ├── alert-rules.yml     # Alerting rules
│   ├── loki-config.yml     # Log aggregation
│   ├── promtail-config.yml # Log collection
│   └── grafana/
│       └── dashboards/
│           └── model-dashboard.json
├── ci-cd/                  # CI/CD pipeline
│   └── validate-deployment.py  # Pre-deployment validation
└── validation-requirements.yaml # Deployment requirements
```

## 🐳 Docker Deployment

### Multi-Stage Build Targets

- **builder**: Builds the wheel package
- **tester**: Runs tests with coverage
- **training**: Full ML stack with GPU support
- **inference**: Minimal production runtime
- **development**: Full dev environment with hot reload

### Local Development

```bash
# Start full development stack
docker-compose up -d

# API only
docker-compose up api

# Training environment
docker-compose --profile training up training

# Monitoring stack
docker-compose --profile monitoring up
```

### Production Build

```bash
# Build inference image
docker build -f deployment/docker/Dockerfile --target inference -t timeseries-transformer:v1.0.0 .

# Build training image
docker build -f deployment/docker/Dockerfile --target training -t timeseries-transformer:training .
```

## ☸️ Kubernetes Deployment

### Quick Start

```bash
# Create namespace and deploy
kubectl apply -f deployment/k8s/namespace.yaml
kubectl apply -f deployment/k8s/

# Check deployment status
kubectl get pods -n timeseries-transformer
kubectl logs -f deployment/timeseries-transformer -n timeseries-transformer
```

### Resource Requirements

- **CPU**: 1000m request, 2000m limit
- **Memory**: 2Gi request, 4Gi limit  
- **GPU**: 1x NVIDIA Tesla T4/V100
- **Storage**: 50Gi fast SSD for models

### Auto Scaling

- **HPA**: 2-10 replicas based on CPU (70%), Memory (80%), and custom metrics
- **VPA**: Automatic resource right-sizing
- **Cluster Autoscaler**: Node scaling based on resource demand

## ⎈ Helm Deployment

### Installation

```bash
# Add dependencies
helm dependency update deployment/helm

# Install with default values
helm install timeseries-transformer deployment/helm

# Install with environment-specific values
helm install timeseries-transformer deployment/helm -f deployment/helm/values-prod.yaml

# Upgrade deployment
helm upgrade timeseries-transformer deployment/helm --set image.tag=v1.1.0
```

### Environment Configurations

#### Development
```bash
helm install ts-dev deployment/helm -f deployment/helm/values-dev.yaml
```
- 1 replica, CPU-only
- Local storage
- Debug logging
- No monitoring

#### Staging
```bash
helm install ts-staging deployment/helm -f deployment/helm/values-staging.yaml
```
- 2 replicas with GPU
- LoadBalancer service
- Prometheus monitoring
- Staging domain

#### Production
```bash
helm install ts-prod deployment/helm -f deployment/helm/values-prod.yaml
```
- 3-20 replicas with auto-scaling
- Multi-AZ deployment
- Full monitoring stack
- Production domain with TLS

## 🔍 Monitoring & Observability

### Metrics Collection

- **Prometheus**: Metrics scraping and storage
- **Custom Metrics**: Model performance, inference latency, business metrics
- **Node Exporter**: Infrastructure metrics
- **cAdvisor**: Container metrics

### Logging

- **Loki**: Centralized log aggregation
- **Promtail**: Log collection from pods
- **Structured Logging**: JSON format with trace correlation

### Dashboards

- **Model Performance**: Latency, throughput, accuracy
- **Infrastructure**: CPU, memory, GPU utilization
- **Business Metrics**: Sharpe ratio, returns, drawdown
- **Alerts**: Critical thresholds and notifications

### Key Alerts

- Model latency P99 > 100ms
- Error rate > 5%
- Memory usage > 90%
- Model accuracy degradation
- GPU memory exhaustion

## 🚀 CI/CD Pipeline

### GitHub Actions Workflow

Triggers on:
- Push to `main`/`develop`
- Tags matching `v*`
- Manual dispatch

### Pipeline Stages

1. **Test & Quality**
   - Linting (black, isort, flake8)
   - Type checking (mypy)
   - Security scanning (bandit, safety)
   - Unit tests with coverage (80% minimum)

2. **Build & Push**
   - Multi-stage Docker build
   - Container registry push (GitHub Container Registry)
   - Build caching for faster builds

3. **Security Scan**
   - Trivy vulnerability scanning
   - SARIF reporting to GitHub Security

4. **Deploy Staging**
   - Automatic on `develop` branch
   - Smoke tests validation
   - Slack notifications

5. **Deploy Production**
   - Manual approval required
   - Canary deployment (10% traffic)
   - Full rollout after validation
   - Comprehensive smoke tests

### Deployment Validation

Pre-deployment checks include:
- Model file integrity
- Inference speed validation
- Memory usage verification
- API compatibility testing
- Security scanning
- Configuration validation

```bash
python deployment/ci-cd/validate-deployment.py \
  --model-path models/latest \
  --requirements deployment/validation-requirements.yaml
```

## 🔐 Security

### Container Security
- Non-root user execution
- Read-only root filesystem
- Security context constraints
- Minimal base images
- Regular vulnerability scanning

### Network Security
- TLS encryption (ingress)
- Network policies
- Rate limiting
- CORS configuration
- API authentication

### Secrets Management
- Kubernetes secrets for sensitive data
- External secrets operator integration
- Rotation policies
- Least privilege access

## 📊 Performance Optimization

### Model Serving
- TorchScript JIT compilation
- Batch processing optimization
- Response caching (Redis)
- Connection pooling
- Graceful degradation

### Infrastructure
- GPU memory optimization
- CPU/memory right-sizing
- Storage performance tuning
- Network optimization
- Load balancer configuration

### Monitoring
- Real-time performance metrics
- Automatic alerting
- Performance trending
- Capacity planning
- SLO/SLI tracking

## 🛠️ Operations

### Health Checks

```bash
# Application health
curl https://api.timeseries-transformer.com/health

# Readiness check
curl https://api.timeseries-transformer.com/ready

# Model info
curl https://api.timeseries-transformer.com/model/info
```

### Troubleshooting

```bash
# Check pod status
kubectl get pods -n timeseries-transformer

# View logs
kubectl logs -f deployment/timeseries-transformer -n timeseries-transformer

# Check resource usage
kubectl top pods -n timeseries-transformer

# Describe pod issues
kubectl describe pod <pod-name> -n timeseries-transformer
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment timeseries-transformer --replicas=5 -n timeseries-transformer

# Update HPA settings
kubectl patch hpa timeseries-transformer -p '{"spec":{"maxReplicas":20}}' -n timeseries-transformer
```

### Updates & Rollbacks

```bash
# Update deployment
helm upgrade timeseries-transformer deployment/helm --set image.tag=v1.1.0

# Rollback deployment
kubectl rollout undo deployment/timeseries-transformer -n timeseries-transformer

# Check rollout status
kubectl rollout status deployment/timeseries-transformer -n timeseries-transformer
```

## 📚 Additional Resources

- [Deployment Standards](./.kiro/steering/deployment-standards.md)
- [ML Infrastructure Guide](./.kiro/steering/ml-infrastructure.md)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/cluster-administration/manage-deployment/)
- [Helm Chart Development](https://helm.sh/docs/chart_best_practices/)
- [Prometheus Monitoring](https://prometheus.io/docs/practices/naming/)

## 🤝 Support

For deployment issues:
1. Check the troubleshooting section above
2. Review logs and metrics in Grafana
3. Consult the runbooks in the monitoring alerts
4. Contact the ML Platform team via Slack #ml-platform