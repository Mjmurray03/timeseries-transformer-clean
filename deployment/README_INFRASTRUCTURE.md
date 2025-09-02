# Time Series Transformer - Production Infrastructure

## 📋 Infrastructure Overview

This document provides a comprehensive guide to the production-ready Docker and Kubernetes infrastructure for the Time Series Transformer project.

## 🐳 Docker Infrastructure

### Container Images

#### 1. Training Container (`Dockerfile.training`)
- **Base**: NVIDIA CUDA 12.1 with cuDNN 8 on Ubuntu 22.04
- **Purpose**: GPU-accelerated model training with full ML stack
- **Features**:
  - Multi-stage build for optimal size and security
  - CUDA development environment with PyTorch GPU support
  - MLflow and Weights & Biases integration
  - Git LFS support for large model files
  - Non-root user for security
  - Comprehensive health checks

#### 2. Inference Container (`Dockerfile.inference`)
- **Base**: Python 3.10 slim for minimal footprint
- **Purpose**: High-performance model serving API
- **Features**:
  - CPU-optimized PyTorch for inference
  - FastAPI with Gunicorn for production serving
  - Redis integration for caching
  - Prometheus metrics collection
  - Multi-worker configuration
  - Minimal dependencies for security

#### 3. Development Container (`Dockerfile.notebook`)
- **Base**: Official Jupyter stack with CUDA support
- **Purpose**: Interactive development and experimentation
- **Features**:
  - JupyterLab with extensive extensions
  - CUDA-enabled environment for GPU development
  - Data science and ML visualization libraries
  - Git integration and LFS support
  - Custom kernels and widgets

### Services Stack (Docker Compose)

```yaml
Services:
├── api (inference)          # Model serving API
├── training                 # GPU training jobs
├── notebook                 # Jupyter development
├── mlflow                   # Experiment tracking
├── minio                    # S3-compatible storage
├── postgres                 # Database backend
├── redis                    # Caching layer
├── prometheus               # Metrics collection
├── grafana                  # Monitoring dashboards
├── loki                     # Log aggregation
├── promtail                 # Log collection
└── nginx                    # Load balancer
```

## ☸️ Kubernetes Deployment

### Core Components

#### 1. Inference Deployment (`deployment.yaml`)
- **Replicas**: 3 with rolling updates
- **Resources**: 2-4 CPU cores, 2-4GB RAM, 1 GPU per pod
- **Features**:
  - Pod anti-affinity for availability
  - Prometheus metrics scraping
  - Health/readiness/startup probes
  - GPU node scheduling
  - ConfigMap and Secret integration

#### 2. Training Jobs (`training-deployment.yaml`)
- **Type**: Kubernetes Job for batch processing
- **Resources**: 2-4 CPU cores, 8-16GB RAM, 1 GPU
- **Features**:
  - GPU node targeting
  - MLflow integration
  - S3 artifact storage
  - Configurable hyperparameters
  - Service account with RBAC

#### 3. Supporting Services
- **MLflow**: Experiment tracking server
- **PostgreSQL**: Metadata backend
- **MinIO**: S3-compatible artifact storage
- **Redis**: Caching and session storage
- **Monitoring**: Prometheus, Grafana, Loki stack

## 🔧 MLflow Integration

### Configuration
```yaml
MLflow Components:
├── Tracking Server (Port 5000)
├── PostgreSQL Backend
├── MinIO Artifact Store
└── Model Registry
```

### Features
- **Experiment Tracking**: Parameters, metrics, artifacts
- **Model Registry**: Versioning and deployment
- **Artifact Storage**: S3-compatible with MinIO
- **REST API**: Integration with training and inference
- **UI Dashboard**: Web interface for experiment management

## 📁 Git LFS Configuration

### Tracked File Types
```
Model Files:      *.pt, *.pth, *.ckpt, *.safetensors
Data Files:       *.parquet, *.hdf5, *.feather
Artifacts:        models/**, artifacts/**, checkpoints/**
Archives:         *.zip, *.tar.gz
Media:            *.png, *.jpg, *.pdf
```

### Configuration
- **Storage**: Large files stored in Git LFS
- **Optimization**: Reduced repository size
- **CI/CD**: Automatic LFS handling in pipelines

## 🚀 Deployment Commands

### Local Development
```bash
# Start development stack
docker compose -f deployment/docker/docker-compose.yml up -d

# Start with specific profile
docker compose --profile development up -d

# Training job
docker compose --profile training up training

# Monitoring stack
docker compose --profile monitoring up -d
```

### Production Kubernetes
```bash
# Deploy namespace and RBAC
kubectl apply -f deployment/k8s/namespace.yaml
kubectl apply -f deployment/k8s/rbac.yaml

# Deploy storage
kubectl apply -f deployment/k8s/pvc.yaml

# Deploy services
kubectl apply -f deployment/k8s/service.yaml
kubectl apply -f deployment/k8s/deployment.yaml

# Deploy training
kubectl apply -f deployment/k8s/training-deployment.yaml

# Deploy ingress
kubectl apply -f deployment/k8s/ingress.yaml

# Scale deployment
kubectl scale deployment timeseries-transformer --replicas=5
```

## 📊 Monitoring and Observability

### Metrics
- **Prometheus**: Application and infrastructure metrics
- **Grafana**: Custom dashboards and alerting
- **Model Metrics**: Inference latency, accuracy, throughput

### Logging
- **Loki**: Centralized log aggregation
- **Promtail**: Log collection from containers
- **Structured Logging**: JSON format with correlation IDs

### Tracing
- **Health Checks**: Liveness, readiness, and startup probes
- **Model Performance**: Real-time inference monitoring
- **Resource Usage**: GPU, CPU, memory tracking

## 🔒 Security Features

### Container Security
- **Non-root Users**: All containers run as unprivileged users
- **Minimal Images**: Reduced attack surface
- **Security Scanning**: Automated vulnerability checks
- **Secrets Management**: Kubernetes secrets for sensitive data

### Network Security
- **Network Policies**: Restricted inter-pod communication
- **TLS Termination**: HTTPS/TLS for all external traffic
- **Service Mesh**: Optional Istio integration
- **RBAC**: Role-based access control

## ✅ Validation Checklist

### Infrastructure Validation
```bash
# Run comprehensive validation
python scripts/validate_infrastructure.py
```

**Validation Items:**
- ✅ Docker builds complete without errors
- ✅ Containers start and pass health checks  
- ✅ GPU accessible from training container
- ✅ MLflow tracking works across containers
- ✅ Redis caching functional
- ✅ API responds to requests
- ✅ Kubernetes deployment scales properly
- ✅ Git LFS tracks large files correctly

### Manual Testing
```bash
# Test Docker builds
docker build -f deployment/docker/Dockerfile.training -t ts-training .
docker build -f deployment/docker/Dockerfile.inference -t ts-inference .
docker build -f deployment/docker/Dockerfile.notebook -t ts-notebook .

# Test health checks
docker run --rm ts-inference python health_check.py

# Test GPU access
docker run --rm --gpus all ts-training nvidia-smi

# Test compose stack
docker compose up --dry-run
```

## 🔧 Troubleshooting

### Common Issues

#### Docker Build Failures
- **Solution**: Check Dockerfile syntax and base image availability
- **Debug**: Use `docker build --no-cache --progress=plain`

#### GPU Access Issues
- **Solution**: Install NVIDIA Docker runtime
- **Debug**: `docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi`

#### MLflow Connection Issues
- **Solution**: Verify PostgreSQL and MinIO connectivity
- **Debug**: Check service logs with `docker compose logs mlflow`

#### Kubernetes Pod Failures
- **Solution**: Check resource limits and node capacity
- **Debug**: `kubectl describe pod <pod-name>`

### Performance Optimization

#### Inference Performance
- **CPU**: Use optimized PyTorch build
- **Memory**: Tune JVM heap for better garbage collection
- **Caching**: Enable Redis for model prediction caching

#### Training Performance
- **GPU**: Use mixed precision training (FP16)
- **I/O**: Use fast SSD storage for data loading
- **Parallelism**: Distributed training across multiple GPUs

## 📚 Additional Resources

### Documentation
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [NVIDIA Docker Documentation](https://github.com/NVIDIA/nvidia-docker)

### Monitoring
- [Prometheus Configuration](../monitoring/prometheus.yml)
- [Grafana Dashboards](../monitoring/grafana/)
- [Alert Rules](../monitoring/alert-rules.yml)

---

## 🎯 Production Readiness

This infrastructure provides:

✅ **Scalability**: Horizontal scaling with load balancing  
✅ **Reliability**: Health checks and automatic restarts  
✅ **Observability**: Comprehensive monitoring and logging  
✅ **Security**: Non-root containers and RBAC  
✅ **Performance**: GPU acceleration and caching  
✅ **Maintainability**: Infrastructure as code  

The system is ready for production deployment with enterprise-grade reliability and performance.