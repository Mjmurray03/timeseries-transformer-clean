"""
Custom Swagger UI Configuration for TimeSeries Transformer API
============================================================

This module provides a modern dark theme Swagger UI with enhanced functionality
for better API documentation experience. It includes custom styling, improved
navigation, and production-ready configuration options.

Features:
- Modern dark theme with syntax highlighting
- Enhanced UX with persistent authorization and deep linking
- Custom CSS for better visual appeal
- Optimized parameter settings for ML API documentation
- Professional branding and favicon
"""

from fastapi import FastAPI, Response
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
import json


def setup_custom_swagger(app: FastAPI):
    """
    Configure custom Swagger UI with modern dark theme and enhanced features.
    
    This function replaces the default Swagger UI with a customized version that:
    - Uses a dark theme optimized for development environments
    - Includes syntax highlighting for better code readability  
    - Provides persistent authorization for easier API testing
    - Enables advanced features like operation filtering and deep linking
    - Applies custom CSS for professional appearance
    
    Args:
        app (FastAPI): The FastAPI application instance to customize
    """
    
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        """
        Custom Swagger UI HTML endpoint with dark theme and enhanced configuration.
        
        Returns:
            HTMLResponse: Customized Swagger UI with modern styling and features
        """
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=f"{app.title} - Interactive API Documentation",
            swagger_ui_parameters={
                # Navigation and UX
                "deepLinking": True,               # Enable deep linking to operations
                "persistAuthorization": True,      # Keep auth tokens between sessions
                "displayOperationId": False,       # Hide operation IDs for cleaner look
                "displayRequestDuration": True,    # Show API response times
                "docExpansion": "list",           # Expand operations list by default
                "filter": True,                   # Enable operation filtering
                "showExtensions": True,           # Display vendor extensions
                "showCommonExtensions": True,     # Show common OpenAPI extensions
                "tryItOutEnabled": True,          # Enable "Try it out" functionality
                
                # Model display settings
                "defaultModelsExpandDepth": 1,    # Expand models one level deep
                "defaultModelExpandDepth": 1,     # Expand model properties one level
                
                # Syntax highlighting and theme
                "syntaxHighlight": {
                    "activate": True,
                    "theme": "monokai"           # Dark syntax highlighting theme
                },
                
                # Layout and appearance
                "layout": "StandaloneLayout",     # Use standalone layout
                "validatorUrl": None,            # Disable validator for faster loading
            }
        )
    
    @app.get("/swagger-custom.css", include_in_schema=False)
    async def custom_swagger_css():
        """
        Custom CSS styles for enhanced Swagger UI appearance.
        
        Provides additional styling beyond the default dark theme including:
        - Improved color scheme for better contrast
        - Custom branding elements
        - Enhanced readability for ML/AI API documentation
        - Responsive design improvements
        
        Returns:
            Response: CSS stylesheet with custom dark theme styles
        """
        custom_css = """
        /* Main Container Styling */
        .swagger-ui {
            background-color: #0d1117 !important;
            color: #c9d1d9 !important;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
        }
        
        /* Top Navigation Bar */
        .swagger-ui .topbar {
            background-color: #161b22 !important;
            border-bottom: 1px solid #30363d !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3) !important;
        }
        
        .swagger-ui .topbar .download-url-wrapper {
            background-color: #21262d !important;
            border-radius: 6px !important;
        }
        
        .swagger-ui .topbar .download-url-wrapper input[type=text] {
            background-color: #0d1117 !important;
            border: 1px solid #30363d !important;
            color: #c9d1d9 !important;
        }
        
        /* API Information Section */
        .swagger-ui .info {
            background-color: #161b22 !important;
            border: 1px solid #30363d !important;
            border-radius: 8px !important;
            margin: 20px 0 !important;
            padding: 20px !important;
        }
        
        .swagger-ui .info .title {
            color: #58a6ff !important;
            font-size: 2.5em !important;
            font-weight: 700 !important;
        }
        
        .swagger-ui .info .description {
            color: #8b949e !important;
            font-size: 1.1em !important;
            line-height: 1.6 !important;
        }
        
        /* Operation Blocks */
        .swagger-ui .opblock {
            background-color: #21262d !important;
            border: 1px solid #30363d !important;
            border-radius: 8px !important;
            margin-bottom: 15px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
        }
        
        .swagger-ui .opblock.opblock-post {
            border-color: #238636 !important;
        }
        
        .swagger-ui .opblock.opblock-get {
            border-color: #1f6feb !important;
        }
        
        .swagger-ui .opblock.opblock-put {
            border-color: #fb8500 !important;
        }
        
        .swagger-ui .opblock.opblock-delete {
            border-color: #f85149 !important;
        }
        
        /* Operation Summary */
        .swagger-ui .opblock .opblock-summary {
            background-color: transparent !important;
            border-bottom: 1px solid #30363d !important;
        }
        
        .swagger-ui .opblock .opblock-summary .opblock-summary-method {
            font-weight: 700 !important;
            min-width: 80px !important;
            border-radius: 4px !important;
        }
        
        .swagger-ui .opblock .opblock-summary-path {
            color: #c9d1d9 !important;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace !important;
        }
        
        /* Parameters Section */
        .swagger-ui .parameters-container {
            background-color: #0d1117 !important;
            border-radius: 6px !important;
            padding: 15px !important;
            margin: 15px 0 !important;
        }
        
        .swagger-ui .parameter__name {
            color: #58a6ff !important;
            font-weight: 600 !important;
        }
        
        .swagger-ui .parameter__type {
            color: #f0883e !important;
            font-weight: 500 !important;
        }
        
        /* Input Fields */
        .swagger-ui input[type=text],
        .swagger-ui input[type=password],
        .swagger-ui input[type=email],
        .swagger-ui textarea,
        .swagger-ui select {
            background-color: #0d1117 !important;
            border: 1px solid #30363d !important;
            color: #c9d1d9 !important;
            border-radius: 6px !important;
            padding: 8px 12px !important;
        }
        
        .swagger-ui input[type=text]:focus,
        .swagger-ui textarea:focus,
        .swagger-ui select:focus {
            border-color: #58a6ff !important;
            box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.1) !important;
            outline: none !important;
        }
        
        /* Buttons */
        .swagger-ui .btn {
            border-radius: 6px !important;
            font-weight: 500 !important;
            padding: 8px 16px !important;
            transition: all 0.2s ease !important;
        }
        
        .swagger-ui .btn.execute {
            background-color: #238636 !important;
            border-color: #238636 !important;
            color: #ffffff !important;
        }
        
        .swagger-ui .btn.execute:hover {
            background-color: #2ea043 !important;
            transform: translateY(-1px) !important;
        }
        
        .swagger-ui .btn.cancel {
            background-color: #21262d !important;
            border-color: #30363d !important;
            color: #c9d1d9 !important;
        }
        
        .swagger-ui .btn.try-out__btn {
            background-color: #1f6feb !important;
            border-color: #1f6feb !important;
            color: #ffffff !important;
        }
        
        /* Response Section */
        .swagger-ui .responses-wrapper {
            background-color: #0d1117 !important;
            border-radius: 6px !important;
            margin-top: 20px !important;
        }
        
        .swagger-ui .response {
            border: 1px solid #30363d !important;
            border-radius: 6px !important;
            margin-bottom: 10px !important;
        }
        
        .swagger-ui .response .response-col_status {
            color: #238636 !important;
            font-weight: 700 !important;
        }
        
        /* Code Blocks */
        .swagger-ui pre {
            background-color: #161b22 !important;
            border: 1px solid #30363d !important;
            border-radius: 6px !important;
            color: #c9d1d9 !important;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace !important;
            padding: 15px !important;
        }
        
        .swagger-ui .highlight-code {
            background-color: #161b22 !important;
            border-radius: 6px !important;
        }
        
        /* Models Section */
        .swagger-ui .model-container {
            background-color: #0d1117 !important;
            border: 1px solid #30363d !important;
            border-radius: 6px !important;
            margin: 15px 0 !important;
        }
        
        .swagger-ui .model .model-title {
            color: #58a6ff !important;
            font-size: 1.2em !important;
            font-weight: 600 !important;
        }
        
        .swagger-ui .property-row .property-name {
            color: #79c0ff !important;
            font-weight: 600 !important;
        }
        
        .swagger-ui .property-row .property-type {
            color: #f0883e !important;
        }
        
        /* Scrollbars */
        .swagger-ui ::-webkit-scrollbar {
            width: 8px !important;
            height: 8px !important;
        }
        
        .swagger-ui ::-webkit-scrollbar-track {
            background-color: #21262d !important;
            border-radius: 4px !important;
        }
        
        .swagger-ui ::-webkit-scrollbar-thumb {
            background-color: #30363d !important;
            border-radius: 4px !important;
        }
        
        .swagger-ui ::-webkit-scrollbar-thumb:hover {
            background-color: #484f58 !important;
        }
        
        /* Loading Animation */
        .swagger-ui .loading-container {
            background-color: #0d1117 !important;
        }
        
        /* Custom TimeSeries API Branding */
        .swagger-ui .info::before {
            content: "🤖 TimeSeries Transformer";
            display: block;
            font-size: 0.9em;
            color: #58a6ff;
            margin-bottom: 10px;
            font-weight: 500;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .swagger-ui .info .title {
                font-size: 1.8em !important;
            }
            
            .swagger-ui .opblock {
                margin-bottom: 10px !important;
            }
        }
        
        /* Animation for smooth transitions */
        .swagger-ui .opblock,
        .swagger-ui .btn,
        .swagger-ui input,
        .swagger-ui textarea {
            transition: all 0.2s ease !important;
        }
        
        /* Focus indicators for accessibility */
        .swagger-ui .btn:focus,
        .swagger-ui input:focus,
        .swagger-ui textarea:focus {
            outline: 2px solid #58a6ff !important;
            outline-offset: 2px !important;
        }
        """
        
        return Response(
            content=custom_css,
            media_type="text/css",
            headers={
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                "Content-Type": "text/css; charset=utf-8"
            }
        )
    
    @app.get("/docs/oauth2-redirect", include_in_schema=False)
    async def swagger_ui_redirect():
        """OAuth2 redirect endpoint for Swagger UI authentication flows."""
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Swagger UI OAuth2 Redirect</title>
        </head>
        <body>
            <script>
                'use strict';
                function run() {
                    var oauth2 = window.opener.swaggerUIRedirectOauth2;
                    var sentState = oauth2.state;
                    var redirectUrl = oauth2.redirectUrl;
                    var isValid, qp, arr;
                    
                    if (/code|token|error/.test(window.location.hash)) {
                        qp = window.location.hash.substring(1);
                    } else {
                        qp = location.search.substring(1);
                    }
                    
                    arr = qp.split("&");
                    arr.forEach(function(v,i,_arr) { _arr[i] = '"' + v.replace('=', '":"') + '"'; });
                    qp = qp ? JSON.parse('{' + arr.join(',') + '}', function(key, value) {
                        return key === "" ? value : decodeURIComponent(value);
                    }) : {};
                    
                    isValid = qp.state === sentState;
                    
                    if ((oauth2.auth.schema.get("flow") === "accessCode" || oauth2.auth.schema.get("flow") === "authorizationCode") && !oauth2.auth.code) {
                        if (!isValid) {
                            oauth2.errCb({
                                authId: oauth2.auth.name,
                                source: "auth",
                                level: "warning",
                                message: "Authorization may be unsafe, passed state was changed in server Passed state wasn't returned from auth server"
                            });
                        }
                        
                        if (qp.code) {
                            delete oauth2.state;
                            oauth2.auth.code = qp.code;
                            oauth2.callback({auth: oauth2.auth, redirectUrl: redirectUrl});
                        } else {
                            let oauthErrorMsg;
                            if (qp.error) {
                                oauthErrorMsg = "["+qp.error+"]: " + (qp.error_description ? qp.error_description+ ". " : "no accessCode received from the server. ") + (qp.error_uri ? "More info: "+qp.error_uri : "");
                            }
                            
                            oauth2.errCb({
                                authId: oauth2.auth.name,
                                source: "auth",
                                level: "error",
                                message: oauthErrorMsg || "[Authorization failed]: no accessCode received from the server"
                            });
                        }
                    } else {
                        oauth2.callback({auth: oauth2.auth, token: qp, isValid: isValid, redirectUrl: redirectUrl});
                    }
                    window.close();
                }
                
                if (document.readyState !== 'loading') {
                    run();
                } else {
                    document.addEventListener('DOMContentLoaded', function() {
                        run();
                    });
                }
            </script>
        </body>
        </html>
        """)


def customize_openapi_schema(app: FastAPI):
    """
    Enhance the OpenAPI schema with additional metadata and examples.
    
    This function customizes the generated OpenAPI specification to provide
    better documentation including:
    - Enhanced descriptions and examples
    - Custom tags and metadata
    - Server information
    - Contact and license details
    
    Args:
        app (FastAPI): The FastAPI application instance to customize
    """
    
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
            
        # Generate base OpenAPI schema
        from fastapi.openapi.utils import get_openapi
        
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        
        # Add custom metadata
        openapi_schema["info"].update({
            "contact": {
                "name": "TimeSeries Transformer API",
                "url": "https://github.com/Mjmurray03/timeseries-transformer-clean",
                "email": "support@timeseries-api.com"
            },
            "license": {
                "name": "MIT License",
                "url": "https://opensource.org/licenses/MIT"
            },
            "termsOfService": "https://api.timeseries-transformer.com/terms"
        })
        
        # Add server information
        openapi_schema["servers"] = [
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://api.timeseries-transformer.com",
                "description": "Production server"
            }
        ]
        
        # Enhance tags with descriptions
        openapi_schema["tags"] = [
            {
                "name": "Health",
                "description": "System health and status monitoring endpoints"
            },
            {
                "name": "Predictions", 
                "description": "Machine learning prediction endpoints for time series analysis"
            },
            {
                "name": "Models",
                "description": "Model information and management endpoints"
            },
            {
                "name": "Backtesting",
                "description": "Historical performance analysis and backtesting endpoints"
            }
        ]
        
        # Add custom examples to prediction endpoints
        if "/predict" in openapi_schema["paths"]:
            predict_schema = openapi_schema["paths"]["/predict"]["post"]
            
            # Add comprehensive examples
            predict_schema["requestBody"]["content"]["application/json"]["examples"] = {
                "2d_array_format": {
                    "summary": "2D Array Format (60x10)",
                    "description": "Features as 2D array: 60 days with 10 features each",
                    "value": {
                        "ticker": "AAPL",
                        "features": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] for _ in range(60)],
                        "horizon": 3
                    }
                },
                "flat_array_format": {
                    "summary": "Flat Array Format (600 elements)",
                    "description": "Features as flat array: 600 elements (60 days * 10 features)",
                    "value": {
                        "ticker": "MSFT",
                        "features": [0.5] * 600,
                        "horizon": 5
                    }
                }
            }
            
            # Enhance response examples
            predict_schema["responses"]["200"]["content"]["application/json"]["examples"] = {
                "successful_prediction": {
                    "summary": "Successful Prediction Response",
                    "description": "Complete prediction response with all components",
                    "value": {
                        "ticker": "AAPL",
                        "predictions": {
                            "price_predictions": [150.25, 151.30, 149.85],
                            "direction_predictions": [1, 1, 0],
                            "volatility_predictions": [0.15, 0.18, 0.12],
                            "quantile_predictions": {
                                "q10": [148.50, 149.20, 147.90],
                                "q50": [150.25, 151.30, 149.85],
                                "q90": [152.10, 153.50, 151.95]
                            }
                        },
                        "confidence_intervals": {
                            "lower_bound": [148.50, 149.20, 147.90],
                            "upper_bound": [152.10, 153.50, 151.95]
                        },
                        "timestamp": "2024-01-15T10:30:00Z",
                        "model_version": "1.0.0",
                        "cache_hit": False
                    }
                }
            }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi