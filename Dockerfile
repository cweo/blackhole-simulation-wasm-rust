# Production stage - serve with nginx
FROM nginx:alpine AS production

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy static files
COPY index.html /usr/share/nginx/html/
COPY style.css /usr/share/nginx/html/

# Copy the pre-built WASM package (build locally with: wasm-pack build --target web --release)
COPY pkg /usr/share/nginx/html/pkg

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
