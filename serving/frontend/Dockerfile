# Use a lightweight web server image as the base
FROM nginx:alpine

# Copy the contents of the frontend directory to the nginx html directory
COPY . /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Command to start nginx when the container starts
CMD ["nginx", "-g", "daemon off;"]
