# Pull latest image of tensorflow serving
FROM tensorflow/serving:latest

COPY . /

# Expose ports
# gRPC
EXPOSE 8500
# REST
EXPOSE 8501

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
CMD ["--model_name=malaria", "--model_base_path=/tf_serving"]
