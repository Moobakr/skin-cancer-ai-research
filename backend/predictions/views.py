from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

from .model_service import model_service
from .preprocessing import open_uploaded_image
from .quality import analyze_image_quality
from .serializers import PredictionRequestSerializer


class HealthView(APIView):
    def get(self, request):
        return Response(model_service.health())


class PredictView(APIView):
    def post(self, request):
        serializer = PredictionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        uploaded_image = serializer.validated_data["image"]
        image = open_uploaded_image(uploaded_image)
        quality_report = analyze_image_quality(image)
        result = model_service.predict(image=image, request_data=serializer.validated_data, quality_report=quality_report)
        return Response(result)
