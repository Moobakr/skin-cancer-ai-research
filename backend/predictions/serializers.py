from rest_framework import serializers


class PredictionRequestSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)
    age = serializers.FloatField(required=False, allow_null=True)
    sex = serializers.CharField(required=False, allow_blank=True)
    anatom_site = serializers.CharField(required=False, allow_blank=True)
    smoke = serializers.CharField(required=False, allow_blank=True)
    drink = serializers.CharField(required=False, allow_blank=True)

    def validate_image(self, value):
        max_mb = 10
        if value.size > max_mb * 1024 * 1024:
            raise serializers.ValidationError(f"Image file is too large. Maximum size is {max_mb} MB.")
        return value
