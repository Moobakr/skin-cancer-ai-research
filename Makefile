.PHONY: backend frontend train-image train-multimodal

backend:
	cd backend && python manage.py runserver

frontend:
	cd frontend && npm run dev

train-image:
	cd ai_model && python -m src.train_image_model --dataset-dir ../data/pad_ufes_20 --output-dir artifacts/image_model --model efficientnet --epochs 25 --batch-size 16

train-multimodal:
	cd ai_model && python -m src.train_multimodal_model --dataset-dir ../data/pad_ufes_20 --output-dir artifacts/multimodal_model --epochs 25 --batch-size 16
