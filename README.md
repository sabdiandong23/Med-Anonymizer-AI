

流程图：
<img width="1452" height="2801" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/89f6cea5-3bbe-4aec-9b8f-b2682ce3d535" />

baseline结果：
Model summary (fused): 73 layers, 3,006,233 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 110/110 5.7it/s 19.3s0.1s
                   all       1752      15422      0.933      0.859      0.943      0.751
          patient_info       1402       5608      0.923      0.827      0.927      0.711
             time_info       1402       4206      0.992      0.918      0.969      0.787
      institution_info       1402       5608      0.884      0.832      0.931      0.752
Speed: 0.3ms preprocess, 2.3ms inference, 0.0ms loss, 1.8ms postprocess per image

backbone的P3/4加CBAM结果：
YOLOv8n_cbam_p34 summary (fused): 83 layers, 3,027,101 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 110/110 5.4it/s 20.5s0.1s
                   all       1752      15422       0.89      0.888      0.931      0.736
          patient_info       1402       5608      0.909       0.82      0.933      0.783
             time_info       1402       4206      0.974      0.931      0.984      0.803
      institution_info       1402       5608      0.787      0.913      0.877      0.623
Speed: 0.3ms preprocess, 2.7ms inference, 0.0ms loss, 1.8ms postprocess per image

backbone的P3/4加ECA的结果：
YOLOv8n_eca_p34 summary (fused): 79 layers, 3,006,239 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 110/110 5.7it/s 19.2s0.1s
                   all       1752      15422      0.909      0.887      0.946      0.747
          patient_info       1402       5608      0.923      0.823       0.94      0.789
             time_info       1402       4206      0.983      0.939      0.988      0.811
      institution_info       1402       5608       0.82      0.898      0.909      0.641
Speed: 0.3ms preprocess, 2.3ms inference, 0.0ms loss, 2.0ms postprocess per image

NECK加ECA的结果：
YOLOv8n_eca_neck_p3 summary (fused): 76 layers, 3,006,236 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 110/110 5.3it/s 20.8s0.2s
                   all       1752      15422       0.91      0.899      0.953       0.76
          patient_info       1402       5608      0.904       0.86      0.954      0.807
             time_info       1402       4206      0.981       0.96      0.991      0.817
      institution_info       1402       5608      0.843      0.875      0.914      0.655
Speed: 0.3ms preprocess, 2.4ms inference, 0.0ms loss, 2.2ms postprocess per image

