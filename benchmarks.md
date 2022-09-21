# Бенчмарки моделей на различных датасетах

| Model | Data                                                          | Accuracy       | Precision      | Recall         | F1  |
| ----- | ------------------------------------------------------------- | -------------- | -------------- | -------------- | --- |
|       | **CLUSTERS** <br> **PATTERN** <br> **Github** <br> **Twitch** | <br> <br> <br> | <br> <br> <br> | <br> <br> <br> |
| CatBoost | **CLUSTERS** <br> **PATTERN** <br> **Github** <br> **Twitch** | -<br> -<br> 0.845358<br> 0.659065| <br> <br> <br> | <br> <br> <br> | <br> <br> 0.714635<br> 0.198391
| GCNConv | **CLUSTERS** <br> **PATTERN** <br> **Github** <br> **Twitch** | -<br> -<br> 0.8374<br> 0.5667| <br> <br> <br> | <br> <br> <br> | <br> <br> 0.7162<br> 0.3471
| GCNConv + CatBoost | **CLUSTERS** <br> **PATTERN** <br> **Github** <br> **Twitch** | -<br> -<br> 0.867745<br> 0.753934| <br> <br> <br> | <br> <br> <br> | <br> <br> 0.724252<br> 0.166374
| Pure attention | **CLUSTERS** <br> **PATTERN** <br> **Github** <br> **Twitch** | -<br> -<br> 0.5846<br> 0.1676| <br> <br> <br> | <br> <br> <br> | <br> <br> 0.4692<br> 0.3497
| attention + GCNConv | **CLUSTERS** <br> **PATTERN** <br> **Github** <br> **Twitch** | -<br> -<br> 0.3760<br> 0.2440| <br> <br> <br> | <br> <br> <br> | <br> <br> 0.4246<br> 0.3923