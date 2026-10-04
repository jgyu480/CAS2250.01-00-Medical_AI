# VESSL 준비

코드는 GitHub에서 가져오고 데이터·가중치는 별도로 연결한다.
common/paths.py는 OCELOT_DATA_ROOT, SSL_WEIGHTS_PATH, OUTPUT_ROOT를 읽는다.
실행 image·GPU·저장공간·결과 보존·Run YAML은 10단계에서 준비한다.
현재는 클라우드 Run을 생성하지 않으며 학습도 시작하지 않는다.
