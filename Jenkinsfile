pipeline {
    agent any

    environment {
        DOCKERHUB_USER = 'zenonix'
        IMAGE_NAME     = 'mlops-lab'
        IMAGE_TAG      = "v${BUILD_NUMBER}"
        ROLL_NO        = '2022BCS0179'
    }

    stages {

        // ── Stage 1: Checkout ────────────────────────────────
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        // ── Stage 2: Setup Python Virtual Environment ────────
        stage('Setup Python Virtual Environment') {
            steps {
                dir('lab6') {
                    sh '''
                        python3 -m venv venv
                        . venv/bin/activate
                        pip install --upgrade pip
                        pip install -r requirements.txt
                    '''
                }
            }
        }

        // ── Stage 3: Train Model ─────────────────────────────
        stage('Train Model') {
            steps {
                dir('lab6') {
                    sh '''
                        . venv/bin/activate
                        python scripts/train.py
                    '''
                }
            }
        }

        // ── Stage 4: Read Accuracy ───────────────────────────
        stage('Read Accuracy') {
            steps {
                dir('lab6') {
                    script {
                        def metrics = readJSON file: 'app/artifacts/metrics.json'
                        env.CURRENT_R2 = metrics.r2_score.toString()
                        env.CURRENT_MSE = metrics.mse.toString()
                        echo "Current R2: ${env.CURRENT_R2}"
                        echo "Current MSE: ${env.CURRENT_MSE}"
                    }
                }
            }
        }

        // ── Stage 5: Compare Accuracy ────────────────────────
        stage('Compare Accuracy') {
            steps {
                script {
                    withCredentials([string(credentialsId: 'best-accuracy', variable: 'BEST_R2')]) {
                        def current = env.CURRENT_R2.toDouble()
                        def best    = BEST_R2.toDouble()
                        echo "Current R2: ${current} | Best R2: ${best}"
                        if (current > best) {
                            env.DEPLOY = 'true'
                            echo "Model improved. Proceeding to build and push Docker image."
                        } else {
                            env.DEPLOY = 'false'
                            echo "${env.ROLL_NO}----Metric did not improve (R2: ${current} <= Best: ${best})"
                        }
                    }
                }
            }
        }

        // ── Stage 6: Build Docker Image (Conditional) ────────
        stage('Build Docker Image') {
            when {
                expression { env.DEPLOY == 'true' }
            }
            steps {
                dir('lab6') {
                    withCredentials([usernamePassword(
                        credentialsId: 'dockerhub-creds',
                        usernameVariable: 'DOCKER_USER',
                        passwordVariable: 'DOCKER_PASS'
                    )]) {
                        sh '''
                            echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                            docker build -t $DOCKER_USER/$IMAGE_NAME:$IMAGE_TAG .
                            docker tag $DOCKER_USER/$IMAGE_NAME:$IMAGE_TAG $DOCKER_USER/$IMAGE_NAME:latest
                        '''
                    }
                }
            }
        }

        // ── Stage 7: Push Docker Image (Conditional) ─────────
        stage('Push Docker Image') {
            when {
                expression { env.DEPLOY == 'true' }
            }
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-creds',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh '''
                        docker push $DOCKER_USER/$IMAGE_NAME:$IMAGE_TAG
                        docker push $DOCKER_USER/$IMAGE_NAME:latest
                    '''
                }
            }
        }
    }

    // ── Artifact Archiving ────────────────────────────────────
    post {
        always {
            archiveArtifacts artifacts: 'lab6/app/artifacts/**', fingerprint: true
        }
        success {
            echo "Pipeline completed successfully."
        }
        failure {
            echo "Pipeline failed."
        }
    }
}