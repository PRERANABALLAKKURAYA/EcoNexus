pipeline {
  agent any
  environment { IMAGE = "econexus"; TAG = "${env.BUILD_NUMBER}" }
  stages {
    stage("Checkout")    { steps { checkout scm } }
    stage("Install")     { steps { sh "pip install -r requirements.txt" } }
    stage("Lint")        { steps { sh "flake8 src tests" } }
    stage("Test")        { steps { sh "pytest --junitxml=report.xml" } }
    stage("Build image") { steps { sh "docker build -t $IMAGE:$TAG ." } }
    stage("Push + Deploy") {
      when { branch "main" }
      steps {
        // REGISTRY and registry login come from Jenkins credentials / environment (NFR8)
        sh "./scripts/push_and_deploy.sh $IMAGE $TAG"
      }
    }
  }
  post { always { junit "report.xml" } }
}
