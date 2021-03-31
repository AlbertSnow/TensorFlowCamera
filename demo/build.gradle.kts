plugins {
    id("com.android.application")
    id("kotlin-android")
}

android {
    setCompileSdkVersion(property("compileSdkVersion") as Int)
    defaultConfig {
        applicationId = "com.otaliastudios.cameraview.demo"
        setMinSdkVersion(property("minSdkVersion") as Int)
        setTargetSdkVersion(property("targetSdkVersion") as Int)
        versionCode = 1
        versionName = "1.0"
        vectorDrawables.useSupportLibrary = true
    }
    sourceSets["main"].java.srcDir("src/main/kotlin")

    aaptOptions {
        noCompress("tflite")
    }
}

dependencies {
    implementation(project(":cameraview"))
    implementation("androidx.appcompat:appcompat:1.2.0")
    implementation("com.google.android.material:material:1.2.0")

    implementation("org.tensorflow:tensorflow-lite:0.0.0-nightly")
    implementation("org.tensorflow:tensorflow-lite-gpu:0.0.0-nightly")
    implementation("org.tensorflow:tensorflow-lite-support:0.0.0-nightly")
}
