import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.chaquo.python")
}

android {
    namespace = "dev.spektr"
    compileSdk = 35

    defaultConfig {
        applicationId = "dev.spektr"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        ndk {
            // Both by default: arm64-v8a is every real device, x86_64 is the
            // emulator, and a debug build that cannot run on the emulator is a
            // nuisance to inherit.
            //
            // But CPython and numpy ship per ABI, so the second architecture
            // roughly doubles the APK. `-Pabi=arm64-v8a` builds just the one,
            // which is what you want when the APK has to reach a device
            // through something with a size limit on it. Same code, same
            // assets, one less architecture.
            val requested = (project.findProperty("abi") as String?)
                ?.split(",")?.map { it.trim() }?.filter { it.isNotEmpty() }
            abiFilters += requested ?: listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

// The app shows the changelog on its home screen, and there is exactly one
// changelog. Copying it in at build time rather than keeping a second copy
// under assets/ means the shipped one cannot quietly fall behind the real
// one, which is the only failure mode a changelog really has.
val copyChangelog by tasks.registering(Copy::class) {
    from(rootProject.file("../CHANGELOG.md"))
    into(layout.buildDirectory.dir("generated/changelog/assets"))
}

android.sourceSets.getByName("main").assets.srcDir(
    layout.buildDirectory.dir("generated/changelog/assets")
)

tasks.matching { it.name.startsWith("merge") && it.name.endsWith("Assets") }
    .configureEach { dependsOn(copyChangelog) }

chaquopy {
    defaultConfig {
        version = "3.13"
        pip {
            install("numpy")
            install("rich")
            install("textual")
        }
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
}
