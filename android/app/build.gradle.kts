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
        // The port's version, from gradle.properties — not the desktop
        // release's. spektr 0.4.0 is the release; this is the second version
        // of the Android build that ships inside it.
        //
        // versionCode has to increase monotonically forever, because Android
        // refuses an update whose code is not higher, so it is derived from
        // the name rather than hand-bumped alongside it: 0.2.0 becomes 200,
        // 1.2.3 becomes 10203.
        versionName = (project.findProperty("spektrAndroidVersion") as String?)
            ?: error("spektrAndroidVersion is missing from gradle.properties")
        versionCode = versionName!!.split("-")[0].split(".").let { (a, b, c) ->
            a.toInt() * 10000 + b.toInt() * 100 + c.toInt()
        }
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

    // Signed only when the key is present.
    //
    // An unsigned release APK cannot be installed at all, so this is not a
    // nicety — it is the difference between an artifact and a file. The key
    // lives in GitHub secrets and reaches this build as a decoded file path
    // in the environment; a checkout without it still builds, unsigned, which
    // is what you want locally and what you must not publish.
    signingConfigs {
        create("release") {
            val store = System.getenv("SPEKTR_KEYSTORE")
            if (store != null) {
                storeFile = file(store)
                storePassword = System.getenv("SPEKTR_KEYSTORE_PASSWORD")
                keyAlias = System.getenv("SPEKTR_KEY_ALIAS")
                keyPassword = System.getenv("SPEKTR_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            if (System.getenv("SPEKTR_KEYSTORE") != null) {
                signingConfig = signingConfigs.getByName("release")
            }
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
//
// Wired through AGP's generated-source API rather than by hand. Hand-wiring
// was wrong twice: naming the merge tasks let a clean release build ship an
// APK with no changelog in it, because the merge ran before the copy; adding
// the lint tasks then hit `generateReleaseLintVitalReportModel`, which reads
// the same directory and is not called anything you would guess. Every one of
// those is the same missing edge, and `addGeneratedSourceDirectory` is the
// mechanism that draws it for every consumer, present and future.
//
// A Copy task cannot be used here — the API wires to a DirectoryProperty and
// Copy exposes a plain File — so this is the same job with a typed output.
abstract class CopyChangelog : DefaultTask() {
    @get:InputFile
    abstract val source: RegularFileProperty

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @TaskAction
    fun copy() {
        val dir = outputDir.get().asFile
        dir.mkdirs()
        source.get().asFile.copyTo(dir.resolve("CHANGELOG.md"), overwrite = true)
    }
}

val copyChangelog = tasks.register<CopyChangelog>("copyChangelog") {
    source.set(rootProject.layout.projectDirectory.file("../CHANGELOG.md"))
}

androidComponents {
    onVariants { variant ->
        variant.sources.assets?.addGeneratedSourceDirectory(
            copyChangelog, CopyChangelog::outputDir
        )
    }
}

chaquopy {
    defaultConfig {
        version = "3.13"
        // Chaquopy needs an interpreter of this same version on the *build*
        // machine to resolve the wheels below, and it finds one by probing
        // PATH for `python3.13`. That probe is silent until it fails, and it
        // failed on a release tag with a single line an hour into the run.
        // CI sets this to the interpreter it installed; unset locally, where
        // the probe has always worked.
        System.getenv("SPEKTR_BUILD_PYTHON")?.takeIf { it.isNotBlank() }
            ?.let { buildPython(it) }
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
