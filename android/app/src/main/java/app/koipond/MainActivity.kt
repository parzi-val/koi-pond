package app.koipond

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import app.koipond.core.StripReader
import app.koipond.ui.KoiApp
import app.koipond.ui.LabScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // ~13 MB model: load off the main thread; scans before it is ready fall back to the classical read
        Thread {
            val t = System.nanoTime()
            StripReader.load(assets.open("koinn.onnx").use { it.readBytes() })
            Log.i("koipond", "strip reader loaded in ${(System.nanoTime() - t) / 1_000_000} ms")
        }.start()
        enableEdgeToEdge()
        setContent { if (BuildConfig.LAB) LabScreen() else KoiApp() }
    }
}
