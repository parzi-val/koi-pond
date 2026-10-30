package app.koipond

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import app.koipond.core.StripReader
import app.koipond.ui.KoiApp

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // ~13 MB model: load off the main thread; scans before it is ready fall back to the classical read
        Thread { StripReader.load(assets.open("koinn.onnx").use { it.readBytes() }) }.start()
        enableEdgeToEdge()
        setContent { KoiApp() }
    }
}
