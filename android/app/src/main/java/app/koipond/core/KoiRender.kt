package app.koipond.core

import app.koipond.core.KoiSpec.BOLD
import app.koipond.core.KoiSpec.CELLS
import app.koipond.core.KoiSpec.EXTRA
import app.koipond.core.KoiSpec.NF
import app.koipond.core.KoiSpec.S
import app.koipond.core.KoiSpec.THIN
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sin

/** Port of koi.py `_render`: a top-down koi pond whose koi carry the codeword in their scale hatches. */
object KoiRender {

    /** One koi swimming head-first around the ring, in body coordinates (u: nose 0 -> tail base 1, v: -1..1). */
    class Koi(private val raw: Pts, private val us: DoubleArray, private val tg: Pts, private val nr: Pts,
              private val hw: Double, private val c: Double, private val d: Double, private val p: Double, private val k: Double) {
        fun half(u: Double) = KoiSpec.profile(u, hw, c, d, p, k)
        private fun at(a: Pts, u: Double) = P(interp(u, us, a.x), interp(u, us, a.y))
        fun tan(u: Double) = at(tg, u)
        fun nrm(u: Double) = at(nr, u)
        fun xy(u: Double, v: Double) = at(raw, u) + at(nr, u) * (v * half(u))
        fun xy(u: DoubleArray, v: DoubleArray) = Pts.of(u.size) { xy(u[it], v[it]) }
        val poly: Pts = run {
            val uu = linspace(0.0, 1.0, 300)
            xy(uu, DoubleArray(300) { -1.0 }) + xy(uu.reversedArray(), DoubleArray(300) { 1.0 })
        }
    }

    private fun koi(rng: Rng, ctr: P, r: Double, phi: Double, spin: Int): Koi {
        val l = rng.uniform(KoiSpec.L_RNG.first, KoiSpec.L_RNG.second)
        val hw = rng.uniform(KoiSpec.HW_RNG.first, KoiSpec.HW_RNG.second)
        val p = rng.uniform(KoiSpec.P_RNG.first, KoiSpec.P_RNG.second)
        val c = rng.uniform(KoiSpec.C_RNG.first, KoiSpec.C_RNG.second)
        val d = rng.uniform(KoiSpec.D_RNG.first, KoiSpec.D_RNG.second)
        val k = rng.uniform(KoiSpec.K_RNG.first, KoiSpec.K_RNG.second)
        val s = linspace(0.0, l, 600)
        val amp = rng.uniform(0.0, 12.0)
        val lam = rng.uniform(.8, 1.3) * l
        val ph = rng.uniform(0.0, 2 * PI)
        val raw = Pts.of(600) { i ->
            val al = phi + spin * (l / 2 - s[i]) / r
            val rr = r + amp * sin(2 * PI * s[i] / lam + ph)
            P(ctr.x + rr * cos(al), ctr.y + rr * sin(al))
        }
        val arc = cumlen(raw)
        val us = DoubleArray(600) { arc[it] / arc.last() }
        val g = gradient(raw)
        val tg = Pts.of(600) { P(g.x[it], g.y[it]).unit() }
        val nr = Pts.of(600) { P(-tg.y[it], tg.x[it]) }
        return Koi(raw, us, tg, nr, hw, c, d, p, k)
    }

    private fun hsv(h0: Double, s: Double, v: Double): DoubleArray {
        val h = ((h0 % 1) + 1) % 1
        val i = (h * 6).toInt()
        val f = h * 6 - i
        val pp = v * (1 - s)
        val q = v * (1 - s * f)
        val t = v * (1 - s * (1 - f))
        val rgb = when (i % 6) {
            0 -> doubleArrayOf(v, t, pp); 1 -> doubleArrayOf(q, v, pp); 2 -> doubleArrayOf(pp, v, t)
            3 -> doubleArrayOf(pp, q, v); 4 -> doubleArrayOf(t, pp, v); else -> doubleArrayOf(v, pp, q)
        }
        return rgb
    }

    fun bitsOf(bytes: ByteArray) = IntArray(bytes.size * 8) { (bytes[it / 8].toInt() shr (7 - it % 8)) and 1 }

    /** codeword: the outer RS codeword (NF * CHUNK bytes). seed: the pond's look. */
    fun render(codeword: ByteArray, seed: Long): Rgb {
        val rng = Rng(seed)
        val words = (0 until NF).map { f ->
            val chunk = ByteArray(KoiSpec.CHUNK) { codeword[f + it * NF] }
            bitsOf(Rs.encode(byteArrayOf((0xA0 or f).toByte()) + chunk, KoiSpec.INNER_NSYM))
        }

        // --- layout: koi swim a ring head-first, in shuffled order ----------------------------
        val ctr = P(S / 2.0, S / 2.0)
        val r0 = rng.uniform(870.0, 910.0)
        val phase = rng.uniform(0.0, 2 * PI)
        val spin = if (rng.uniform() < .5) -1 else 1
        val slots = rng.permutation(NF)
        val kois = Prof.sec("r.layout") { (0 until NF).map { f ->
            koi(rng, ctr, r0 + rng.uniform(-25.0, 25.0), phase + 2 * PI * slots[f] / NF + rng.uniform(-.05, .05), spin)
        } }

        val ink = Ink(S, S)
        fun line(p: Pts, w: Double, a: Int = 255) = ink.line(p, w, a / 255.0)
        fun wline(p: Pts, w: Double, amp: Double = .5, wl: Double = 40.0, a: Int = 255) = line(wobble(p, amp, wl, rng), w, a)
        fun inside(seg: Pts, m: Double = 25.0): Pts {
            val bad = (0 until seg.n).firstOrNull { seg.x[it] <= m || seg.x[it] >= S - m || seg.y[it] <= m || seg.y[it] >= S - m }
            return if (bad == null) seg else seg.sub(0, max(bad, 2).coerceAtMost(seg.n))
        }

        /** Fin: thin wobbly rays + wavy membrane edge; returns the membrane polygon. */
        fun rays(base: Pts, d: List<P>, lens: DoubleArray, gap: Double, sweep: P?): Pts {
            val q = linspace(0.0, 1.0, 10)
            val rs = (0 until base.n).map { i ->
                inside(Pts.of(10) { j ->
                    val sw = sweep?.times(.12 * lens[i] * q[j] * q[j]) ?: P(0.0, 0.0)
                    base[i] + d[i] * (gap + q[j] * (lens[i] - gap)) + sw
                })
            }
            val tips = Pts.of(rs.size) { rs[it][rs[it].n - 1] }
            for (i in rs.indices) {
                wline(rs[i], 1.05, .5, 35.0)
                if (gap > 0) line(Pts.of(base[i], rs[i][0]), .7, 110)  // faint root: never reads bold
            }
            val margin = wobble(smoothMargin(tips, base, .1), 2.2, 22.0, rng)
            line(margin, THIN)
            for (i in 0 until base.n - 1) {
                val p0 = (base[i] + base[i + 1]) * .5
                val p1 = (tips[i] + tips[i + 1]) * .5
                line(Pts.of(p0 + (p1 - p0) * .3, p0 + (p1 - p0) * rng.uniform(.5, .75)), .5, 200)
            }
            return base + margin.reversed()
        }

        val fins = ArrayList<Pts>()
        Prof.sec("r.koi-strokes") {
        val arcs = DoubleArray(16) { Math.toRadians(-90.0 + 12.0 * it) }
        val hatch = DoubleArray(6) { Math.toRadians(40.0 + 8.0 * it) }
        for ((f, kd) in kois.withIndex()) {
            val fly = rng.uniform() < .35  // butterfly koi: long, flowing fins
            val allBits = words[f] + IntArray(CELLS.size - words[f].size + EXTRA.size) { if (rng.uniform() < .5) 1 else 0 }
            for ((ci, cell) in (CELLS + EXTRA).withIndex()) {
                val u0 = cell.u - .55 * cell.du
                line(Pts.of(16) { kd.xy(u0 + cell.du * cos(arcs[it]), cell.v + .5 * cell.dv * sin(arcs[it])) }, 1.5)
                val side = if (allBits[ci] == 1) -1 else 1
                for (rr in doubleArrayOf(.45, .6, .75))
                    line(Pts.of(6) { kd.xy(u0 + rr * cell.du * cos(hatch[it]), cell.v + side * rr * .5 * cell.dv * sin(hatch[it])) }, 1.2)
            }
            // bold outline, closed across the tail base: the decoder's silhouette
            line(wobble(kd.poly, .5, 70.0, rng), BOLD)
            line(wobble(kd.xy(DoubleArray(20) { 1.0 }, linspace(-1.0, 1.0, 20)), .3, 40.0, rng), BOLD * .85)
            for (sd in intArrayOf(-1, 1)) {
                val e = kd.xy(.085, .7 * sd)
                ink.ring(e, 7.5, BOLD * .7)
                ink.disc(e, 3.5)
                ink.erase(e + kd.tan(.085) * 1.2, 1.1)
                ink.ring(kd.xy(.035, .35 * sd), 2.6, .9)
                val vv = linspace(.45, 1.0, 12).map { it * sd }.toDoubleArray()
                wline(kd.xy(DoubleArray(12) { .2 - .04 * ((1 - abs(vv[it])) / .55).pow(2) }, vv), THIN, .4, 30.0)
                for ((bu, bv, ln) in listOf(Triple(.012, .25, rng.uniform(26.0, 40.0)), Triple(.045, .45, rng.uniform(18.0, 28.0)))) {
                    val b0 = kd.xy(bu, bv * sd)
                    val dd = kd.tan(bu) * -cos(.9) + kd.nrm(bu) * (sd * sin(.9))
                    line(Pts.of(10) { val q = it / 9.0; b0 + dd * (q * ln) + kd.nrm(bu) * (6 * q * q * sd) }, 1.0)
                }
                repeat(4) {
                    val p = kd.xy(rng.uniform(.04, .17), rng.uniform(.1, .55) * sd)
                    line(Pts.of(p, p + kd.tan(.1) * rng.uniform(4.0, 8.0)), .9, 220)
                }
                val pec = if (fly) 150.0 to 210.0 else 85.0 to 135.0
                val pel = if (fly) 80.0 to 120.0 else 40.0 to 70.0
                for ((ua, ub, n, lr, tilt) in listOf(FinSpec(.17, .26, if (fly) 11 else 9, pec, .75), FinSpec(.45, .51, 6, pel, .9))) {
                    val uS = linspace(ua, ub, n)
                    val tl = linspace(.75, 1.15, n).map { it * tilt }
                    val d = (0 until n).map { (kd.nrm(uS[it]) * (sd * cos(tl[it])) + kd.tan(uS[it]) * sin(tl[it])).unit() }
                    val base = rng.uniform(lr.first, lr.second)
                    val sp = linspace(.15, 1.0, n)
                    val lens = DoubleArray(n) { base * (.7 + .3 * sin(PI * sp[it])) }
                    fins += rays(kd.xy(uS, DoubleArray(n) { sd.toDouble() }), d, lens, 6.0, kd.tan(uS.last()))
                }
            }
            // tail: broad, softly forked; rays start off the bold base so they don't blur into it
            val nT = rng.int(13, 19)
            val vs = linspace(-1.0, 1.0, nT)
            val spread = rng.uniform(.45, .7)
            val fork = rng.uniform(.1, .4)
            val tlen = if (fly) rng.uniform(165.0, 195.0) else rng.uniform(120.0, 170.0)
            val d = vs.map { kd.tan(1.0) * cos(spread * it) + kd.nrm(1.0) * sin(spread * it) }
            fins += rays(kd.xy(DoubleArray(nT) { 1.0 }, vs), d, DoubleArray(nT) { tlen * (1 - fork * (1 - abs(vs[it]).pow(1.2))) }, 14.0, null)
        }

        }
        // --- washes (1x): water, koi bodies, patches, translucent fins ------------------------
        val n = S * S
        val img = Wash.paper(n)
        val low = Prof.sec("r.water-noise") { upscaleCubic(DoubleArray(144) { rng.normal(.08) }, 12, S, S) }
        val water = hsv(rng.uniform(.42, .52), rng.uniform(.12, .2), .94)
        Prof.sec("r.water") { Wash.water(img, low.a, water) }
        val bodies = Mask(S, S)
        Prof.sec("r.bodies") { for (kd in kois) bodies.fillPoly(kd.poly) }
        val edge = Prof.sec("r.body-edt") { edt(bodies.a, S, S) }
        Wash.bodies(img, bodies.a, edge.a)  // rounded-body shading, wide and gentle
        val keep = Prof.sec("r.keep") { rectFilter(bodies.toFloat(), 5, false) }
        val layers = mapOf("red" to Mask(S, S), "gold" to Mask(S, S), "yellow" to Mask(S, S), "blue" to Mask(S, S), "cha" to Mask(S, S))
        val tt90 = linspace(0.0, 2 * PI, 90)
        fun patch(layer: String, kd: Koi, uc: Double, vc: Double, ru: Double, rv: Double) {
            val a1 = rng.uniform(0.0, 6.0)
            val a2 = rng.uniform(0.0, 6.0)
            layers.getValue(layer).fillPoly(Pts.of(90) {
                val t = tt90[it]
                val wob = 1 + .1 * sin(3 * t + a1) + .06 * sin(7 * t + a2)
                kd.xy(uc + ru * wob * cos(t), vc + rv * wob * sin(t))
            })
        }
        for (kd in kois) {
            val kind = rng.choice(listOf("kohaku", "kohaku", "tancho", "maruten", "ogon", "yamabuki", "asagi", "chagoi", "shiro"))
            when (kind) {
                "ogon" -> layers.getValue("gold").fillPoly(kd.poly, .85)
                "yamabuki" -> layers.getValue("yellow").fillPoly(kd.poly, .85)
                "chagoi" -> layers.getValue("cha").fillPoly(kd.poly, .85)
            }
            if (kind == "kohaku") {  // stepped red blocks across the back, head to tail
                var u = rng.uniform(.04, .12)
                while (u < .85) {
                    val ru = rng.uniform(.05, .11)
                    patch("red", kd, u + ru, rng.uniform(-.25, .25), ru, rng.uniform(.8, 1.4))
                    u += 2 * ru + rng.uniform(.04, .12)
                }
            }
            if (kind == "tancho" || kind == "maruten") patch("red", kd, .09, 0.0, .045, .5)
            if (kind == "maruten") repeat(rng.int(1, 3)) {
                patch("red", kd, rng.uniform(.35, .8), rng.uniform(-.2, .2), rng.uniform(.05, .1), rng.uniform(.7, 1.2))
            }
            if (kind == "asagi") {
                patch("blue", kd, .55, 0.0, .45, .55)
                for (sd in intArrayOf(-1, 1)) patch("red", kd, .5, 1.05 * sd, .4, .3)
            }
        }
        val cols = mapOf(
            "red" to hsv(rng.uniform(.01, .05), rng.uniform(.6, .8), .96), "gold" to hsv(rng.uniform(.1, .13), rng.uniform(.45, .6), .97),
            "yellow" to hsv(rng.uniform(.13, .16), rng.uniform(.35, .5), .98), "blue" to hsv(rng.uniform(.55, .6), rng.uniform(.15, .25), .92),
            "cha" to hsv(rng.uniform(.06, .09), rng.uniform(.3, .4), .9))
        fun tint(m: Mask, col: DoubleArray, sigma: Double, gain: Float, clip: FImg? = null) {
            val a = Wash.soft(m, sigma) ?: return
            Wash.tint(img, a.a, col, gain, clip?.a)
        }
        Prof.sec("r.patches") { for ((name, m) in layers) tint(m, cols.getValue(name), 2.5, 1f, keep) }
        val finm = Mask(S, S)
        for (poly in fins) finm.fillPoly(poly, .55)
        Prof.sec("r.fin-wash") { Wash.soft(finm, 3.0)?.let { Wash.lighten(img, it.a, bodies.a) } }

        // --- plants, placed in the water the koi leave free ---------------------------------
        val q = 4
        val fs = S / q
        val tFoot = System.nanoTime()
        var fp = Mask(fs, fs)
        for (poly in kois.map { it.poly } + fins) fp.fillPoly(poly, 1.0, 1.0 / q)
        run {  // dilate the footprint by ~30 px
            val inv = ByteArray(fs * fs) { if (fp.a[it].toInt() == 0) 1 else 0 }
            val dd = edt(inv, fs, fs)
            fp = Mask(fs, fs, ByteArray(fs * fs) { if (dd.a[it] <= 30.0 / q) -1 else 0 })
        }
        fun free(): FImg {
            val inv = ByteArray(fs * fs) { if (fp.a[it].toInt() == 0) 1 else 0 }
            val e = edt(inv, fs, fs)
            for (i in e.a.indices) e.a[i] *= q
            return e
        }
        fun clear(pts: Pts, fpAt: ByteArray) = (0 until pts.n).all {
            val x = (pts.x[it] / q).toInt().coerceIn(0, fs - 1)
            val y = (pts.y[it] / q).toInt().coerceIn(0, fs - 1)
            fpAt[y * fs + x].toInt() == 0
        }
        fun spot(r: Double): P? {
            val fr = free()
            repeat(300) {
                val p = P(rng.uniform(r + 30, S - r - 30), rng.uniform(r + 30, S - r - 30))
                if (fr[(p.x / q).toInt(), (p.y / q).toInt()] > r) {
                    fp.fillCircle(p.x / q, p.y / q, (r / q).toInt() + 2.0)
                    return p
                }
            }
            return null
        }
        Prof.ms["r.footprint"] = (System.nanoTime() - tFoot) / 1_000_000
        val green = Mask(S, S)
        val pink = Mask(S, S)
        val leaf = Mask(S, S)
        val stone = Mask(S, S)

        Prof.sec("r.stones") {
        // stone rim: rounded stones along the pond edge, wherever no koi or fin reaches
        val per = ArrayList<P>()
        for (t in linspace(0.0, S.toDouble(), 60)) per += P(t, 0.0)
        for (t in linspace(0.0, S.toDouble(), 60)) per += P(S.toDouble(), t)
        for (t in linspace(S.toDouble(), 0.0, 60)) per += P(t, S.toDouble())
        for (t in linspace(S.toDouble(), 0.0, 60)) per += P(0.0, t)
        val rim = Pts.of(per.size) { per[it] + (ctr - per[it]).unit() * 80.0 }
        val lens = cumlen(rim)
        var walk = 0.0
        while (walk < lens.last()) {
            val r = rng.uniform(32.0, 62.0)
            val c = P(interp(walk, lens, rim.x), interp(walk, lens, rim.y)) + P(rng.normal(10.0), rng.normal(10.0))
            walk += 2 * r * rng.uniform(.8, 1.0)
            val a1 = rng.uniform(0.0, 6.0)
            val a2 = rng.uniform(0.0, 6.0)
            val poly = Pts.of(40) {
                val t = 2 * PI * it / 40
                polar(c, r * (1 + .12 * sin(2 * t + a1) + .07 * sin(5 * t + a2)), t)
            }
            if ((poly.x + poly.y).any { it < 12 || it > S - 12 } || !clear(poly, fp.a)) continue
            stone.fillPoly(poly, rng.uniform(.7, 1.0))
            fp.fillPoly(poly, 1.0, 1.0 / q)
            wline(poly.closed(), THIN, .6, 30.0)
            val h0 = 3.6 + rng.uniform(-.3, .3)
            line(Pts.of(10) { polar(c, .72 * r, h0 + it / 9.0) }, .6, 120)
            repeat(rng.int(0, 3)) {
                var p0 = c + P(rng.normal(r * .3), rng.normal(r * .3))
                val crack = ArrayList<P>()
                repeat(5) { p0 = p0 + P(rng.normal(4.0), rng.normal(4.0)); crack += p0 }
                line(Pts.of(5) { crack[it] }, .7, 150)
            }
            repeat(rng.int(2, 6)) {
                val p0 = poly[rng.int(0, poly.n)]
                repeat(3) { line(Pts.of(p0, p0 + P(rng.normal(5.0), rng.normal(5.0))), .6, 170) }
            }
        }

        }
        Prof.sec("r.reeds") {
        // reed clumps in the corners: tapered blades with a midrib
        for (corner in listOf(P(0.0, 0.0), P(S.toDouble(), 0.0), P(0.0, S.toDouble()), P(S.toDouble(), S.toDouble()))) {
            val aim = atan2(ctr.y - corner.y, ctr.x - corner.x)
            val fr = fp.a.copyOf()  // a snapshot: blades of one corner may overlap each other
            repeat(rng.int(2, 4)) {
                val rr = rng.uniform(60.0, 200.0)
                val root = corner + P(cos(aim + rng.uniform(-.6, .6)), sin(aim + rng.uniform(-.6, .6))) * rr
                repeat(rng.int(6, 12)) {
                    val ang = aim + rng.uniform(-.7, .7)
                    val ln = rng.uniform(160.0, 380.0)
                    val wd = rng.uniform(5.0, 10.0)
                    val dv = P(cos(ang), sin(ang))
                    val nv = P(-sin(ang), cos(ang))
                    val jit = P(rng.normal(8.0), rng.normal(8.0))
                    val bend = rng.uniform(-.5, .5)
                    val tt = linspace(0.0, 1.0, 24)
                    val mid = Pts.of(24) { root + jit + dv * (ln * tt[it]) + nv * (bend * ln * tt[it] * tt[it]) }
                    val g = gradient(mid)
                    val hw = DoubleArray(24) { wd * sin(PI * (tt[it] * .95 + .05).coerceIn(0.0, 1.0)).pow(.6) }
                    val nn = Pts.of(24) { P(-g.y[it], g.x[it]).unit() }
                    val bl = Pts.of(24) { mid[it] + nn[it] * hw[it] } + Pts.of(24) { mid[23 - it] - nn[23 - it] * hw[23 - it] }
                    if (!clear(bl, fr)) return@repeat
                    leaf.fillPoly(bl, rng.uniform(.6, 1.0))
                    fp.fillPoly(bl, 1.0, 1.0 / q)
                    line(bl.closed(), .9)
                    line(mid.sub(2, 21), .5, 150)
                }
            }
        }

        }
        Prof.sec("r.pads") {
        // lily pads, some with lotus, plus loose ripples
        repeat(rng.int(8, 14)) {
            val r = rng.uniform(55.0, 135.0)
            val p = spot(r + 15) ?: return@repeat
            val a0 = rng.uniform(0.0, 2 * PI)
            val gapw = Math.toRadians(rng.uniform(18.0, 30.0))
            val t = linspace(a0 + gapw / 2, a0 + 2 * PI - gapw / 2, 90)
            val rimP = Pts.of(90) { polar(p, r, t[it]) }
            green.fillPoly(Pts.of(p) + rimP)
            wline(Pts.of(p) + rimP + Pts.of(p), THIN, .8, 45.0)
            for (tv in linspace(t[0] + .25, t[89] - .25, rng.int(8, 14))) {
                val dd = P(cos(tv), sin(tv))
                line(Pts.of(8) { val qq = .12 + (.9 - .12) * it / 7; p + dd * (qq * r) + P(-dd.y, dd.x) * (4 * qq * qq) }, .8, 200)
            }
            val nr = rng.int(1, 3)
            for (f in DoubleArray(nr) { rng.uniform(1.12, 1.45) }.sorted()) {
                val ta = rng.uniform(0.0, 2 * PI)
                val span = rng.uniform(2.5, 5.0)
                line(Pts.of(80) { polar(p, f * r, ta + span * it / 79) }, .9, 120)
            }
            if (rng.uniform() < .45) {  // a lotus on the pad: three rings of petals, painted over it
                val big = min(r * .75, rng.uniform(55.0, 75.0))
                for ((k, nrr) in listOf(9 to big, 7 to big * .68, 5 to big * .42).withIndex()) {
                    val (np, rr) = nrr
                    for (i in 0 until np) {
                        val ang = 2 * PI * (i + .5 * k) / np + rng.uniform(-.1, .1)
                        val dd = P(cos(ang), sin(ang))
                        val pp = P(-sin(ang), cos(ang))
                        val tt = linspace(0.0, 1.0, 14)
                        val side = DoubleArray(14) { sin(PI * tt[it]).pow(.8) * rr * .34 }
                        val petal = Pts.of(14) { p + dd * (tt[it] * rr) + pp * side[it] } +
                            Pts.of(14) { p + dd * (tt[13 - it] * rr) - pp * side[13 - it] }
                        pink.fillPoly(petal, .6 + .2 * k)
                        line(petal, 1.0)
                        line(Pts.of(9) { p + dd * (tt[it + 2] * rr) }, .45, 130)
                    }
                }
                ink.erase(p, 6.0)
                ink.ring(p, 6.0, .9)
                repeat(8) { ink.disc(p + P(rng.normal(2.5), rng.normal(2.5)), 1.0) }
            }
        }
        repeat(rng.int(3, 7)) {
            val p = spot(70.0) ?: return@repeat
            val top = rng.uniform(2.0, 3.2)
            for (f in linspace(1.0, top, rng.int(2, 4))) {
                val ta = rng.uniform(0.0, 2 * PI)
                val span = rng.uniform(3.0, 6.0)
                line(Pts.of(90) { val a = ta + span * it / 89; P(p.x + 20 * f * cos(a), p.y + 20 * f * .8 * sin(a)) }, .9, 110)
            }
        }
        }
        Wash.cut(green.a, pink.a)
        Prof.sec("r.plant-tints") {
        tint(stone, hsv(rng.uniform(.06, .1), rng.uniform(.06, .12), .9), 1.5, .92f)
        tint(leaf, hsv(rng.uniform(.2, .26), rng.uniform(.3, .42), .86), 1.2, .92f)
        tint(green, hsv(rng.uniform(.24, .32), rng.uniform(.25, .38), .88), 2.0, .92f)
        tint(pink, hsv(rng.uniform(.93, .99), rng.uniform(.38, .5), .98), 1.5, .92f)
        }

        // --- ink over everything ------------------------------------------------------------
        val out = Rgb(S, S)
        Prof.sec("r.composite") { Wash.composite(img, ink.img.a, out.px) }
        return out
    }

    private data class FinSpec(val ua: Double, val ub: Double, val n: Int, val lr: Pair<Double, Double>, val tilt: Double)
}
