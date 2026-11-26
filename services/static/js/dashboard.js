(() => {
    const configEl = document.getElementById("dashboard-config");
    const config = configEl ? JSON.parse(configEl.textContent) : {};

    const video = document.getElementById("videoPlayer");
    const canvas = document.getElementById("bboxOverlay");
    const ctx = canvas.getContext("2d");
    const bboxToggle = document.getElementById("bboxToggle");

    const statsData = config.stats || {};
    const videoMeta = config.videoMeta || {};

    let activeStart = config.start ?? null;
    let activeEnd = config.end ?? null;
    let activeTrackLogs = null;
    let activeTrackMeta = null;
    let segmentHandler = null;
    let rafId = null;
    const trackCache = new Map();

    function updateSegmentDisplay(startVal, endVal, label) {
        const startEl = document.getElementById("startDisplay");
        const endEl = document.getElementById("endDisplay");
        const durEl = document.getElementById("durationDisplay");
        const badgeEl = document.getElementById("activeTrackBadge");

        if (startVal !== null && endVal !== null) {
            startEl.textContent = `${Number(startVal).toFixed(2)}s`;
            endEl.textContent = `${Number(endVal).toFixed(2)}s`;
            durEl.textContent = `${Math.max(0, Number(endVal) - Number(startVal)).toFixed(2)}s`;
        } else {
            startEl.textContent = "-";
            endEl.textContent = "-";
            durEl.textContent = "-";
        }
        if (label) {
            badgeEl.textContent = label;
        }
    }

    function runQuickQuery(text) {
        document.getElementById("queryInput").value = text;
        document.getElementById("searchForm").submit();
    }

    function filterByField(fieldId, value) {
        document.getElementById(fieldId).value = value;
        document.getElementById("searchForm").submit();
    }

    function updatePlaybackRate() {
        video.playbackRate = Number(document.getElementById("speedSelect").value || 1);
    }

    function syncCanvasSize() {
        const rect = video.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
            const targetW = Math.round(rect.width);
            const targetH = Math.round(rect.height);
            if (canvas.width !== targetW || canvas.height !== targetH) {
                canvas.width = targetW;
                canvas.height = targetH;
            }
        }
    }

    function clearOverlay() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    function findClosestFrameLog(logs, currentTimeSec) {
        if (!logs || logs.length === 0) return null;
        let lo = 0;
        let hi = logs.length - 1;
        while (lo < hi) {
            const mid = Math.floor((lo + hi) / 2);
            if (logs[mid].time_sec < currentTimeSec) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let best = logs[lo];
        if (
            lo > 0 &&
            Math.abs(logs[lo - 1].time_sec - currentTimeSec) <
                Math.abs(best.time_sec - currentTimeSec)
        ) {
            best = logs[lo - 1];
        }
        if (Math.abs(best.time_sec - currentTimeSec) > 0.35) {
            return null;
        }
        return best;
    }

    function drawOverlay() {
        syncCanvasSize();
        clearOverlay();

        if (!bboxToggle.checked || !activeTrackLogs || !activeTrackMeta) {
            return;
        }

        const entry = findClosestFrameLog(activeTrackLogs, video.currentTime);
        if (!entry) return;

        const srcW = activeTrackMeta.video_width || videoMeta.width || 1920;
        const srcH = activeTrackMeta.video_height || videoMeta.height || 1080;
        const scaleX = canvas.width / srcW;
        const scaleY = canvas.height / srcH;

        const { x1, y1, x2, y2 } = entry.bbox;
        const rx = x1 * scaleX;
        const ry = y1 * scaleY;
        const rw = (x2 - x1) * scaleX;
        const rh = (y2 - y1) * scaleY;

        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = 2.5;
        ctx.strokeRect(rx, ry, rw, rh);

        const label = `#${activeTrackMeta.id} ${activeTrackMeta.name}`;
        ctx.font = "600 13px Inter, sans-serif";
        const textW = ctx.measureText(label).width;
        const labelY = Math.max(20, ry);

        ctx.fillStyle = "rgba(16, 185, 129, 0.9)";
        ctx.fillRect(rx, labelY - 20, textW + 12, 20);
        ctx.fillStyle = "#062016";
        ctx.fillText(label, rx + 6, labelY - 5);
    }

    function startOverlayLoop() {
        if (rafId !== null) cancelAnimationFrame(rafId);
        const tick = () => {
            drawOverlay();
            if (!video.paused && !video.ended) {
                rafId = requestAnimationFrame(tick);
            } else {
                rafId = null;
            }
        };
        rafId = requestAnimationFrame(tick);
    }

    function clearSegmentHandler() {
        if (segmentHandler) {
            video.removeEventListener("timeupdate", segmentHandler);
            segmentHandler = null;
        }
    }

    function seekAndPlay(startSec, endSec) {
        if (startSec === null || endSec === null) return;
        video.pause();
        clearSegmentHandler();

        const runPlay = async () => {
            const safeStart = Math.max(
                0,
                Math.min(startSec, Math.max(0, (video.duration || startSec) - 0.05))
            );
            video.currentTime = safeStart;
            updatePlaybackRate();

            segmentHandler = () => {
                drawOverlay();
                if (video.currentTime >= endSec) {
                    video.pause();
                    video.currentTime = Math.min(endSec, video.duration || endSec);
                    clearSegmentHandler();
                }
            };
            video.addEventListener("timeupdate", segmentHandler);

            try {
                await video.play();
            } catch (err) {
                console.error("Video playback failed:", err);
            }
        };

        if (video.readyState >= 1) {
            runPlay();
        } else {
            video.addEventListener("loadedmetadata", runPlay, { once: true });
        }
    }

    function playActiveSegment() {
        seekAndPlay(activeStart, activeEnd);
    }

    function playFullVideo() {
        clearSegmentHandler();
        activeTrackLogs = null;
        activeTrackMeta = null;
        clearOverlay();
        document
            .querySelectorAll(".result-card")
            .forEach((c) => c.classList.remove("active-track"));
        updateSegmentDisplay(0, video.duration || videoMeta.duration, "Full Video");
        video.currentTime = 0;
        updatePlaybackRate();
        video.play().catch((err) => console.error("Playback failed:", err));
    }

    function stopSegment() {
        video.pause();
        clearSegmentHandler();
    }

    async function fetchTrackDetail(trackId) {
        if (trackCache.has(trackId)) {
            return trackCache.get(trackId);
        }
        const res = await fetch(`/api/tracks/${trackId}`);
        if (!res.ok) {
            throw new Error(`Failed to load track #${trackId}`);
        }
        const data = await res.json();
        trackCache.set(trackId, data);
        return data;
    }

    async function playTrack(trackId) {
        try {
            const detail = await fetchTrackDetail(trackId);
            activeTrackMeta = detail;
            activeTrackLogs = detail.frame_logs || [];
            activeStart = detail.offset_start;
            activeEnd = Math.max(detail.offset_start + 0.5, detail.offset_end);

            document
                .querySelectorAll(".result-card")
                .forEach((c) => c.classList.remove("active-track"));
            const card = document.getElementById(`card-${trackId}`);
            if (card) card.classList.add("active-track");

            updateSegmentDisplay(
                activeStart,
                activeEnd,
                `Track #${detail.id} (${detail.name})`
            );
            seekAndPlay(activeStart, activeEnd);
        } catch (err) {
            console.error(err);
            alert(err.message);
        }
    }

    async function inspectTrack(trackId) {
        try {
            const detail = await fetchTrackDetail(trackId);
            document.getElementById("modalTitle").textContent =
                `Track #${detail.id} — ${detail.name}`;
            document.getElementById("modalImage").src = detail.image_url || "";
            document.getElementById("modalMeta").innerHTML = `
                <div><strong>Class:</strong> ${detail.name}</div>
                <div><strong>Video Offset:</strong> ${detail.offset_start}s – ${detail.offset_end}s (${detail.duration}s)</div>
                <div><strong>Start Timestamp:</strong> ${detail.start}</div>
                <div><strong>End Timestamp:</strong> ${detail.end}</div>
                <div><strong>Confidence Range:</strong> ${detail.min_conf} – ${detail.max_conf}</div>
                <div><strong>Logged Frames:</strong> ${detail.frame_count}</div>
                <div><strong>Image File:</strong> ${detail.filename}</div>
            `;
            document.getElementById("modalPlayBtn").onclick = () => {
                closeModal();
                playTrack(trackId);
            };
            document.getElementById("trackModal").classList.add("open");
        } catch (err) {
            alert(err.message);
        }
    }

    function closeModal(event) {
        if (!event || event.target.id === "trackModal") {
            document.getElementById("trackModal").classList.remove("open");
        }
    }

    function jumpToTime(sec) {
        clearSegmentHandler();
        const runJump = () => {
            video.currentTime = Math.max(0, Math.min(sec, video.duration || sec));
            video.play().catch(() => {});
        };
        if (video.readyState >= 1) {
            runJump();
        } else {
            video.addEventListener("loadedmetadata", runJump, { once: true });
        }
    }

    function renderAnalytics() {
        const timelineEl = document.getElementById("timelineChart");
        const buckets = statsData.timeline_buckets || [];
        const maxBucket = Math.max(1, ...buckets.map((b) => b.count));

        timelineEl.innerHTML = "";
        buckets.forEach((b) => {
            const h = Math.max(6, Math.round((b.count / maxBucket) * 100));
            const wrap = document.createElement("div");
            wrap.className = "t-bar-wrap";
            wrap.title = `${b.start_sec}s–${b.end_sec}s: ${b.count} active tracks`;
            wrap.addEventListener("click", () => jumpToTime(b.start_sec));

            const bar = document.createElement("div");
            bar.className = "t-bar";
            bar.style.height = `${h}%`;

            const label = document.createElement("div");
            label.className = "t-bar-label";
            label.textContent = b.label;

            wrap.append(bar, label);
            timelineEl.appendChild(wrap);
        });

        const classEl = document.getElementById("classChart");
        const classEntries = Object.entries(statsData.class_counts || {}).slice(0, 8);
        const maxClass = Math.max(1, ...classEntries.map(([, v]) => v));

        classEl.innerHTML = "";
        classEntries.forEach(([name, count]) => {
            const pct = Math.max(3, Math.round((count / maxClass) * 100));
            const item = document.createElement("div");
            item.className = "dist-item";
            item.addEventListener("click", () => filterByField("classSelect", name));

            const nameSpan = document.createElement("span");
            nameSpan.textContent = name;

            const track = document.createElement("div");
            track.className = "dist-track";
            const fill = document.createElement("div");
            fill.className = "dist-fill";
            fill.style.width = `${pct}%`;
            track.appendChild(fill);

            const countSpan = document.createElement("span");
            countSpan.className = "dist-count";
            countSpan.textContent = count;

            item.append(nameSpan, track, countSpan);
            classEl.appendChild(item);
        });
    }

    function bindEvents() {
        video.addEventListener("play", startOverlayLoop);
        video.addEventListener("seeked", drawOverlay);
        video.addEventListener("loadedmetadata", syncCanvasSize);
        window.addEventListener("resize", drawOverlay);
        bboxToggle.addEventListener("change", drawOverlay);

        document
            .getElementById("playSegmentBtn")
            ?.addEventListener("click", playActiveSegment);
        document
            .getElementById("playFullBtn")
            ?.addEventListener("click", playFullVideo);
        document
            .getElementById("stopSegmentBtn")
            ?.addEventListener("click", stopSegment);
        document
            .getElementById("speedSelect")
            ?.addEventListener("change", updatePlaybackRate);

        document.querySelectorAll("[data-quick-query]").forEach((btn) => {
            btn.addEventListener("click", () =>
                runQuickQuery(btn.getAttribute("data-quick-query"))
            );
        });

        document.querySelectorAll("[data-play-track]").forEach((btn) => {
            btn.addEventListener("click", () =>
                playTrack(Number(btn.getAttribute("data-play-track")))
            );
        });

        document.querySelectorAll("[data-inspect-track]").forEach((btn) => {
            btn.addEventListener("click", () =>
                inspectTrack(Number(btn.getAttribute("data-inspect-track")))
            );
        });

        document
            .getElementById("trackModal")
            ?.addEventListener("click", closeModal);
        document
            .getElementById("modalCloseBtn")
            ?.addEventListener("click", () => closeModal());
    }

    updateSegmentDisplay(activeStart, activeEnd, "Aggregate Segment");
    renderAnalytics();
    bindEvents();
})();