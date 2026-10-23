#include "models.h"
#include <QCryptographicHash>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QJsonDocument>
#include <queue>
#ifdef VS_DIRECTML
#include <dml_provider_factory.h>
#endif

namespace vs {
std::unique_ptr<Ort::Session> Models::load(const QString &file, bool gpu) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(gpu ? 1 : 2);
    options.SetInterOpNumThreads(1);
#ifdef VS_DIRECTML
    if (gpu) {
        options.DisableMemPattern();
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(options, 0));
    }
#else
    Q_UNUSED(gpu);
#endif
#ifdef Q_OS_WIN
    return std::make_unique<Ort::Session>(environment, reinterpret_cast<const wchar_t *>(file.utf16()),
                                          options);
#else
    return std::make_unique<Ort::Session>(environment, file.toUtf8().constData(), options);
#endif
}
Models::Models(const QString &folder, bool gpu) {
    QFile manifest(folder + "/native-models.json");
    if (!manifest.open(QIODevice::ReadOnly))
        throw std::runtime_error("Модели не найдены. Распакуйте всю папку из нового ZIP.");
    auto files = QJsonDocument::fromJson(manifest.readAll()).object()["files"].toObject();
    for (auto it = files.begin(); it != files.end(); ++it) {
        QFile f(folder + "/" + it.key());
        if (!f.open(QIODevice::ReadOnly))
            throw std::runtime_error(("Не найдена модель " + it.key()).toStdString());
        QCryptographicHash hash(QCryptographicHash::Sha256);
        while (!f.atEnd())
            hash.addData(f.read(1024 * 1024));
        if (QString::fromLatin1(hash.result().toHex()) != it.value().toString())
            throw std::runtime_error(("Повреждена модель " + it.key()).toStdString());
    }
#ifndef VS_DIRECTML
    gpu = false;
#endif
    try {
        body = load(folder + "/pose.onnx", gpu);
        ball = load(folder + "/ball.onnx", gpu);
        backend = gpu ? "GPU · DirectML" : "CPU";
        if (gpu) {
            QImage warm(640, 360, QImage::Format_RGB888);
            warm.fill(QColor(114, 114, 114));
            pose(warm);
            balls({warm});
            resetTemporal();
        }
    } catch (const Ort::Exception &e) {
        if (!gpu)
            throw;
        warning = "GPU недоступна; используется CPU: " + QString::fromUtf8(e.what());
        body = load(folder + "/pose.onnx", false);
        ball = load(folder + "/ball.onnx", false);
        backend = "CPU (GPU fallback)";
    }
    // Small ReID/OCR requests stay on CPU; avoid GPU submission overhead for tiny crops.
    reid = load(folder + "/reid.onnx", false);
    ocr = load(folder + "/jersey-rec.onnx", false);
    Ort::AllocatorWithDefaultOptions allocator;
    auto dictionary = ocr->GetModelMetadata().LookupCustomMetadataMapAllocated("character", allocator);
    if (dictionary) {
        characters = QString::fromUtf8(dictionary.get()).split('\n');
        characters.prepend("");
        characters.append(" ");
    }
    auto shape = body->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    poseSize = int(shape.at(2));
}
std::vector<float> Models::chw(const QImage &source, int w, int h, bool imagenet) {
    QImage image = source.scaled(w, h, Qt::IgnoreAspectRatio, Qt::SmoothTransformation)
                       .convertToFormat(QImage::Format_RGB888);
    std::vector<float> tensor(size_t(w) * h * 3);
    static constexpr float mean[] = {.485f, .456f, .406f}, sd[] = {.229f, .224f, .225f};
    for (int y = 0; y < h; ++y) {
        auto row = image.constScanLine(y);
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                float v = row[x * 3 + c] / 255.f;
                tensor[size_t(c) * w * h + y * w + x] = imagenet ? (v - mean[c]) / sd[c] : v;
            }
    }
    return tensor;
}
std::vector<Ort::Value> Models::infer(Ort::Session &s, std::vector<float> &data,
                                      const std::vector<int64_t> &shape) {
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor =
        Ort::Value::CreateTensor<float>(memory, data.data(), data.size(), shape.data(), shape.size());
    Ort::AllocatorWithDefaultOptions allocator;
    auto input = s.GetInputNameAllocated(0, allocator);
    auto output = s.GetOutputNameAllocated(0, allocator);
    const char *in[] = {input.get()};
    const char *out[] = {output.get()};
    return s.Run(Ort::RunOptions{nullptr}, in, &tensor, 1, out, 1);
}
std::vector<Person> Models::pose(const QImage &image) {
    QElapsedTimer timer;
    timer.start();
    double scale = std::min(double(poseSize) / image.width(), double(poseSize) / image.height());
    int w = std::round(image.width() * scale), h = std::round(image.height() * scale);
    int px = (poseSize - w) / 2, py = (poseSize - h) / 2;
    QImage padded(poseSize, poseSize, QImage::Format_RGB888);
    padded.fill(QColor(114, 114, 114));
    {
        QPainter painter(&padded);
        painter.drawImage(QPoint(px, py),
                          image.scaled(w, h, Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
    }
    auto input = chw(padded, poseSize, poseSize);
    auto outputs = infer(*body, input, {1, 3, poseSize, poseSize});
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3 || shape[1] != 56)
        throw std::runtime_error("Неподдерживаемый формат pose-модели");
    size_t n = size_t(shape[2]);
    const float *data = outputs[0].GetTensorData<float>();
    std::vector<Person> proposals;
    for (size_t i = 0; i < n; ++i) {
        float score = data[4 * n + i];
        if (score < .18)
            continue;
        Person p;
        p.score = score;
        double x = (data[i] - px) / scale, y = (data[n + i] - py) / scale, bw = data[2 * n + i] / scale,
               bh = data[3 * n + i] / scale;
        double x1 = std::clamp(x - bw / 2, 0., double(image.width() - 1)),
               y1 = std::clamp(y - bh / 2, 0., double(image.height() - 1));
        double x2 = std::clamp(x + bw / 2, 0., double(image.width())),
               y2 = std::clamp(y + bh / 2, 0., double(image.height()));
        p.box = {x1, y1, x2 - x1, y2 - y1};
        if (p.box.w < 5 || p.box.h < 15)
            continue;
        for (size_t k = 0; k < 17; ++k)
            p.joints[k] = {
                {(data[(5 + 3 * k) * n + i] - px) / scale, (data[(6 + 3 * k) * n + i] - py) / scale},
                data[(7 + 3 * k) * n + i]};
        proposals.push_back(p);
    }
    std::sort(proposals.begin(), proposals.end(),
              [](const auto &a, const auto &b) { return a.score > b.score; });
    if (proposals.size() > 250)
        proposals.resize(250);
    std::vector<Person> people;
    for (auto &p : proposals) {
        bool duplicate = false;
        for (auto &q : people)
            if (overlap(p.box, q.box) > .55) {
                duplicate = true;
                break;
            }
        if (duplicate)
            continue;
        QImage crop = image.copy(QRect(int(p.box.x), int(p.box.y), int(p.box.w), int(p.box.h)))
                          .scaled(24, 72)
                          .convertToFormat(QImage::Format_RGB888);
        Feature f(3 * 32, 0);
        for (int y = 0; y < crop.height(); ++y) {
            auto row = crop.constScanLine(y);
            for (int x = 0; x < crop.width(); ++x) {
                QColor c(row[x * 3], row[x * 3 + 1], row[x * 3 + 2]);
                int hue = std::max(0, c.hsvHue()), sat = c.hsvSaturation();
                int bin = (hue * 8 / 360) * 4 + sat * 4 / 256;
                f[(y / 24) * 32 + std::min(31, bin)]++;
            }
        }
        p.color = normalized(f);
        people.push_back(std::move(p));
    }
    poseMs = timer.nsecsElapsed() / 1e6;
    return people;
}
std::vector<std::vector<BallCandidate>> Models::balls(const std::vector<QImage> &images) {
    if (images.empty() || images.size() > 9)
        throw std::runtime_error("Некорректное окно мяча");
    QElapsedTimer timer;
    timer.start();
    std::vector<QImage> context;
    for (auto &im : history)
        context.push_back(im);
    for (auto &im : images)
        context.push_back(im);
    while (context.size() > 9)
        context.erase(context.begin());
    while (context.size() < 9)
        context.insert(context.begin(), context.front());
    constexpr int w = 512, h = 288;
    std::vector<float> input(9 * w * h);
    for (int i = 0; i < 9; ++i) {
        QImage im = context[i]
                        .scaled(w, h, Qt::IgnoreAspectRatio, Qt::SmoothTransformation)
                        .convertToFormat(QImage::Format_RGB888);
        for (int y = 0; y < h; ++y) {
            auto row = im.constScanLine(y);
            for (int x = 0; x < w; ++x)
                input[size_t(i) * w * h + y * w + x] =
                    float((77 * row[x * 3] + 150 * row[x * 3 + 1] + 29 * row[x * 3 + 2]) >> 8) / 255.f;
        }
    }
    auto output = infer(*ball, input, {1, 9, h, w});
    auto shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 4 || shape[1] < 9 || shape[2] != h || shape[3] != w)
        throw std::runtime_error("Неподдерживаемый формат модели мяча");
    const float *heat = output[0].GetTensorData<float>();
    std::vector<std::vector<BallCandidate>> rows;
    for (size_t frame = 0; frame < images.size(); ++frame) {
        const float *hm = heat + (9 - images.size() + frame) * w * h;
        std::vector<uint8_t> seen(w * h);
        std::vector<BallCandidate> candidates;
        for (int start = 0; start < w * h; ++start) {
            if (seen[start] || hm[start] < .35f)
                continue;
            std::vector<int> queue{start};
            seen[start] = 1;
            double total = 0, xsum = 0, ysum = 0, score = 0;
            for (size_t q = 0; q < queue.size(); ++q) {
                int pos = queue[q], x = pos % w, y = pos / w;
                float v = hm[pos];
                total += v;
                xsum += x * v;
                ysum += y * v;
                score = std::max(score, double(v));
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        int xx = x + dx, yy = y + dy;
                        if (xx < 0 || xx >= w || yy < 0 || yy >= h)
                            continue;
                        int next = yy * w + xx;
                        if (!seen[next] && hm[next] >= .35f) {
                            seen[next] = 1;
                            queue.push_back(next);
                        }
                    }
            }
            if (queue.size() >= 2 && total > 0)
                candidates.push_back(
                    {{xsum / total * images[frame].width() / w, ysum / total * images[frame].height() / h},
                     score,
                     total});
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](auto &a, auto &b) { return a.support > b.support; });
        if (candidates.size() > 8)
            candidates.resize(8);
        rows.push_back(std::move(candidates));
    }
    for (auto &im : images) {
        history.push_back(im);
        while (history.size() > 8)
            history.pop_front();
    }
    ballMs = timer.nsecsElapsed() / 1e6;
    return rows;
}
void Models::extract(Person &p, const QImage &image, bool numbered) {
    if (!p.embedding.empty())
        return;
    QElapsedTimer timer;
    timer.start();
    QRect rect(int(p.box.x), int(p.box.y), int(p.box.w), int(p.box.h));
    auto crop = image.copy(rect);
    if (crop.isNull())
        return;
    auto input = chw(crop, 128, 256, true);
    auto output = infer(*reid, input, {1, 3, 256, 128});
    auto n = output[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const float *f = output[0].GetTensorData<float>();
    p.embedding = normalized(Feature(f, f + n));
    if (numbered && p.box.h >= 80 && !characters.isEmpty()) {
        // Numbers are a supplementary observation; no crop OCR can change an enrolled team.
        auto torso = crop.copy(QRect(int(crop.width() * .12), int(crop.height() * .18),
                                     int(crop.width() * .76), int(crop.height() * .42)));
        int tw = std::clamp(int(48. * torso.width() / std::max(1, torso.height())), 16, 320);
        QImage padded(320, 48, QImage::Format_RGB888);
        padded.fill(Qt::black);
        {
            QPainter painter(&padded);
            painter.drawImage(QRect(0, 0, tw, 48), torso);
        }
        auto rec = chw(padded, 320, 48);
        for (auto &v : rec)
            v = (v - .5f) / .5f;
        auto out = infer(*ocr, rec, {1, 3, 48, 320});
        auto shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() == 3) {
            int steps = int(shape[1]), classes = int(shape[2]), previous = -1;
            auto values = out[0].GetTensorData<float>();
            QString text;
            double conf = 0;
            int count = 0;
            bool onlyDigits = true;
            for (int t = 0; t < steps; ++t) {
                const float *row = values + size_t(t) * classes;
                int best = int(std::max_element(row, row + classes) - row);
                if (best && best != previous) {
                    QString ch = characters.value(best);
                    if (ch.size() != 1 || !ch[0].isDigit())
                        onlyDigits = false;
                    text += ch;
                    conf += row[best];
                    ++count;
                }
                previous = best;
            }
            if (onlyDigits && count && text.size() <= 2) {
                bool ok = false;
                int number = text.toInt(&ok);
                if (ok && number < 100) {
                    p.jersey = number;
                    p.jerseyScore = conf / count;
                }
            }
        }
    }
    reidMs += timer.nsecsElapsed() / 1e6;
}
} // namespace vs
