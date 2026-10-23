#pragma once
#include "project.h"
#include <deque>
#include <onnxruntime_cxx_api.h>

namespace vs {
class Models {
  public:
    explicit Models(const QString &folder, bool gpu = true);
    std::vector<Person> pose(const QImage &image);
    std::vector<std::vector<BallCandidate>> balls(const std::vector<QImage> &images);
    void extract(Person &person, const QImage &image, bool numbered);
    void resetTemporal() {
        history.clear();
    }
    QString backend, warning;
    double poseMs = 0, ballMs = 0, reidMs = 0;

  private:
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "VolleyScope"};
    std::unique_ptr<Ort::Session> body, ball, reid, ocr;
    QStringList characters;
    std::deque<QImage> history;
    int poseSize = 960;
    std::vector<float> chw(const QImage &image, int width, int height, bool imagenet = false);
    std::vector<Ort::Value> infer(Ort::Session &session, std::vector<float> &tensor,
                                  const std::vector<int64_t> &shape);
    std::unique_ptr<Ort::Session> load(const QString &file, bool gpu);
};
} // namespace vs
