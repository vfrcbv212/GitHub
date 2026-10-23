#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace vs {
struct Point {
    double x = 0, y = 0;
};
struct Box {
    double x = 0, y = 0, w = 0, h = 0;
    Point foot() const {
        return {x + w * .5, y + h};
    }
};
struct Joint {
    Point p;
    float score = 0;
};
using Feature = std::vector<float>;
double distance(Point a, Point b);
double overlap(Box a, Box b);
double cosine(const Feature &a, const Feature &b);
Feature normalized(Feature a);
bool inside(const std::vector<Point> &polygon, Point p);
struct Person {
    Box box;
    std::array<Joint, 17> joints{};
    float score = 0;
    Feature color, embedding;
    int jersey = -1;
    double jerseyScore = 0;
    int tracklet = 0, player = 0, reviewedPlayer = 0;
    char team = '?';
    bool referee = false, uncertain = false, active = false, readNumber = true;
    std::vector<int> candidates;
};
struct Player {
    int id = 0;
    char team = '?';
    std::string name, role = "player";
    int jersey = -1;
    bool manual = true, initiallyActive = true, active = true;
    std::vector<Feature> gallery, colors;
    double lastSeen = -1, lastInside = -100;
    Point lastPoint{};
    void remember(const Feature &f, const Feature &c);
};
struct Tracklet {
    int id = 0, player = 0, hits = 0, insideHits = 0;
    Box box;
    Point velocity{};
    double time = 0, born = 0, lastFeature = -100,lastNumber=-100;
    Feature color;
    bool referee = false;
    std::vector<int> candidates;
    char side = '?';
};
struct IdentityOptions {
    bool uniform = false, numbered = false;
    bool autoEnroll = true;
};
class IdentityTracker {
  public:
    std::map<int, Player> players;
    std::vector<Point> court;
    std::optional<std::array<Point, 2>> center;
    std::optional<Box> referee;
    IdentityOptions options;
    int addPlayer(char team, const std::string &name = "", const std::string &role = "player",
                  int jersey = -1);
    void calibrate(char team, Point foot);
    void setNearTeam(char team) {
        nearTeam = team;
    }
    void swapSides();
    void resetContinuity();
    void assign(int tracklet, int player);
    void update(std::vector<Person> &people, double timestamp,
                const std::function<void(size_t)> &extract = {});
    char sideOf(Point p) const;
    size_t identityCount() const {
        return players.size();
    }

  private:
    std::map<int, Tracklet> tracks;
    int nextTrack = 1, nextPlayer = 1;
    char nearTeam = 'A';
    std::map<int, char> sideTeams;
    double firstTime = -1;
    std::vector<int> matchGallery(const Person &person, const Tracklet &track, double now,
                                  const std::vector<int> &occupied) const;
};
struct BallCandidate {
    Point point;
    double score = 0, support = 0;
};
struct Ball {
    Point point;
    std::vector<Point> trail;
    double score = 0;
    bool uncertain = false;
};
class BallTracker {
  public:
    std::optional<Ball> update(std::vector<BallCandidate> candidates, double time,
                               const std::vector<Person> &people, std::optional<Point> seed = {});
    void reset();

  private:
    struct Row {
        double time;
        std::vector<BallCandidate> candidates;
    };
    std::vector<Row> history;
    std::optional<Point> last;
    Point velocity{};
    double lastTime = -100;
    std::optional<Point> stationary;
    double stationarySince = 0;
    bool seeded = false;
    std::vector<std::pair<double, Point>> trail;
};
} // namespace vs
