#include "core/tracking.h"
#include <iostream>
#include <stdexcept>
using namespace vs;
static void check(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
static Person person(double x = 200, double y = 100) {
    Person p;
    p.box = {x, y, 45, 150};
    p.score = .95;
    p.embedding = normalized({1, 0, 0, 0});
    p.color = p.embedding;
    return p;
}
static IdentityTracker fixture() {
    IdentityTracker t;
    t.court = {{0, 0}, {640, 0}, {640, 480}, {0, 480}};
    t.center = std::array<Point, 2>{{{0, 220}, {640, 220}}};
    t.options.autoEnroll = false;
    return t;
}
static void identityRegression() {
    auto t = fixture();
    int p = t.addPlayer('B', "Player", "libero");
    t.players[p].remember(person().embedding, person().color);
    t.calibrate('B', {220, 250});
    for (int cycle = 0; cycle < 105; ++cycle) {
        std::vector<Person> rows{person(200 + (cycle % 4) * 3)};
        t.update(rows, cycle * 3.);
        check(rows[0].player == p, "lost/reappearing player changed identity");
        check(rows[0].team == 'B', "team changed after reacquisition");
    }
    check(t.identityCount() == 1, "a disappearance created another roster number");
    t.resetContinuity();
    std::vector<Person> back{person()};
    t.update(back, 10);
    check(back[0].player == p, "seek discarded identity memory");
}
static void ambiguityRegression() {
    auto t = fixture();
    int a = t.addPlayer('A'), b = t.addPlayer('A');
    t.players[a].remember(person().embedding, person().color);
    t.players[b].remember(person().embedding, person().color);
    t.calibrate('A', {220, 250});
    std::vector<Person> rows{person()};
    t.update(rows, 0);
    check(rows[0].player == 0 && rows[0].candidates.size() == 2,
          "ambiguous same-uniform people were confidently merged");
    int tid = rows[0].tracklet;
    t.assign(tid, b);
    rows = {person()};
    t.update(rows, .016);
    check(rows[0].player == b, "manual identity confirmation not applied");
    std::vector<Person> pair{person(), person(220)};
    t.update(pair, .032);
    check(!(pair[0].player && pair[0].player == pair[1].player),
          "one identity assigned to two simultaneous people");
}
static void teamRegression() {
    auto t = fixture();
    t.calibrate('B', {100, 300});
    check(t.sideOf({200, 300}) == 'B' && t.sideOf({200, 120}) == 'A',
          "manual teams do not calibrate court sides");
    auto reversed = fixture();
    reversed.center = std::array<Point, 2>{{{640, 220}, {0, 220}}};
    reversed.calibrate('B', {100, 300});
    check(reversed.sideOf({200, 300}) == 'B', "reversing center-line points swaps team names");
    int id = t.addPlayer('B');
    t.players[id].remember(person().embedding, person().color);
    for (int i = 0; i < 50; ++i) {
        std::vector<Person> rows{person(200, 100 - i * 2)};
        t.update(rows, i / 60.);
        check(rows[0].player == id && rows[0].team == 'B', "jump across projected line changes team or ID");
    }
}
static void spectators() {
    auto t = fixture();
    int id = t.addPlayer('A');
    t.players[id].remember(person().embedding, person().color);
    for (int i = 0; i < 200; ++i) {
        auto p = person(600);
        p.embedding = normalized({0, 1, 0, 0});
        p.color = p.embedding;
        std::vector<Person> rows{p};
        t.update(rows, i / 60.);
        check(rows[0].player == 0, "unknown person silently consumed a player number");
    }
    check(t.identityCount() == 1, "unmatched spectator inflates identity counter");
}
static void ballRegression() {
    BallTracker t;
    int seen = 0;
    for (int i = 0; i < 60; ++i) {
        double tm = i / 60.;
        std::vector<BallCandidate> c{
            {{500, 100}, .99, 10}, {{100 + i * 7., 180 - i * 2.}, .9, 8}, {{300 + i * .4, 400}, .98, 9}};
        auto b = t.update(c, tm, {});
        if (b) {
            ++seen;
            check(b->point.y < 200 && b->point.x != 500,
                  "stationary or rolling spare ball steals active trajectory");
        }
    }
    check(seen > 40, "moving ball never acquires a track");
    check(!t.update({}, 2, {}), "missing ball was fabricated");
    t.reset();
    check(!t.update({{{20, 20}, .9, 1}}, 3, {}), "single cold-start candidate confirmed without context");
    auto b = t.update({{{20, 20}, .9, 1}}, 3.01, {}, Point{20, 20});
    check(bool(b), "manual ball seed ignored");
}
static void substituteRegression() {
    auto t = fixture();
    std::vector<int> ids;
    for (int i = 0; i < 7; ++i) {
        int id = t.addPlayer('A', "", i == 6 ? "libero" : "player");
        ids.push_back(id);
        Feature f(7);
        f[i] = 1;
        t.players[id].remember(f, f);
        if (i == 6) {
            t.players[id].initiallyActive = false;
            t.players[id].active = false;
        }
    }
    auto observationFor = [&](int i, double x) {
        auto p = person(x);
        p.embedding = Feature(7);
        p.embedding[i] = 1;
        p.color = p.embedding;
        return p;
    };
    for (int frame = 0; frame < 20; ++frame) {
        std::vector<Person> rows;
        for (int i = 0; i < 6; ++i)
            rows.push_back(observationFor(i, 20 + i * 70));
        t.update(rows, frame / 60.);
    }
    check(t.players[ids[0]].active, "starting player not active");
    for (int frame = 0; frame < 12; ++frame) {
        std::vector<Person> rows;
        for (int i = 1; i < 6; ++i)
            rows.push_back(observationFor(i, 20 + i * 70));
        t.update(rows, 2 + frame / 60.);
    }
    check(t.players[ids[0]].active, "offscreen server removed without an entering substitute");
    for (int frame = 0; frame < 15; ++frame) {
        std::vector<Person> rows;
        for (int i = 1; i < 7; ++i)
            rows.push_back(observationFor(i, 20 + (i % 6) * 70));
        t.update(rows, 3 + frame / 60.);
    }
    check(t.players[ids[6]].active && !t.players[ids[0]].active,
          "libero substitution did not preserve distinct roster cards");
    check(t.identityCount() == 7, "substitution allocated another identity");
    t.swapSides();
    std::vector<Person> back{observationFor(6, 150)};
    t.resetContinuity();
    t.update(back, 10);
    check(back[0].player == ids[6] && back[0].team == 'A', "side switch changes the returning libero's team");
}
static void manualBallAndBench() {
    BallTracker tracker;
    auto corrected = tracker.update({}, 0, {}, Point{25, 30});
    check(corrected && distance(corrected->point, {25, 30}) < .001,
          "human ball correction ignored when network missed the ball");
    tracker.reset();
    auto spectator = person();
    spectator.active = false;
    spectator.joints[9] = {{100, 100}, 1};
    for (int i = 0; i < 20; ++i)
        check(!tracker.update({{{100, 100}, .99, 10}}, i / 60., {spectator}),
              "a stationary spare ball held on the bench was treated as the game ball");
    spectator.active = true;
    check(bool(tracker.update({{{100, 100}, .99, 10}}, .4, {spectator})),
          "game ball held by an active player was rejected");
}
static void featureBudget() {
    auto tracker = fixture();
    int id = tracker.addPlayer('A');
    tracker.players[id].remember(person().embedding, person().color);
    int calls = 0;
    for (int frame = 0; frame < 120; ++frame) {
        auto p = person();
        p.embedding.clear();
        std::vector<Person> rows{p};
        tracker.update(rows, frame / 60., [&](size_t i) {
            ++calls;
            rows[i].embedding = person().embedding;
        });
        check(rows[0].player == id, "feature reuse broke a continuously visible identity");
    }
    check(calls <= 6, "ReID runs unnecessarily on every healthy frame");
}
int main() {
    try {
        identityRegression();
        ambiguityRegression();
        teamRegression();
        spectators();
        ballRegression();
        substituteRegression();
        manualBallAndBench();
        featureBudget();
        std::cout << "PASS: persistent IDs, 105 losses, seeks, jumps, team calibration, ambiguity, "
                     "simultaneous uniqueness, spectators, spare balls\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
