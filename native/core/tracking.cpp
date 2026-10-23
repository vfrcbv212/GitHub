#include "tracking.h"
#include <limits>
#include <numeric>

namespace vs {
double distance(Point a, Point b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}
double overlap(Box a, Box b) {
    double w = std::max(0., std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x));
    double h = std::max(0., std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y));
    double i = w * h;
    return i / std::max(1., a.w * a.h + b.w * b.h - i);
}
Feature normalized(Feature a) {
    double n = 0;
    for (auto x : a)
        n += x * x;
    n = std::sqrt(n);
    if (n > 1e-8)
        for (auto &x : a)
            x = float(x / n);
    return a;
}
double cosine(const Feature &a, const Feature &b) {
    if (a.empty() || a.size() != b.size())
        return 0;
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}
bool inside(const std::vector<Point> &v, Point p) {
    if (v.size() < 3)
        return false;
    bool in = false;
    for (size_t i = 0, j = v.size() - 1; i < v.size(); j = i++) {
        const auto &a = v[i];
        const auto &b = v[j];
        if ((a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x)
            in = !in;
    }
    return in;
}
static double signDistance(std::array<Point, 2> l, Point p) {
    auto a = l[0], b = l[1];
    return ((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)) / std::max(1., distance(a, b));
}
void Player::remember(const Feature &f, const Feature &c) {
    if (!f.empty() && (gallery.empty() || cosine(gallery.back(), f) < .995)) {
        gallery.push_back(normalized(f));
        if (gallery.size() > 16)
            gallery.erase(gallery.begin() + 1); // retain initial reviewed sample
    }
    if (!c.empty()) {
        colors.push_back(normalized(c));
        if (colors.size() > 12)
            colors.erase(colors.begin() + 1);
    }
}
int IdentityTracker::addPlayer(char team, const std::string &name, const std::string &role, int jersey) {
    while (players.count(nextPlayer))
        ++nextPlayer;
    int id = nextPlayer++;
    Player p;
    p.id = id;
    p.team = team;
    p.name = name;
    p.role = role;
    p.jersey = jersey;
    players[id] = p;
    return id;
}
void IdentityTracker::calibrate(char team, Point p) {
    if (!center || team == '?')
        return;
    double d = signDistance(*center, p);
    if (std::abs(d) < 5)
        return;
    int s = d > 0 ? 1 : -1;
    sideTeams[s] = team;
    sideTeams[-s] = team == 'A' ? 'B' : 'A';
}
char IdentityTracker::sideOf(Point p) const {
    if (!center)
        return '?';
    double d = signDistance(*center, p);
    if (std::abs(d) < 15)
        return '?';
    int s = d > 0 ? 1 : -1;
    if (auto it = sideTeams.find(s); it != sideTeams.end())
        return it->second;
    // Geometric orientation never determines the *name* of the team.
    double bottom = signDistance(*center, {(center->at(0).x + center->at(1).x) / 2, 100000});
    return (d * bottom > 0) ? nearTeam : (nearTeam == 'A' ? 'B' : 'A');
}
void IdentityTracker::resetContinuity() {
    tracks.clear();
    for (auto &[id, p] : players) {
        p.active = p.initiallyActive;
        p.lastInside = -100;
    }
}
void IdentityTracker::swapSides() {
    nearTeam = nearTeam == 'A' ? 'B' : 'A';
    for (auto &[sign, team] : sideTeams)
        team = team == 'A' ? 'B' : 'A';
}
void IdentityTracker::assign(int tid, int id) {
    if (!players.count(id) || !tracks.count(tid))
        return;
    for (auto &[key, t] : tracks)
        if (t.player == id)
            t.player = 0;
    tracks[tid].player = id;
    tracks[tid].candidates.clear();
}
static double gallerySimilarity(const std::vector<Feature> &gallery, const Feature &f) {
    double best = 0;
    for (const auto &g : gallery)
        best = std::max(best, cosine(g, f));
    return best;
}
std::vector<int> IdentityTracker::matchGallery(const Person &p, const Tracklet &t, double now,
                                               const std::vector<int> &occupied) const {
    std::vector<std::pair<double, int>> ranked;
    for (const auto &[id, card] : players) {
        if (std::find(occupied.begin(), occupied.end(), id) != occupied.end())
            continue;
        // A side is a weak return gate, not a rule that can change an enrolled team.
        double sim = gallerySimilarity(card.gallery, p.embedding);
        if (sim < .64)
            continue;
        double color = gallerySimilarity(card.colors, p.color);
        double score = sim + (options.uniform ? 0 : .035 * color);
        if (t.side != '?' && card.team != '?' && card.team != t.side && t.hits >= 8)
            score -= .025;
        if (options.numbered && p.jerseyScore > .92 && card.jersey >= 0) {
            if (p.jersey != card.jersey)
                continue;
            score += .06;
        }
        if (now >= card.lastSeen && now - card.lastSeen < 2 &&
            distance(card.lastPoint, p.box.foot()) < p.box.h)
            score += .025;
        ranked.emplace_back(score, id);
    }
    std::sort(ranked.rbegin(), ranked.rend());
    std::vector<int> result;
    if (ranked.empty() || ranked[0].first < .73)
        return result;
    double gap = std::abs(now - players.at(ranked[0].second).lastSeen);
    const double threshold = gap > 3 ? .85 : .79;
    bool unique = ranked.size() == 1 || ranked[0].first - ranked[1].first > .055;
    if (ranked[0].first >= threshold && unique)
        return {ranked[0].second};
    for (auto [score, id] : ranked)
        if (ranked[0].first - score < .085 && result.size() < 4)
            result.push_back(id);
    // A single weak match is still ambiguous. Negative ID denotes an unconfirmed suggestion.
    if (result.size() == 1)
        result[0] = -result[0];
    return result;
}
// Rectangular Hungarian assignment with dummy columns (unmatched detections).
static std::vector<int> assignment(const std::vector<std::vector<double>> &a) {
    int n = int(a.size());
    if (!n)
        return {};
    int m = int(a[0].size());
    std::vector<double> u(n + 1), v(m + 1);
    std::vector<int> p(m + 1), way(m + 1);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(m + 1, 1e9);
        std::vector<bool> used(m + 1);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            double delta = 1e9;
            for (int j = 1; j <= m; ++j)
                if (!used[j]) {
                    double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            for (int j = 0; j <= m; ++j)
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else
                    minv[j] -= delta;
            j0 = j1;
        } while (p[j0]);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    std::vector<int> result(n, -1);
    for (int j = 1; j <= m; ++j)
        if (p[j])
            result[p[j] - 1] = j - 1;
    return result;
}
void IdentityTracker::update(std::vector<Person> &people, double now,
                             const std::function<void(size_t)> &extract) {
    if (firstTime < 0)
        firstTime = now;
    std::vector<int> ids;
    for (auto &[id, t] : tracks)
        if (now >= t.time && now - t.time < 1.2)
            ids.push_back(id);
    // Extract on overlaps, reacquisition, and periodic clear samples, not for every person in every frame.
    for (size_t j = 0; j < people.size(); ++j) {
        bool need = ids.empty();
        double best = 0;
        Tracklet *close = nullptr;
        for (int id : ids) {
            double o = overlap(tracks[id].box, people[j].box);
            if (o > best) {
                best = o;
                close = &tracks[id];
            }
        }
        if (!close || best < .18 || now - close->time > .2 || now - close->lastFeature > .65 ||
            (close->player == 0 && now-close->lastFeature>.25))
            need = true;
        for (size_t k = 0; k < people.size(); ++k)
            if (k != j && overlap(people[j].box, people[k].box) > .18 && (!close || now-close->lastFeature>.1))
                need = true;
        people[j].readNumber=!close||now-close->lastNumber>.65;
        if (need && extract)
            extract(j);
    }
    std::vector<std::vector<double>> cost(people.size(),
                                          std::vector<double>(ids.size() + people.size(), .92));
    for (size_t j = 0; j < people.size(); ++j)
        for (size_t i = 0; i < ids.size(); ++i) {
            auto &t = tracks[ids[i]];
            double dt = now - t.time;
            Box predicted = t.box;
            predicted.x += t.velocity.x * std::min(dt, .3);
            predicted.y += t.velocity.y * std::min(dt, .3);
            double o = overlap(predicted, people[j].box),
                   d = distance(predicted.foot(), people[j].box.foot()) / std::max(30., t.box.h);
            if (o < .04 && d > .65) {
                cost[j][i] = 10;
                continue;
            }
            double sim = 0;
            bool have = t.player && players.count(t.player) && !people[j].embedding.empty();
            if (have)
                sim = gallerySimilarity(players.at(t.player).gallery, people[j].embedding);
            if (have && sim < (dt > .6 ? .82 : dt > .3 ? .72 : .46)) {
                cost[j][i] = 10;
                continue;
            }
            if (have && options.numbered && people[j].jerseyScore > .97 && players.at(t.player).jersey >= 0 &&
                people[j].jersey != players.at(t.player).jersey) {
                cost[j][i] = 10;
                continue;
            }
            cost[j][i] = .55 * (1 - o) + .25 * std::min(1., d) +
                         .2 * (1 - (have ? sim : cosine(t.color, people[j].color)));
        }
    auto matched = assignment(cost);
    std::vector<int> used;
    for (size_t j = 0; j < people.size(); ++j) {
        auto &p = people[j];
        int tid = 0;
        if (matched[j] >= 0 && size_t(matched[j]) < ids.size() && cost[j][matched[j]] < .89)
            tid = ids[matched[j]];
        if (!tid) {
            tid = nextTrack++;
            Tracklet t;
            t.id = tid;
            t.box = p.box;
            t.time = now;
            t.born = now;
            tracks[tid] = t;
        }
        p.tracklet = tid;
        if (p.reviewedPlayer && players.count(p.reviewedPlayer) &&
            std::find(used.begin(), used.end(), p.reviewedPlayer) == used.end()) {
            assign(tid, p.reviewedPlayer);
            players[p.reviewedPlayer].remember(p.embedding, p.color);
        }
        if (tracks[tid].player)
            used.push_back(tracks[tid].player);
    }
    for (size_t j = 0; j < people.size(); ++j) {
        auto &p = people[j];
        auto &t = tracks[p.tracklet];
        double dt = now - t.time;
        t.hits++;
        if (dt > 0 && dt < .5) {
            t.velocity.x = .6 * t.velocity.x + .4 * (p.box.x - t.box.x) / dt;
            t.velocity.y = .6 * t.velocity.y + .4 * (p.box.y - t.box.y) / dt;
        }
        t.box = p.box;
        t.time = now;
        t.color = p.color;
        if (!p.embedding.empty())
            t.lastFeature = now;
        if(!p.embedding.empty()&&p.readNumber)t.lastNumber=now;
        if (referee && overlap(*referee, p.box) > .42) {
            t.referee = true;
            t.player = 0;
        }
        if (t.referee) {
            p.referee = true;
            continue;
        }
        if (inside(court, p.box.foot()))
            ++t.insideHits;
        else
            t.insideHits = std::max(0, t.insideHits - 1);
        if (!t.player) {
            char side = sideOf(p.box.foot());
            if (side != '?')
                t.side = side;
            auto found = p.embedding.empty()?t.candidates:matchGallery(p, t, now, used);
            t.candidates = found;
            if (found.size() == 1 && found[0] > 0) {
                t.player = found[0];
                used.push_back(t.player);
            }
            // Auto-enrollment is limited to initial setup. Disappearances later never consume new player
            // numbers.
            else if (found.empty() && options.autoEnroll && now - firstTime < 8 && t.hits >= 10 &&
                     t.insideHits >= 8 && t.side != '?' && !p.embedding.empty()) {
                int count = 0;
                for (const auto &[id, card] : players)
                    if (card.team == t.side)
                        ++count;
                if (count < 6) {
                    t.player = addPlayer(t.side);
                    players[t.player].manual = false;
                    used.push_back(t.player);
                }
            }
        }
        p.player = t.player;
        p.uncertain = !t.player && t.hits >= 3;
        p.candidates = t.candidates;
        for (int &id : p.candidates)
            id = std::abs(id);
        if (t.player) {
            auto &card = players[t.player];
            p.team = card.team;
            card.lastSeen = now;
            card.lastPoint = p.box.foot();
            if (t.insideHits >= 8) {
                card.active = true;
                card.lastInside = now;
            }
            if (card.active) {
                std::vector<int> active;
                for (const auto &[id, other] : players)
                    if (other.team == card.team && other.active)
                        active.push_back(id);
                if (active.size() > 6) {
                    auto old = std::min_element(active.begin(), active.end(), [&](int a, int b) {
                        return players[a].lastInside < players[b].lastInside;
                    });
                    if (*old != card.id && now - players[*old].lastInside > 2)
                        players[*old].active = false;
                }
            }
            p.active = card.active;
            bool clear = p.score > .5;
            for (size_t k = 0; k < people.size(); ++k)
                if (k != j && overlap(p.box, people[k].box) > .15)
                    clear = false;
            if (clear)
                card.remember(p.embedding, p.color);
        }
    }
    for (auto it = tracks.begin(); it != tracks.end();)
        if (now - it->second.time > 10 || now < it->second.time)
            it = tracks.erase(it);
        else
            ++it;
}
void BallTracker::reset() {
    history.clear();
    last.reset();
    velocity = {};
    lastTime = -100;
    trail.clear();
    stationary.reset();
    seeded = false;
}
std::optional<Ball> BallTracker::update(std::vector<BallCandidate> c, double now,
                                        const std::vector<Person> &people, std::optional<Point> seed) {
    history.push_back({now, c});
    if (history.size() > 3)
        history.erase(history.begin());
    double gap = now - lastTime;
    if (c.empty()) {
        if (gap > .25)
            trail.clear();
        return {};
    }
    auto nearHand = [&](Point p) {
        for (auto &person : people)
            for (int k : {9, 10})
                if (person.joints[k].score > .45 &&
                    distance(p, person.joints[k].p) < std::max(22., person.box.h * .18))
                    return true;
        return false;
    };
    std::optional<Point> expected;
    if (seed) {
        expected = seed;
        seeded = true;
    } else if (last && gap < .5)
        expected = Point{last->x + velocity.x * gap, last->y + velocity.y * gap};
    if (gap > .5 && !seed)
        seeded = false;
    if (!expected) {
        std::vector<BallCandidate> moving;
        for (auto &candidate : c) {
            bool valid = nearHand(candidate.point);
            if (history.size() == 3 && now - history.front().time < .25) {
                Point cursor = candidate.point;
                bool coherent = true;
                for (int i = 1; i >= 0; --i) {
                    if (history[i].candidates.empty()) {
                        coherent = false;
                        break;
                    }
                    auto nearest = std::min_element(
                        history[i].candidates.begin(), history[i].candidates.end(), [&](auto &a, auto &b) {
                            return distance(a.point, cursor) < distance(b.point, cursor);
                        });
                    if (distance(nearest->point, cursor) > std::max(35., 1800 * (now - history[i].time))) {
                        coherent = false;
                        break;
                    }
                    cursor = nearest->point;
                }
                double dt = now - history.front().time;
                double speed = distance(candidate.point, cursor) / std::max(.001, dt);
                valid = valid || (coherent && speed > 100);
            }
            if (valid)
                moving.push_back(candidate);
        }
        c = std::move(moving);
    }
    if (expected)
        c.erase(std::remove_if(c.begin(), c.end(),
                               [&](auto &a) {
                                   return distance(a.point, *expected) >
                                          std::max(75., 1800 * std::min(gap, .4));
                               }),
                c.end());
    if (c.empty())
        return {};
    std::sort(c.begin(), c.end(), [&](auto &a, auto &b) {
        return expected ? distance(a.point, *expected) < distance(b.point, *expected) : a.score > b.score;
    });
    if (!expected && c.size() > 1 && c[0].score - c[1].score < .1)
        return {};
    auto chosen = c.front();
    if (!stationary || distance(chosen.point, *stationary) > 5) {
        stationary = chosen.point;
        stationarySince = now;
    } else if (now - stationarySince > .5 && !seeded && !nearHand(chosen.point)) {
        last.reset();
        trail.clear();
        return {};
    }
    if (last && gap > 0 && gap < .25) {
        velocity.x = .6 * velocity.x + .4 * (chosen.point.x - last->x) / gap;
        velocity.y = .6 * velocity.y + .4 * (chosen.point.y - last->y) / gap;
    } else {
        velocity = {};
        trail.clear();
    }
    last = chosen.point;
    lastTime = now;
    trail.push_back({now, chosen.point});
    trail.erase(std::remove_if(trail.begin(), trail.end(), [&](auto &r) { return now - r.first > .5; }),
                trail.end());
    Ball ball;
    ball.point = chosen.point;
    ball.score = chosen.score;
    ball.uncertain = gap > .5;
    for (auto &row : trail)
        ball.trail.push_back(row.second);
    return ball;
}
} // namespace vs
