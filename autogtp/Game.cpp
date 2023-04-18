/*
    This file is part of SAI, which is a fork of Leela Zero.
    Copyright (C) 2017-2018 Marco Calignano
    Coptright (C) 2018-2019 SAI Team

    SAI is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SAI is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SAI.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <QUuid>
#include <QFile>
#include <QTextStream>
#include <QRegularExpression>
#include <QFileInfo>
#include "Game.h"
#include <QtGlobal>
#include <QString>

#if QT_VERSION < QT_VERSION_CHECK(5, 14, 0)
namespace Qt
{
    static auto endl = ::endl;
    static auto SkipEmptyParts = QString::SkipEmptyParts;
}
#endif

Game::Game(const Engine& engine) :
    QProcess(),
    m_engine(engine),
    m_isHandicap(false),
    m_resignation(false),
    m_blackToMove(true),
    m_blackResigned(false),
    m_passes(0),
    m_moveNum(0)
{
    m_fileName = QUuid::createUuid().toRfc4122().toHex();
}

bool Game::checkGameEnd() {
    return (m_resignation ||
            m_passes > 1 ||
            m_moveNum > (BOARD_SIZE * BOARD_SIZE * 2));
}

void Game::error(int errnum) {
    QTextStream(stdout) << "*ERROR*: ";
    switch (errnum) {
        case Game::NO_LEELAZ:
            QTextStream(stdout)
                << "No 'sai' binary found." << Qt::endl;
            break;
        case Game::PROCESS_DIED:
            QTextStream(stdout)
                << "The 'sai' process died unexpected." << Qt::endl;
            break;
        case Game::WRONG_GTP:
            QTextStream(stdout)
                << "Error in GTP response." << Qt::endl;
            break;
        case Game::LAUNCH_FAILURE:
            QTextStream(stdout)
                << "Could not talk to engine after launching." << Qt::endl;
            break;
        default:
            QTextStream(stdout)
                << "Unexpected error." << Qt::endl;
            break;
    }
}

bool Game::eatNewLine() {
    char readBuffer[256];
    // Eat double newline from GTP protocol
    if (!waitReady()) {
        error(Game::PROCESS_DIED);
        return false;
    }
    auto readCount = readLine(readBuffer, 256);
    if (readCount < 0) {
        error(Game::WRONG_GTP);
        return false;
    }
    return true;
}

bool Game::sendGtpCommand(QString cmd) {
    write(qPrintable(cmd.append("\n")));
    waitForBytesWritten(-1);
    if (!waitReady()) {
        error(Game::PROCESS_DIED);
        return false;
    }
    char readBuffer[256];
    int readCount = readLine(readBuffer, 256);
    if (readCount <= 0 || readBuffer[0] != '=') {
        QTextStream(stdout) << "GTP: " << readBuffer << Qt::endl;
        error(Game::WRONG_GTP);
        return false;
    }
    if (!eatNewLine()) {
        error(Game::PROCESS_DIED);
        return false;
    }
    return true;
}

void Game::checkVersion(const VersionTuple &min_version) {
    write(qPrintable("version\n"));
    waitForBytesWritten(-1);
    if (!waitReady()) {
        error(Game::LAUNCH_FAILURE);
        exit(EXIT_FAILURE);
    }
    char readBuffer[256];
    int readCount = readLine(readBuffer, 256);
    //If it is a GTP comment just print it and wait for the real answer
    //this happens with the winogard tuning
    if (readBuffer[0] == '#') {
        readBuffer[readCount-1] = 0;
        QTextStream(stdout) << readBuffer << Qt::endl;
        if (!waitReady()) {
            error(Game::PROCESS_DIED);
            exit(EXIT_FAILURE);
        }
        readCount = readLine(readBuffer, 256);
    }
    // We expect to read at last "=, space, something"
    if (readCount <= 3 || readBuffer[0] != '=') {
        QTextStream(stdout) << "GTP: " << readBuffer << Qt::endl;
        error(Game::WRONG_GTP);
        exit(EXIT_FAILURE);
    }
    QString version_buff(&readBuffer[2]);
    version_buff = version_buff.simplified();
    QStringList version_list = version_buff.split(".");
    if (version_list.size() < 2) {
        QTextStream(stdout)
            << "Unexpected SAI version: " << version_buff << Qt::endl;
        exit(EXIT_FAILURE);
    }
    if (version_list.size() < 3) {
        version_list.append("0");
    }
    int versionCount = (version_list[0].toInt() - std::get<0>(min_version)) * 10000;
    versionCount += (version_list[1].toInt() - std::get<1>(min_version)) * 100;
    versionCount += version_list[2].toInt() - std::get<2>(min_version);
    if (versionCount < 0) {
        QTextStream(stdout)
            << "SAI version is too old, saw " << version_buff
            << " but expected "
            << std::get<0>(min_version) << "."
            << std::get<1>(min_version) << "."
            << std::get<2>(min_version)  << Qt::endl;
        QTextStream(stdout)
            << "Check https://github.com/sai-dev/sai for updates." << Qt::endl;
        exit(EXIT_FAILURE);
    }
    if (!eatNewLine()) {
        error(Game::WRONG_GTP);
        exit(EXIT_FAILURE);
    }
}

bool Game::gameStart(const VersionTuple &min_version,
                     const QString &sgf,
                     const int moves) {
    start(m_engine.getCmdLine());
    if (!waitForStarted()) {
        error(Game::NO_LEELAZ);
        return false;
    }
    // This either succeeds or we exit immediately, so no need to
    // check any return values.
    checkVersion(min_version);
    QTextStream(stdout) << "Engine has started." << Qt::endl;
    //If there is an sgf file to start playing from then it will contain
    //whether there is handicap in use. If there is no sgf file then instead,
    //check whether there are any handicap commands to send (these fail
    //if the board is not empty).
    //Then send the rest of the GTP commands after any SGF has been loaded so
    //that they can override any settings loaded from the SGF.
    if (!sgf.isEmpty()) {
        QFile sgfFile(sgf + ".sgf");
        if (!sgfFile.exists()) {
            QTextStream(stdout) << "Cannot find sgf file " << sgf << Qt::endl;
            exit(EXIT_FAILURE);
        }
        sgfFile.open(QIODevice::Text | QIODevice::ReadOnly);
        const auto sgfData = QTextStream(&sgfFile).readAll();
        const auto re = QRegularExpression("HA\\[\\d+\\]");
        const auto match = re.match(sgfData);
        m_isHandicap = match.hasMatch();
        sgfFile.close();
        if (moves == 0) {
            loadSgf(sgf);
        } else {
            loadSgf(sgf, moves);
        }
        setMovesCount(moves);
    } else {
        for (auto command : m_engine.m_commands.filter("handicap")) {
            QTextStream(stdout) << command << Qt::endl;
            if (!sendGtpCommand(command))
            {
                QTextStream(stdout) << "GTP failed on: " << command << Qt::endl;
                exit(EXIT_FAILURE);
            }
            m_isHandicap = true;
            m_blackToMove = false;
        }
    }
    const auto re = QRegularExpression("^((?!handicap).)*$");
    for (auto command : m_engine.m_commands.filter(re)) {
        QTextStream(stdout) << command << Qt::endl;
        if (!sendGtpCommand(command))
        {
            QTextStream(stdout) << "GTP failed on: " << command << Qt::endl;
            exit(EXIT_FAILURE);
        }
    }
    QTextStream(stdout) << "Starting GTP commands sent." << Qt::endl;
    return true;
}

void Game::move() {
    m_moveNum++;
    QString moveCmd;
    if (m_blackToMove) {
        moveCmd = "genmove b\n";
    } else {
        moveCmd = "genmove w\n";
    }
    write(qPrintable(moveCmd));
    waitForBytesWritten(-1);
}

void Game::setMovesCount(int moves) {
    m_moveNum = moves;
    //The game always starts at move 0 (GTP states that handicap stones are not part
    //of the move history), so if there is no handicap then black moves on even
    //numbered turns but if there is handicap then black moves on odd numbered turns.
    m_blackToMove = (moves % 2) == (m_isHandicap ? 1 : 0);
}

bool Game::waitReady() {
    while (!canReadLine() && state() == QProcess::Running) {
        waitForReadyRead(-1);
    }
    // somebody crashed
    if (state() != QProcess::Running) {
        return false;
    }
    return true;
}

bool Game::readMove() {
    char readBuffer[256];
    int readCount = readLine(readBuffer, 256);
    if (readCount <= 3 || readBuffer[0] != '=') {
        error(Game::WRONG_GTP);
        QTextStream(stdout) << "Error read " << readCount << " '";
        QTextStream(stdout) << readBuffer << "'" << Qt::endl;
        terminate();
        return false;
    }
    // Skip "= "
    m_moveDone = readBuffer;
    m_moveDone.remove(0, 2);
    m_moveDone = m_moveDone.simplified();
    if (!eatNewLine()) {
        error(Game::PROCESS_DIED);
        return false;
    }
    if (readCount == 0) {
        error(Game::WRONG_GTP);
    }
    QTextStream(stdout) << m_moveNum << " (";
    QTextStream(stdout) << (m_blackToMove ? "B " : "W ") << m_moveDone << ") ";
    QTextStream(stdout).flush();
    if (m_moveDone.compare(QStringLiteral("pass"),
                          Qt::CaseInsensitive) == 0) {
        m_passes++;
    } else if (m_moveDone.compare(QStringLiteral("resign"),
                                 Qt::CaseInsensitive) == 0) {
        m_resignation = true;
        m_blackResigned = m_blackToMove;
    } else {
        m_passes = 0;
    }
    return true;
}

bool Game::setMove(const QString& m) {
    if (!sendGtpCommand(m)) {
        return false;
    }
    m_moveNum++;
    QStringList moves = m.split(" ");
    if (moves.at(2)
        .compare(QStringLiteral("pass"), Qt::CaseInsensitive) == 0) {
        m_passes++;
    } else if (moves.at(2)
               .compare(QStringLiteral("resign"), Qt::CaseInsensitive) == 0) {
        m_resignation = true;
        m_blackResigned = (moves.at(1).compare(QStringLiteral("black"), Qt::CaseInsensitive) == 0);
    } else {
        m_passes = 0;
    }
    m_blackToMove = !m_blackToMove;
    return true;
}

bool Game::nextMove() {
    if (checkGameEnd()) {
        return false;
    }
    m_blackToMove = !m_blackToMove;
    return true;
}

bool Game::getScore() {
    if (m_resignation) {
        if (m_blackResigned) {
            m_winner = QString(QStringLiteral("white"));
            m_result = "W+Resign ";
            QTextStream(stdout) << "Score: " << m_result << Qt::endl;
        } else {
            m_winner = QString(QStringLiteral("black"));
            m_result = "B+Resign ";
            QTextStream(stdout) << "Score: " << m_result << Qt::endl;
        }
    } else{
        write("final_score\n");
        waitForBytesWritten(-1);
        if (!waitReady()) {
            error(Game::PROCESS_DIED);
            return false;
        }
        char readBuffer[256];
        readLine(readBuffer, 256);
        m_result = readBuffer;
        m_result.remove(0, 2);
        if (readBuffer[2] == 'W') {
            m_winner = QString(QStringLiteral("white"));
        } else if (readBuffer[2] == 'B') {
            m_winner = QString(QStringLiteral("black"));
        } else if (readBuffer[2] == '0') {
            m_winner = QString(QStringLiteral("jigo"));
        }
        if (!eatNewLine()) {
            error(Game::PROCESS_DIED);
            return false;
        }
        QTextStream(stdout) << "Score: " << m_result;
    }
    if (m_winner.isNull()) {
        QTextStream(stdout) << "No winner found" << Qt::endl;
        return false;
    }
    QTextStream(stdout) << "Winner: " << m_winner << Qt::endl;
    return true;
}

int Game::getWinner() {
    if (m_winner.compare(QStringLiteral("white"),
                         Qt::CaseInsensitive) == 0)
        return Game::WHITE;
    else if (m_winner.compare(QStringLiteral("black"),
                              Qt::CaseInsensitive) == 0)
        return Game::BLACK;

    return Game::JIGO;
}

bool Game::writeSgf() {
    return sendGtpCommand(qPrintable("printsgf " + m_fileName + ".sgf"));
}

bool Game::loadTraining(const QString &fileName) {
    QTextStream(stdout) << "Loading " << fileName + ".train" << Qt::endl;
    return sendGtpCommand(qPrintable("load_training " + fileName + ".train"));

}

bool Game::saveTraining() {
     QTextStream(stdout) << "Saving " << m_fileName + ".train" << Qt::endl;
     return sendGtpCommand(qPrintable("save_training " + m_fileName + ".train"));
}


bool Game::loadSgf(const QString &fileName) {
    QTextStream(stdout) << "Loading " << fileName + ".sgf" << Qt::endl;
    return sendGtpCommand(qPrintable("loadsgf " + fileName + ".sgf"));
}

bool Game::loadSgf(const QString &fileName, const int moves) {
    QTextStream(stdout) << "Loading " << fileName + ".sgf with " << moves << " moves" << Qt::endl;
    return sendGtpCommand(qPrintable("loadsgf " + fileName + ".sgf " + QString::number(moves+1)));
}

bool Game::komi(float komi) {
    QTextStream(stdout) << "Setting komi " << komi << Qt::endl;
    return sendGtpCommand(qPrintable("komi " + QString::number(komi)));
}

void Game::fixSgfPlayer(QString& sgfData, const Engine& whiteEngine) {
    QRegularExpression oldPlayer("PW\\[Human\\]");
    QString playerName("PB[SAI ");
    QRegularExpression le("PB\\[SAI \\S+ ");
    QRegularExpressionMatch match = le.match(sgfData);
    if (match.hasMatch()) {
        playerName = match.captured(0);
    }
    playerName = "PW" + playerName.remove(0, 2);
    playerName += whiteEngine.getNetworkFile().left(8);
    playerName += "]";
    sgfData.replace(oldPlayer, playerName);
}

void Game::fixSgfComment(QString& sgfData, Game& whiteGame,
                         const bool isSelfPlay) {
    const Engine& whiteEngine = whiteGame.getEngine();
    // If this function is modified, a corresponding update is
    // required to SGFTree::state_to_string() in order to get the same
    // sgfhash as the server
    QRegularExpression oldComment("(C\\[SAI)( options:.*)\\]");
    QString comment("\\1");
    if (!isSelfPlay) {
        comment += " Black";
    }
    comment += "\\2 Starting GTP commands:";
    for (const auto& command : m_engine.m_commands) {
        comment += " " + command;
    }
    if (!isSelfPlay) {
        comment += " White options:";
        comment += whiteEngine.m_options + " " + whiteEngine.m_network;
        comment += " Starting GTP commands:";
        for (const auto& command : whiteEngine.m_commands) {
            comment += " " + command;
        }
    }
    comment += "]";
    comment.replace(QRegularExpression("\\s\\s+"), " ");
    sgfData.replace(oldComment, comment);
    if (!isSelfPlay) {
        QString whiteSgf;
        if (whiteGame.getSgf(whiteSgf)) {
            mergeSgfComments(sgfData, whiteSgf);
        }
    }
}

void Game::mergeSgfComments(QString& blackSgf, const QString& whiteSgf) const {
    QStringList blackMoves = blackSgf.split(";");
    QStringList whiteMoves = whiteSgf.split(";");
    for (int i = 3; i < blackMoves.size() && i < whiteMoves.size() ; i += 2) {
        // i=0 '('
        // i=1 header
        // i=2 move 1 by black
        blackMoves[i] = whiteMoves.at(i);
    }
    blackSgf = blackMoves.join(";");
}

bool Game::getSgf(QString& sgf) {
    writeSgf();
    QFile sgfFile(m_fileName + ".sgf");
    if (!sgfFile.open(QIODevice::Text | QIODevice::ReadOnly)) {
        return false;
    }
    sgf = sgfFile.readAll();
    sgfFile.close();
    QFile::remove(m_fileName + ".sgf");
    return true;
}

void Game::fixSgfResult(QString& sgfData, const bool resignation) {
    if (resignation) {
        QRegularExpression oldResult("RE\\[B\\+.*\\]");
        QString newResult("RE[B+Resign] ");
        sgfData.replace(oldResult, newResult);
        if (!sgfData.contains(newResult, Qt::CaseInsensitive)) {
            QRegularExpression oldwResult("RE\\[W\\+.*\\]");
            sgfData.replace(oldwResult, newResult);
        }
        QRegularExpression lastpass(";W\\[tt\\]\\)");
        QString noPass(")");
        sgfData.replace(lastpass, noPass);
    }
}

bool Game::fixSgf(Game& whiteGame, const bool resignation,
                  const bool isSelfPlay) {
    const Engine& whiteEngine = whiteGame.getEngine();
    QFile sgfFile(m_fileName + ".sgf");
    if (!sgfFile.open(QIODevice::Text | QIODevice::ReadOnly)) {
        return false;
    }
    QString sgfData = sgfFile.readAll();
    fixSgfPlayer(sgfData, whiteEngine);
    fixSgfComment(sgfData, whiteGame, isSelfPlay);
    fixSgfResult(sgfData, resignation);
    sgfFile.close();
    if (sgfFile.open(QFile::WriteOnly | QFile::Truncate)) {
        QTextStream out(&sgfFile);
        out << sgfData;
    }
    sgfFile.close();

    return true;
}

bool Game::dumpTraining() {
    return sendGtpCommand(
        qPrintable("dump_training " + m_winner + " " + m_fileName + ".txt"));
}

bool Game::dumpDebug() {
    return sendGtpCommand(
        qPrintable("dump_debug " + m_fileName + ".debug.txt"));
}

void Game::gameQuit() {
    write(qPrintable("quit\n"));
    waitForFinished(-1);
}
