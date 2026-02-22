import 'dart:math';
import 'package:web_socket_channel/web_socket_channel.dart';

import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flextra',
      theme: ThemeData(
        // This is the theme of your application.
        //
        // TRY THIS: Try running your application with "flutter run". You'll see
        // the application has a purple toolbar. Then, without quitting the app,
        // try changing the seedColor in the colorScheme below to Colors.green
        // and then invoke "hot reload" (save your changes or press the "hot
        // reload" button in a Flutter-supported IDE, or press "r" if you used
        // the command line to start the app).
        //
        // Notice that the counter didn't reset back to zero; the application
        // state is not lost during the reload. To reset the state, use hot
        // restart instead.
        //
        // This works for code too, not just values: Most code changes can be
        // tested with just a hot reload.
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
      ),
      home: const MyHomePage(title: 'Flextra'),
    );
  }
}

class AdminDashboardPage extends StatelessWidget {
  const AdminDashboardPage({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        leading: PopupMenuButton<String>(
          icon: const Icon(Icons.account_circle),
          onSelected: (value) {
            if (value == 'logout') {
              Navigator.of(context).pushReplacement(
                MaterialPageRoute(builder: (_) => const SignInPage()),
              );
            }
          },
          itemBuilder: (context) => const [
            PopupMenuItem<String>(value: 'logout', child: Text('Log out')),
          ],
        ),
        title: const Text('Administrator dashboard'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: ListView(
          children: [
            Text(
              'Session statistics (placeholder)',
              style: theme.textTheme.headlineSmall,
            ),
            const SizedBox(height: 24),
            Card(
              child: ListTile(
                title: const Text('Total trials'),
                subtitle: const Text('Number of choices made across sessions.'),
                trailing: const Text('—'), // placeholder
              ),
            ),
            const SizedBox(height: 12),
            Card(
              child: ListTile(
                title: const Text('Overall accuracy'),
                subtitle: const Text('Percentage of correct choices.'),
                trailing: const Text('— %'),
              ),
            ),
            const SizedBox(height: 12),
            Card(
              child: ListTile(
                title: const Text('Current rule streak'),
                subtitle: const Text(
                  'How many correct in a row for the current rule.',
                ),
                trailing: const Text('—'),
              ),
            ),
            const SizedBox(height: 24),
            Text(
              'Future metrics could include:',
              style: theme.textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            const Text('• Time to learn each rule'),
            const Text('• Errors by rule type (color vs shape vs quantity)'),
            const Text('• Response times per trial'),
          ],
        ),
      ),
    );
  }
}

enum CardColorOption { red, blue, green }

enum CardShapeOption { circle, square, triangle }

enum RuleType {
  // Single-feature rules
  pickRed,
  pickBlue,
  pickGreen,
  moreShapes,
  fewerShapes,
  circles,
  squares,
  triangles,

  // More difficult: combined feature rules
  redCircle,
  blueSquare,
  greenTriangle,
  redMoreShapes,
  circleFewerShapes,
}

class GameCard {
  final CardColorOption color;
  final CardShapeOption shape;
  final int count;

  const GameCard({
    required this.color,
    required this.shape,
    required this.count,
  });
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final Random _random = Random();

  late RuleType _currentRule;
  late List<GameCard> _cards; // always length 2

  int _streak = 0;
  final int _requiredStreak =
      5; // minimum correct answers in a row before rule may change

  int _trialsThisRule = 0; // number of trials under the current hidden rule

  bool? _lastCorrect; // null = no answer yet
  GameCard? _previousCorrectCard; // last correctly selected card

  WebSocketChannel? _channel;
  String? _lastServerMessage;

  @override
  void initState() {
    super.initState();
    _currentRule = _randomRule();
    _cards = _generateCardsForRule(_currentRule);
    _connectToWebSocket();
  }

  @override
  void dispose() {
    _channel?.sink.close();
    super.dispose();
  }

  void _connectToWebSocket() {
    try {
      _channel = WebSocketChannel.connect(Uri.parse('ws://127.0.0.1:9000'));
      _channel!.stream.listen(
        (message) {
          setState(() {
            _lastServerMessage = message.toString();
          });
        },
        onError: (error) {
          debugPrint('WebSocket error: $error');
        },
      );
    } catch (e) {
      debugPrint('Failed to connect to WebSocket: $e');
    }
  }

  RuleType _randomRule() {
    final rules = RuleType.values;
    return rules[_random.nextInt(rules.length)];
  }

  List<GameCard> _generateCardsForRule(RuleType rule) {
    // We construct the two cards so that exactly one of them satisfies the rule.
    GameCard correct;
    GameCard other;

    CardColorOption randomColor() =>
        CardColorOption.values[_random.nextInt(CardColorOption.values.length)];
    CardShapeOption randomShape() =>
        CardShapeOption.values[_random.nextInt(CardShapeOption.values.length)];

    switch (rule) {
      case RuleType.pickRed:
        correct = GameCard(
          color: CardColorOption.red,
          shape: randomShape(),
          count: _random.nextInt(3) + 1,
        );
        other = GameCard(
          color: CardColorOption.blue,
          shape: randomShape(),
          count: correct.count,
        );
        break;
      case RuleType.pickBlue:
        correct = GameCard(
          color: CardColorOption.blue,
          shape: randomShape(),
          count: _random.nextInt(3) + 1,
        );
        other = GameCard(
          color: CardColorOption.red,
          shape: randomShape(),
          count: correct.count,
        );
        break;
      case RuleType.pickGreen:
        correct = GameCard(
          color: CardColorOption.green,
          shape: randomShape(),
          count: _random.nextInt(3) + 1,
        );
        other = GameCard(
          color: CardColorOption.red,
          shape: randomShape(),
          count: correct.count,
        );
        break;
      case RuleType.moreShapes:
        final baseShape = randomShape();
        final baseColor = randomColor();
        final smallCount = _random.nextInt(2) + 1; // 1-2
        final bigCount = smallCount + _random.nextInt(2) + 1; // 2-4 and > small
        correct = GameCard(color: baseColor, shape: baseShape, count: bigCount);
        other = GameCard(color: baseColor, shape: baseShape, count: smallCount);
        break;
      case RuleType.fewerShapes:
        final baseShape = randomShape();
        final baseColor = randomColor();
        final smallCount = _random.nextInt(2) + 1;
        final bigCount = smallCount + _random.nextInt(2) + 1;
        correct = GameCard(
          color: baseColor,
          shape: baseShape,
          count: smallCount,
        );
        other = GameCard(color: baseColor, shape: baseShape, count: bigCount);
        break;
      case RuleType.circles:
        final baseColor = randomColor();
        final count = _random.nextInt(3) + 1;
        correct = GameCard(
          color: baseColor,
          shape: CardShapeOption.circle,
          count: count,
        );
        other = GameCard(
          color: baseColor,
          shape: CardShapeOption.square,
          count: count,
        );
        break;
      case RuleType.squares:
        final baseColor = randomColor();
        final count = _random.nextInt(3) + 1;
        correct = GameCard(
          color: baseColor,
          shape: CardShapeOption.square,
          count: count,
        );
        other = GameCard(
          color: baseColor,
          shape: CardShapeOption.triangle,
          count: count,
        );
        break;
      case RuleType.triangles:
        final baseColor = randomColor();
        final count = _random.nextInt(3) + 1;
        correct = GameCard(
          color: baseColor,
          shape: CardShapeOption.triangle,
          count: count,
        );
        other = GameCard(
          color: baseColor,
          shape: CardShapeOption.circle,
          count: count,
        );
        break;

      // Combined rules: enforce multiple feature constraints for the correct card
      case RuleType.redCircle:
        correct = GameCard(
          color: CardColorOption.red,
          shape: CardShapeOption.circle,
          count: _random.nextInt(3) + 1,
        );
        // Make sure the other differs in at least one feature
        other = GameCard(
          color: CardColorOption.blue,
          shape: CardShapeOption.square,
          count: correct.count,
        );
        break;
      case RuleType.blueSquare:
        correct = GameCard(
          color: CardColorOption.blue,
          shape: CardShapeOption.square,
          count: _random.nextInt(3) + 1,
        );
        other = GameCard(
          color: CardColorOption.green,
          shape: CardShapeOption.circle,
          count: correct.count,
        );
        break;
      case RuleType.greenTriangle:
        correct = GameCard(
          color: CardColorOption.green,
          shape: CardShapeOption.triangle,
          count: _random.nextInt(3) + 1,
        );
        other = GameCard(
          color: CardColorOption.red,
          shape: CardShapeOption.circle,
          count: correct.count,
        );
        break;
      case RuleType.redMoreShapes:
        final baseShape = randomShape();
        final smallCount = _random.nextInt(2) + 1;
        final bigCount = smallCount + _random.nextInt(2) + 1;
        correct = GameCard(
          color: CardColorOption.red,
          shape: baseShape,
          count: bigCount,
        );
        other = GameCard(
          color: CardColorOption.blue,
          shape: baseShape,
          count: smallCount,
        );
        break;
      case RuleType.circleFewerShapes:
        final baseColor = randomColor();
        final smallCount = _random.nextInt(2) + 1;
        final bigCount = smallCount + _random.nextInt(2) + 1;
        correct = GameCard(
          color: baseColor,
          shape: CardShapeOption.circle,
          count: smallCount,
        );
        other = GameCard(
          color: baseColor,
          shape: CardShapeOption.square,
          count: bigCount,
        );
        break;
    }

    // Randomize which side is correct.
    if (_random.nextBool()) {
      return [correct, other];
    } else {
      return [other, correct];
    }
  }

  bool _isCorrectChoice(int index) {
    final left = _cards[0];
    final right = _cards[1];

    bool matchesRule(GameCard card) {
      switch (_currentRule) {
        case RuleType.pickRed:
          return card.color == CardColorOption.red;
        case RuleType.pickBlue:
          return card.color == CardColorOption.blue;
        case RuleType.pickGreen:
          return card.color == CardColorOption.green;
        case RuleType.moreShapes:
          return card.count == max(left.count, right.count);
        case RuleType.fewerShapes:
          return card.count == min(left.count, right.count);
        case RuleType.circles:
          return card.shape == CardShapeOption.circle;
        case RuleType.squares:
          return card.shape == CardShapeOption.square;
        case RuleType.triangles:
          return card.shape == CardShapeOption.triangle;
        case RuleType.redCircle:
          return card.color == CardColorOption.red &&
              card.shape == CardShapeOption.circle;
        case RuleType.blueSquare:
          return card.color == CardColorOption.blue &&
              card.shape == CardShapeOption.square;
        case RuleType.greenTriangle:
          return card.color == CardColorOption.green &&
              card.shape == CardShapeOption.triangle;
        case RuleType.redMoreShapes:
          return card.color == CardColorOption.red &&
              card.count == max(left.count, right.count);
        case RuleType.circleFewerShapes:
          return card.shape == CardShapeOption.circle &&
              card.count == min(left.count, right.count);
      }
    }

    return matchesRule(_cards[index]);
  }

  void _onCardTapped(int index) {
    final correct = _isCorrectChoice(index);

    setState(() {
      _lastCorrect = correct;
      if (correct) {
        _previousCorrectCard = _cards[index];
        _streak++;
      } else {
        _streak = 0;
        _previousCorrectCard = null;
      }

      // Wisconsin-style progression: keep the same rule for a fixed
      // number of trials, then switch to a new rule.
      _trialsThisRule++;
      if (_trialsThisRule >= 10) {
        _trialsThisRule = 0;
        final oldRule = _currentRule;
        // Ensure we actually change to a different rule if possible.
        do {
          _currentRule = _randomRule();
        } while (_currentRule == oldRule && RuleType.values.length > 1);
        _streak = 0;
      }

      _cards = _generateCardsForRule(_currentRule);
    });
  }

  Color _mapCardColor(CardColorOption color) {
    switch (color) {
      case CardColorOption.red:
        return Colors.red.shade400;
      case CardColorOption.blue:
        return Colors.blue.shade400;
      case CardColorOption.green:
        return Colors.green.shade400;
    }
  }

  IconData _mapShapeIcon(CardShapeOption shape) {
    switch (shape) {
      case CardShapeOption.circle:
        return Icons.circle;
      case CardShapeOption.square:
        return Icons.crop_square;
      case CardShapeOption.triangle:
        return Icons.change_history; // triangle icon
    }
  }

  Widget _buildCard(GameCard card, int index) {
    return GestureDetector(
      onTap: () => _onCardTapped(index),
      child: Card(
        elevation: 6,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
        child: Container(
          height: 220,
          width: double.infinity,
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            color: _mapCardColor(card.color).withOpacity(0.18),
            borderRadius: BorderRadius.circular(24),
            border: Border.all(color: _mapCardColor(card.color), width: 3),
          ),
          child: Center(
            child: Wrap(
              alignment: WrapAlignment.center,
              spacing: 12,
              runSpacing: 12,
              children: List.generate(
                card.count,
                (i) => Icon(
                  _mapShapeIcon(card.shape),
                  size: 40,
                  color: _mapCardColor(card.color),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildReferenceCard(GameCard card) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Container(
        height: 120,
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: _mapCardColor(card.color).withOpacity(0.12),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: _mapCardColor(card.color), width: 2),
        ),
        child: Center(
          child: Wrap(
            alignment: WrapAlignment.center,
            spacing: 8,
            runSpacing: 8,
            children: List.generate(
              card.count,
              (i) => Icon(
                _mapShapeIcon(card.shape),
                size: 28,
                color: _mapCardColor(card.color),
              ),
            ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    // This method is rerun every time setState is called, for instance as done
    // by the _incrementCounter method above.
    //
    // The Flutter framework has been optimized to make rerunning build methods
    // fast, so that you can just rebuild anything that needs updating rather
    // than having to individually change instances of widgets.
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        leading: PopupMenuButton<String>(
          icon: const Icon(Icons.account_circle),
          onSelected: (value) {
            if (value == 'logout') {
              Navigator.of(context).pushReplacement(
                MaterialPageRoute(builder: (_) => const SignInPage()),
              );
            }
          },
          itemBuilder: (context) => const [
            PopupMenuItem<String>(value: 'logout', child: Text('Log out')),
          ],
        ),
        title: Text(widget.title),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            const Text(
              'Figure out the hidden rule by tapping one of the cards.',
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            Row(
              children: [
                Expanded(child: _buildCard(_cards[0], 0)),
                const SizedBox(width: 16),
                Expanded(child: _buildCard(_cards[1], 1)),
              ],
            ),
            const SizedBox(height: 24),
            if (_lastCorrect != null)
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    _lastCorrect! ? Icons.check_circle : Icons.cancel,
                    color: _lastCorrect! ? Colors.green : Colors.red,
                    size: 40,
                  ),
                  const SizedBox(width: 8),
                  Text(
                    _lastCorrect! ? 'Correct' : 'Try again',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ],
              ),
            const SizedBox(height: 16),
            if (_previousCorrectCard != null && _lastCorrect == true) ...[
              Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  'Previous correct card:',
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ),
              const SizedBox(height: 8),
              _buildReferenceCard(_previousCorrectCard!),
            ],
            const SizedBox(height: 16),
            if (_lastServerMessage != null) ...[
              Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  'Message from server:',
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                _lastServerMessage!,
                style: Theme.of(context).textTheme.bodyLarge,
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class SignInPage extends StatelessWidget {
  const SignInPage({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(title: const Text('Choose role')),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Center(
          child: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  'Select how you want to sign in',
                  style: theme.textTheme.headlineSmall,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 32),
                SizedBox(
                  height: 80,
                  child: ElevatedButton.icon(
                    icon: const Icon(Icons.person, size: 32),
                    label: const Text('User', style: TextStyle(fontSize: 20)),
                    onPressed: () {
                      Navigator.of(context).pushReplacement(
                        MaterialPageRoute(
                          builder: (_) => const MyHomePage(title: 'Flextra'),
                        ),
                      );
                    },
                  ),
                ),
                const SizedBox(height: 16),
                SizedBox(
                  height: 80,
                  child: OutlinedButton.icon(
                    icon: const Icon(Icons.admin_panel_settings, size: 32),
                    label: const Text(
                      'Administrator',
                      style: TextStyle(fontSize: 20),
                    ),
                    onPressed: () {
                      Navigator.of(context).pushReplacement(
                        MaterialPageRoute(
                          builder: (_) => const AdminDashboardPage(),
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
