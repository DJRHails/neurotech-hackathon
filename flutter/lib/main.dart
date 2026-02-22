import 'dart:math';

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

// ---------------------------------------------------------------------------
// Palette — "Clinical Luxe": dark matte surfaces with bioluminescent accents
// ---------------------------------------------------------------------------
class FlexPalette {
  FlexPalette._();

  static const bg = Color(0xFF0F0F1A);
  static const surface = Color(0xFF1A1A2E);
  static const surfaceLight = Color(0xFF252540);
  static const textPrimary = Color(0xFFF0EDE8);
  static const textSecondary = Color(0xFF9896A8);
  static const cardRed = Color(0xFFFF6B6B);
  static const cardBlue = Color(0xFF4ECDC4);
  static const cardGreen = Color(0xFFA8E06C);
  static const correct = Color(0xFF00E5A0);
  static const incorrect = Color(0xFFFF6B6B);
  static const amber = Color(0xFFFFB347);
}

// ---------------------------------------------------------------------------
// Theme builder
// ---------------------------------------------------------------------------
ThemeData flextraTheme() {
  final display = GoogleFonts.outfit();
  final body = GoogleFonts.dmSans();

  return ThemeData(
    brightness: Brightness.dark,
    scaffoldBackgroundColor: FlexPalette.bg,
    colorScheme: const ColorScheme.dark(
      primary: FlexPalette.amber,
      secondary: FlexPalette.cardBlue,
      surface: FlexPalette.surface,
    ),
    textTheme: TextTheme(
      displayLarge: display.copyWith(
        fontSize: 48,
        fontWeight: FontWeight.w700,
        letterSpacing: -1.5,
        color: FlexPalette.textPrimary,
      ),
      headlineMedium: display.copyWith(
        fontSize: 28,
        fontWeight: FontWeight.w600,
        color: FlexPalette.textPrimary,
      ),
      titleLarge: display.copyWith(
        fontSize: 20,
        fontWeight: FontWeight.w600,
        color: FlexPalette.textPrimary,
      ),
      titleMedium: body.copyWith(
        fontSize: 16,
        fontWeight: FontWeight.w500,
        color: FlexPalette.textPrimary,
      ),
      bodyLarge: body.copyWith(
        fontSize: 16,
        color: FlexPalette.textPrimary,
      ),
      bodyMedium: body.copyWith(
        fontSize: 14,
        color: FlexPalette.textSecondary,
      ),
      labelLarge: display.copyWith(
        fontSize: 14,
        fontWeight: FontWeight.w600,
        letterSpacing: 1.2,
        color: FlexPalette.textSecondary,
      ),
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: Colors.transparent,
      elevation: 0,
      centerTitle: true,
    ),
    cardTheme: CardThemeData(
      color: FlexPalette.surfaceLight,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20),
      ),
    ),
  );
}

// ---------------------------------------------------------------------------
// App entry
// ---------------------------------------------------------------------------
void main() {
  runApp(const FlextraApp());
}

class FlextraApp extends StatelessWidget {
  const FlextraApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flextra',
      debugShowCheckedModeBanner: false,
      theme: flextraTheme(),
      home: const SignInPage(),
    );
  }
}

// ---------------------------------------------------------------------------
// Sign-in / Role selection
// ---------------------------------------------------------------------------
class SignInPage extends StatefulWidget {
  const SignInPage({super.key});

  @override
  State<SignInPage> createState() => _SignInPageState();
}

class _SignInPageState extends State<SignInPage>
    with SingleTickerProviderStateMixin {
  late final AnimationController _anim;

  @override
  void initState() {
    super.initState();
    _anim = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..forward();
  }

  @override
  void dispose() {
    _anim.dispose();
    super.dispose();
  }

  void _navigate(Widget page) {
    Navigator.of(context).pushReplacement(
      PageRouteBuilder(
        pageBuilder: (_, __, ___) => page,
        transitionDuration: const Duration(milliseconds: 500),
        transitionsBuilder: (_, a, __, child) =>
            FadeTransition(opacity: a, child: child),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final tt = Theme.of(context).textTheme;

    return Scaffold(
      body: Center(
        child: FadeTransition(
          opacity: CurvedAnimation(
            parent: _anim,
            curve: Curves.easeOut,
          ),
          child: SlideTransition(
            position: Tween<Offset>(
              begin: const Offset(0, 0.08),
              end: Offset.zero,
            ).animate(CurvedAnimation(
              parent: _anim,
              curve: Curves.easeOutCubic,
            )),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 32),
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 420),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // Logo mark — neural dot cluster
                    _buildLogoMark(),
                    const SizedBox(height: 24),
                    Text('FLEXTRA', style: tt.displayLarge),
                    const SizedBox(height: 8),
                    Text(
                      'COGNITIVE ASSESSMENT',
                      style: tt.labelLarge,
                    ),
                    const SizedBox(height: 56),
                    _RoleCard(
                      icon: Icons.psychology_outlined,
                      label: 'Participant',
                      subtitle: 'Begin the card sorting task',
                      color: FlexPalette.cardBlue,
                      onTap: () => _navigate(
                        const GamePage(),
                      ),
                    ),
                    const SizedBox(height: 16),
                    _RoleCard(
                      icon: Icons.insights_outlined,
                      label: 'Researcher',
                      subtitle: 'View session analytics',
                      color: FlexPalette.amber,
                      onTap: () => _navigate(
                        const AdminDashboardPage(),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildLogoMark() {
    return SizedBox(
      width: 64,
      height: 64,
      child: CustomPaint(painter: _NeuralDotPainter()),
    );
  }
}

class _NeuralDotPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final paint = Paint()..style = PaintingStyle.fill;

    // Central dot
    paint.color = FlexPalette.amber;
    canvas.drawCircle(center, 6, paint);

    // Orbital dots
    final orbitals = [
      (FlexPalette.cardBlue, 22.0, -0.5),
      (FlexPalette.cardRed, 20.0, 1.8),
      (FlexPalette.cardGreen, 24.0, 3.6),
      (FlexPalette.textSecondary, 18.0, 5.0),
    ];

    final linePaint = Paint()
      ..color = FlexPalette.textSecondary.withValues(alpha: 0.25)
      ..strokeWidth = 1;

    for (final (color, radius, angle) in orbitals) {
      final pos = Offset(
        center.dx + radius * cos(angle),
        center.dy + radius * sin(angle),
      );
      canvas.drawLine(center, pos, linePaint);
      paint.color = color;
      canvas.drawCircle(pos, 4, paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _RoleCard extends StatefulWidget {
  final IconData icon;
  final String label;
  final String subtitle;
  final Color color;
  final VoidCallback onTap;

  const _RoleCard({
    required this.icon,
    required this.label,
    required this.subtitle,
    required this.color,
    required this.onTap,
  });

  @override
  State<_RoleCard> createState() => _RoleCardState();
}

class _RoleCardState extends State<_RoleCard> {
  bool _hovering = false;

  @override
  Widget build(BuildContext context) {
    final tt = Theme.of(context).textTheme;

    return MouseRegion(
      onEnter: (_) => setState(() => _hovering = true),
      onExit: (_) => setState(() => _hovering = false),
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 250),
          curve: Curves.easeOut,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
          decoration: BoxDecoration(
            color: _hovering
                ? widget.color.withValues(alpha: 0.12)
                : FlexPalette.surfaceLight,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: _hovering
                  ? widget.color.withValues(alpha: 0.5)
                  : FlexPalette.surfaceLight,
              width: 1.5,
            ),
            boxShadow: _hovering
                ? [
                    BoxShadow(
                      color: widget.color.withValues(alpha: 0.15),
                      blurRadius: 24,
                      spreadRadius: 0,
                    ),
                  ]
                : [],
          ),
          child: Row(
            children: [
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: widget.color.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(
                  widget.icon,
                  color: widget.color,
                  size: 24,
                ),
              ),
              const SizedBox(width: 20),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(widget.label, style: tt.titleLarge),
                    const SizedBox(height: 2),
                    Text(widget.subtitle, style: tt.bodyMedium),
                  ],
                ),
              ),
              Icon(
                Icons.arrow_forward_rounded,
                color: FlexPalette.textSecondary,
                size: 20,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Data models & enums
// ---------------------------------------------------------------------------
enum CardColorOption { red, blue, green }

enum CardShapeOption { circle, square, triangle }

enum MatchRule { color, shape, count }

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

const kKeyCards = [
  GameCard(color: CardColorOption.red, shape: CardShapeOption.circle, count: 1),
  GameCard(color: CardColorOption.blue, shape: CardShapeOption.square, count: 2),
  GameCard(
    color: CardColorOption.green,
    shape: CardShapeOption.triangle,
    count: 3,
  ),
];

// ---------------------------------------------------------------------------
// Game page — classic WCST layout
// ---------------------------------------------------------------------------
class GamePage extends StatefulWidget {
  const GamePage({super.key});

  @override
  State<GamePage> createState() => _GamePageState();
}

class _GamePageState extends State<GamePage>
    with TickerProviderStateMixin {
  final Random _random = Random(42);

  late List<MatchRule> _ruleOrder;
  int _ruleIndex = 0;
  late MatchRule _currentRule;
  late GameCard _responseCard;

  int _streak = 0;
  int _totalTrials = 0;
  int _totalCorrect = 0;

  bool? _lastCorrect;

  WebSocketChannel? _channel;
  String? _lastServerMessage;

  late AnimationController _feedbackAnim;
  late AnimationController _cardEntryAnim;

  @override
  void initState() {
    super.initState();
    _ruleOrder = List.of(MatchRule.values)..shuffle(_random);
    _currentRule = _ruleOrder[_ruleIndex];
    _responseCard = _generateResponseCard();
    _connectToWebSocket();

    _feedbackAnim = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    );
    _cardEntryAnim = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 400),
    )..forward();
  }

  @override
  void dispose() {
    _channel?.sink.close();
    _feedbackAnim.dispose();
    _cardEntryAnim.dispose();
    super.dispose();
  }

  void _connectToWebSocket() {
    try {
      _channel = WebSocketChannel.connect(
        Uri.parse('ws://127.0.0.1:9000'),
      );
      _channel!.stream.listen(
        (message) {
          setState(() => _lastServerMessage = message.toString());
        },
        onError: (error) => debugPrint('WebSocket error: $error'),
      );
    } catch (e) {
      debugPrint('Failed to connect to WebSocket: $e');
    }
  }

  // ---- Game logic ----

  void _advanceRule() {
    _ruleIndex = (_ruleIndex + 1) % _ruleOrder.length;
    _currentRule = _ruleOrder[_ruleIndex];
    _streak = 0;
  }

  GameCard _generateResponseCard() {
    // Generate a card where at least 2 dimensions point to
    // different key cards, avoiding trivial all-same matches.
    while (true) {
      final color = CardColorOption.values[_random.nextInt(3)];
      final shape = CardShapeOption.values[_random.nextInt(3)];
      final count = _random.nextInt(3) + 1;
      final ci = color.index;
      final si = shape.index;
      final ni = count - 1;
      if (ci != si || si != ni) {
        return GameCard(color: color, shape: shape, count: count);
      }
    }
  }

  int _correctKeyIndex() {
    switch (_currentRule) {
      case MatchRule.color:
        return _responseCard.color.index;
      case MatchRule.shape:
        return _responseCard.shape.index;
      case MatchRule.count:
        return _responseCard.count - 1;
    }
  }

  void _onOptionTapped(int keyIndex) {
    final correct = keyIndex == _correctKeyIndex();

    setState(() {
      _lastCorrect = correct;
      _totalTrials++;
      if (correct) {
        _totalCorrect++;
        _streak++;
      } else {
        _streak = 0;
      }

      if (_streak >= 10) {
        _advanceRule();
      }

      _responseCard = _generateResponseCard();
    });

    debugPrint('Current rule: ${_currentRule.name}');

    _feedbackAnim.forward(from: 0);
    _cardEntryAnim.forward(from: 0);
  }

  // ---- Visual helpers ----

  Color _mapColor(CardColorOption c) {
    switch (c) {
      case CardColorOption.red:
        return FlexPalette.cardRed;
      case CardColorOption.blue:
        return FlexPalette.cardBlue;
      case CardColorOption.green:
        return FlexPalette.cardGreen;
    }
  }

  IconData _mapShapeIcon(CardShapeOption s) {
    switch (s) {
      case CardShapeOption.circle:
        return Icons.circle;
      case CardShapeOption.square:
        return Icons.crop_square;
      case CardShapeOption.triangle:
        return Icons.change_history;
    }
  }

  // ---- Build ----

  @override
  Widget build(BuildContext context) {
    final tt = Theme.of(context).textTheme;

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            _buildTopBar(tt),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      'Match this card',
                      style: tt.bodyMedium,
                    ),
                    const SizedBox(height: 16),
                    _buildReferenceCard(),
                    const SizedBox(height: 32),
                    Text(
                      'to one of these',
                      style: tt.bodyMedium,
                    ),
                    const SizedBox(height: 16),
                    _buildOptionCards(),
                    const SizedBox(height: 24),
                    _buildFeedback(tt),
                    if (_lastServerMessage != null) ...[
                      const SizedBox(height: 16),
                      _buildServerMessage(tt),
                    ],
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTopBar(TextTheme tt) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(8, 8, 16, 0),
      child: Row(
        children: [
          IconButton(
            icon: const Icon(
              Icons.arrow_back_rounded,
              color: FlexPalette.textSecondary,
            ),
            onPressed: () => Navigator.of(context).pushReplacement(
              MaterialPageRoute(builder: (_) => const SignInPage()),
            ),
          ),
          const Spacer(),
          Container(
            padding: const EdgeInsets.symmetric(
              horizontal: 14,
              vertical: 6,
            ),
            decoration: BoxDecoration(
              color: _streak > 0
                  ? FlexPalette.correct.withValues(alpha: 0.12)
                  : FlexPalette.surfaceLight,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                color: _streak > 0
                    ? FlexPalette.correct.withValues(alpha: 0.3)
                    : Colors.transparent,
              ),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  Icons.local_fire_department_rounded,
                  size: 16,
                  color: _streak > 0
                      ? FlexPalette.correct
                      : FlexPalette.textSecondary,
                ),
                const SizedBox(width: 6),
                Text(
                  '$_streak',
                  style: tt.titleMedium?.copyWith(
                    color: _streak > 0
                        ? FlexPalette.correct
                        : FlexPalette.textSecondary,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 8),
          Text(
            'Trial $_totalTrials',
            style: tt.bodyMedium,
          ),
        ],
      ),
    );
  }

  Widget _buildReferenceCard() {
    final color = _mapColor(_responseCard.color);

    return FadeTransition(
      opacity: CurvedAnimation(
        parent: _cardEntryAnim,
        curve: Curves.easeOut,
      ),
      child: Container(
        height: 160,
        width: double.infinity,
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.08),
          borderRadius: BorderRadius.circular(24),
          border: Border.all(
            color: color.withValues(alpha: 0.4),
            width: 2,
          ),
          boxShadow: [
            BoxShadow(
              color: color.withValues(alpha: 0.15),
              blurRadius: 40,
              spreadRadius: 0,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: Center(
          child: Wrap(
            alignment: WrapAlignment.center,
            spacing: 16,
            runSpacing: 16,
            children: List.generate(
              _responseCard.count,
              (_) => Icon(
                _mapShapeIcon(_responseCard.shape),
                size: 48,
                color: color,
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildOptionCards() {
    return Row(
      children: List.generate(kKeyCards.length, (i) {
        final card = kKeyCards[i];
        final color = _mapColor(card.color);
        final isLast = i == kKeyCards.length - 1;

        return Expanded(
          child: Padding(
            padding: EdgeInsets.only(right: isLast ? 0 : 12),
            child: GestureDetector(
              onTap: () => _onOptionTapped(i),
              child: Container(
                height: 140,
                decoration: BoxDecoration(
                  color: FlexPalette.surfaceLight,
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                    color: color.withValues(alpha: 0.25),
                    width: 1.5,
                  ),
                ),
                child: Center(
                  child: Wrap(
                    alignment: WrapAlignment.center,
                    spacing: 10,
                    runSpacing: 10,
                    children: List.generate(
                      card.count,
                      (_) => Icon(
                        _mapShapeIcon(card.shape),
                        size: 32,
                        color: color,
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        );
      }),
    );
  }

  Widget _buildFeedback(TextTheme tt) {
    if (_lastCorrect == null) {
      return const SizedBox(height: 40);
    }

    final isCorrect = _lastCorrect!;
    final color = isCorrect ? FlexPalette.correct : FlexPalette.incorrect;

    return FadeTransition(
      opacity: CurvedAnimation(
        parent: _feedbackAnim,
        curve: Curves.easeOut,
      ),
      child: Container(
        height: 40,
        padding: const EdgeInsets.symmetric(horizontal: 20),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              isCorrect
                  ? Icons.check_rounded
                  : Icons.close_rounded,
              color: color,
              size: 20,
            ),
            const SizedBox(width: 8),
            Text(
              isCorrect ? 'Correct' : 'Try again',
              style: tt.titleMedium?.copyWith(
                color: color,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildServerMessage(TextTheme tt) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: FlexPalette.surfaceLight,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          const Icon(
            Icons.sensors_rounded,
            size: 16,
            color: FlexPalette.amber,
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              _lastServerMessage!,
              style: tt.bodyMedium?.copyWith(
                fontFamily: 'monospace',
                fontSize: 12,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Admin dashboard
// ---------------------------------------------------------------------------
class AdminDashboardPage extends StatelessWidget {
  const AdminDashboardPage({super.key});

  static List<double> _generateErpHitRates() {
    final rng = Random(42);
    final data = <double>[];
    double value = 0.82;
    for (int i = 0; i < 30; i++) {
      final noise = (rng.nextDouble() - 0.45) * 0.12;
      value = (value + noise - 0.008).clamp(0.15, 0.95);
      data.add(value);
    }
    return data;
  }

  @override
  Widget build(BuildContext context) {
    final tt = Theme.of(context).textTheme;
    final erpData = _generateErpHitRates();
    final latestRate = (erpData.last * 100).toStringAsFixed(1);

    return Scaffold(
      body: SafeArea(
        child: CustomScrollView(
          slivers: [
            SliverToBoxAdapter(child: _buildHeader(context, tt)),
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(24, 0, 24, 32),
              sliver: SliverList(
                delegate: SliverChildListDelegate([
                  _buildMetricRow(tt, latestRate),
                  const SizedBox(height: 32),
                  _buildErpChart(tt, erpData),
                  const SizedBox(height: 32),
                  Text(
                    'PLANNED METRICS',
                    style: tt.labelLarge,
                  ),
                  const SizedBox(height: 16),
                  _buildFutureMetric(
                    tt,
                    Icons.timer_outlined,
                    'Rule acquisition latency',
                    'Average trials to learn each rule type',
                    FlexPalette.cardBlue,
                  ),
                  const SizedBox(height: 12),
                  _buildFutureMetric(
                    tt,
                    Icons.category_outlined,
                    'Error distribution',
                    'Breakdown by rule type: color, shape, quantity',
                    FlexPalette.cardRed,
                  ),
                  const SizedBox(height: 12),
                  _buildFutureMetric(
                    tt,
                    Icons.speed_outlined,
                    'Response latency',
                    'Reaction time per trial with trend analysis',
                    FlexPalette.cardGreen,
                  ),
                  const SizedBox(height: 12),
                  _buildFutureMetric(
                    tt,
                    Icons.timeline_outlined,
                    'Perseverative errors',
                    'Continued use of previous rule after switch',
                    FlexPalette.amber,
                  ),
                ]),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context, TextTheme tt) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(8, 8, 24, 32),
      child: Row(
        children: [
          IconButton(
            icon: const Icon(
              Icons.arrow_back_rounded,
              color: FlexPalette.textSecondary,
            ),
            onPressed: () => Navigator.of(context).pushReplacement(
              MaterialPageRoute(builder: (_) => const SignInPage()),
            ),
          ),
          const SizedBox(width: 8),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('ANALYTICS', style: tt.labelLarge),
              const SizedBox(height: 2),
              Text('Session Overview', style: tt.headlineMedium),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMetricRow(TextTheme tt, String latestErp) {
    return Row(
      children: [
        Expanded(
          child: _buildStatCard(
            tt,
            label: 'Trials',
            value: '147',
            color: FlexPalette.cardBlue,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _buildStatCard(
            tt,
            label: 'Accuracy',
            value: '68 %',
            color: FlexPalette.correct,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _buildStatCard(
            tt,
            label: 'ERP Hit',
            value: '$latestErp%',
            color: FlexPalette.amber,
          ),
        ),
      ],
    );
  }

  Widget _buildErpChart(TextTheme tt, List<double> data) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: FlexPalette.surfaceLight,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(
                Icons.show_chart_rounded,
                size: 16,
                color: FlexPalette.amber,
              ),
              const SizedBox(width: 8),
              Text('ERP HIT RATE', style: tt.labelLarge),
              const Spacer(),
              Text('30 blocks', style: tt.bodyMedium),
            ],
          ),
          const SizedBox(height: 20),
          SizedBox(
            height: 140,
            child: CustomPaint(
              size: const Size(double.infinity, 140),
              painter: _ErpChartPainter(data),
            ),
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('Block 1', style: tt.bodyMedium),
              Text('Block ${data.length}', style: tt.bodyMedium),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildStatCard(
    TextTheme tt, {
    required String label,
    required String value,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: color.withValues(alpha: 0.2),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: tt.bodyMedium),
          const SizedBox(height: 8),
          Text(
            value,
            style: tt.headlineMedium?.copyWith(
              color: color,
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFutureMetric(
    TextTheme tt,
    IconData icon,
    String title,
    String description,
    Color color,
  ) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: FlexPalette.surfaceLight,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        children: [
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(icon, color: color, size: 20),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: tt.titleMedium),
                const SizedBox(height: 2),
                Text(description, style: tt.bodyMedium),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _ErpChartPainter extends CustomPainter {
  final List<double> data;

  _ErpChartPainter(this.data);

  @override
  void paint(Canvas canvas, Size size) {
    if (data.length < 2) return;

    const yMin = 0.0;
    const yMax = 1.0;
    final n = data.length;

    Offset pointAt(int i) {
      final x = i / (n - 1) * size.width;
      final y = size.height - ((data[i] - yMin) / (yMax - yMin)) * size.height;
      return Offset(x, y);
    }

    // Grid lines at 25%, 50%, 75%
    final gridPaint = Paint()
      ..color = FlexPalette.textSecondary.withValues(alpha: 0.1)
      ..strokeWidth = 1;
    for (final frac in [0.25, 0.5, 0.75]) {
      final y = size.height - frac * size.height;
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }

    // Gradient fill under the line
    final fillPath = Path()..moveTo(0, size.height);
    for (int i = 0; i < n; i++) {
      final p = pointAt(i);
      fillPath.lineTo(p.dx, p.dy);
    }
    fillPath.lineTo(size.width, size.height);
    fillPath.close();

    final fillPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          FlexPalette.amber.withValues(alpha: 0.2),
          FlexPalette.amber.withValues(alpha: 0.0),
        ],
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height));
    canvas.drawPath(fillPath, fillPaint);

    // Line
    final linePath = Path();
    for (int i = 0; i < n; i++) {
      final p = pointAt(i);
      if (i == 0) {
        linePath.moveTo(p.dx, p.dy);
      } else {
        linePath.lineTo(p.dx, p.dy);
      }
    }
    final linePaint = Paint()
      ..color = FlexPalette.amber
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    canvas.drawPath(linePath, linePaint);

    // Dots at each data point
    final dotPaint = Paint()
      ..color = FlexPalette.amber
      ..style = PaintingStyle.fill;
    final dotBgPaint = Paint()
      ..color = FlexPalette.surfaceLight
      ..style = PaintingStyle.fill;
    for (int i = 0; i < n; i++) {
      final p = pointAt(i);
      canvas.drawCircle(p, 4, dotBgPaint);
      canvas.drawCircle(p, 2.5, dotPaint);
    }
  }

  @override
  bool shouldRepaint(covariant _ErpChartPainter old) => old.data != data;
}
