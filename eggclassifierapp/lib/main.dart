import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper_plus/tflite_flutter_helper_plus.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(EggClassifierApp());
}

class EggClassifierApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Egg Classifier',
      theme: ThemeData(
        brightness: Brightness.dark,
        primaryColor: Color(0xFF4CAF50),
        colorScheme: ColorScheme.dark(
          primary: Color(0xFF4CAF50),
          secondary: Color(0xFF81C784),
          surface: Color(0xFF121212),
          background: Color(0xFF121212),
          onBackground: Colors.white,
        ),
        scaffoldBackgroundColor: Color(0xFF121212),
        appBarTheme: AppBarTheme(
          backgroundColor: Color(0xFF1E1E1E),
          elevation: 0,
          titleTextStyle: TextStyle(
            color: Colors.white,
            fontSize: 20,
            fontFamily: 'Consolas', // Tambahkan fontFamily di sini

            fontWeight: FontWeight.bold,
          ),
        ),
        cardTheme: CardTheme(
          color: Color(0xFF1E1E1E),
          elevation: 4,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: Color(0xFF4CAF50),
            foregroundColor: Colors.white,
            padding: EdgeInsets.symmetric(horizontal: 24, vertical: 12),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        ),
      ),
      home: EggClassifierScreen(),
    );
  }
}

class EggClassifierScreen extends StatefulWidget {
  @override
  _EggClassifierScreenState createState() => _EggClassifierScreenState();
}

class _EggClassifierScreenState extends State<EggClassifierScreen> {
  File? _image;
  final ImagePicker _picker = ImagePicker();
  Interpreter? _interpreter;
  String _result = "No image selected";
  double _probability = 0.0;
  bool _isModelLoaded = false;
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    _loadModel().then((_) {
      setState(() {
        _isModelLoaded = true;
      });
    });
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/final_model.tflite');
      print("Model loaded successfully");
    } catch (e) {
      print("Error loading model: $e");
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    if (!_isModelLoaded) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Model is still loading. Please wait.")),
      );
      return;
    }

    final pickedFile = await _picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _isProcessing = true;
        _result = "Processing...";
      });
      await _classifyImage(_image!);
      setState(() {
        _isProcessing = false;
      });
    }
  }

  Future<void> _classifyImage(File image) async {
    if (_interpreter == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Model is not initialized. Please wait.")),
      );
      return;
    }

    try {
      final rawImage = image.readAsBytesSync();
      final decodedImage = img.decodeImage(rawImage);

      if (decodedImage == null) {
        print("Failed to decode image");
        setState(() {
          _result = "Failed to process image";
          _probability = 0.0;
        });
        return;
      }

      final resizedImage = img.copyResize(
        decodedImage,
        width: 150,
        height: 150,
      );
      final inputImage = TensorImage.fromImage(resizedImage);

      final processor =
          TensorProcessorBuilder().add(NormalizeOp(0, 255)).build();
      final inputBuffer = processor.process(inputImage.tensorBuffer);

      // Ensure input shape is [1, 150, 150, 3]
      var input = inputBuffer.buffer.asFloat32List().reshape([1, 150, 150, 3]);
      var output = List.generate(1, (_) => List.filled(1, 0.0));

      _interpreter!.run(input, output);

      _probability = output[0][0];
      setState(() {
        _result = _probability > 0.5 ? "Infertile" : "Fertile";
      });
    } catch (e) {
      print("Error during classification: $e");
      setState(() {
        _result = "Error during classification";
        _probability = 0.0;
      });
    }
  }

  Widget _buildProbabilityIndicator() {
    Color getColor() {
      if (_result == "Fertile") {
        return Colors.green;
      } else if (_result == "Infertile") {
        return Colors.red;
      } else {
        return Colors.grey;
      }
    }

    // Calculate both probabilities
    double fertileProbability = 1 - _probability;
    double infertileProbability = _probability;

    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20.0),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                "Fertile",
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 11,
                  fontFamily: "'Consolas', cursive",
                ),
              ),
              Text(
                "Infertile",
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 11,
                  letterSpacing: 1.5,
                  fontFamily: 'consolas',
                ),
              ),
            ],
          ),
        ),
        SizedBox(height: 4),
        Container(
          width: MediaQuery.of(context).size.width * 0.8,
          height: 20,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(10),
            color: Colors.grey[800],
          ),
          child: Stack(
            children: [
              // Probability "Infertile"
              Container(
                width: MediaQuery.of(context).size.width * 0.8 * _probability,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(10),
                  color: Colors.red.withOpacity(0.7),
                ),
              ),
              // Probability "Fertile"
              Container(
                width:
                    MediaQuery.of(context).size.width *
                    0.8 *
                    (1 - _probability),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(10),
                  color: Colors.green.withOpacity(0.7),
                ),
              ),
              // Indicator
              Positioned(
                left:
                    MediaQuery.of(context).size.width *
                    0.8 *
                    (1 - _probability),
                child: Container(width: 3, height: 20, color: Colors.white),
              ),
            ],
          ),
        ),
        SizedBox(height: 8),
        // Display both probabilities
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              "Fertile: ${(fertileProbability * 100).toStringAsFixed(2)}%",
              style: TextStyle(
                fontWeight: FontWeight.bold,
                color: Colors.green,
                fontSize: 10.6,
                letterSpacing: 1.9,
                fontFamily: 'consolas',
              ),
            ),
            SizedBox(width: 20),
            Text(
              "Infertile: ${(infertileProbability * 100).toStringAsFixed(2)}%",
              style: TextStyle(
                fontWeight: FontWeight.bold,
                color: Colors.red,
                fontSize: 10.6,
                letterSpacing: 1.9,
                fontFamily: 'consolas',
              ),
            ),
          ],
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Egg Classifier',
          style: TextStyle(
            fontSize: 12, // Atur ukuran teks di sini
            fontWeight: FontWeight.bold,
            fontFamily: "'Roboto Mono', monospace",
            fontStyle: FontStyle.italic,
            letterSpacing: 1.9, // Atur jarak antar huruf
            // Pastikan fontFamily tetap konsisten
          ),
        ),
        centerTitle: true,
      ),
      body:
          _isModelLoaded
              ? SafeArea(
                child: Center(
                  child: SingleChildScrollView(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Card(
                          margin: EdgeInsets.all(16),
                          child: Padding(
                            padding: EdgeInsets.all(16),
                            child: Column(
                              children: [
                                _image == null
                                    ? Container(
                                      height: 250,
                                      width: 250,
                                      decoration: BoxDecoration(
                                        color: Colors.grey[900],
                                        borderRadius: BorderRadius.circular(12),
                                      ),
                                      child: Icon(
                                        Icons.egg_outlined,
                                        size: 100,
                                        color: Colors.grey[700],
                                      ),
                                    )
                                    : ClipRRect(
                                      borderRadius: BorderRadius.circular(12),
                                      child: Image.file(
                                        _image!,
                                        height: 250,
                                        width: 250,
                                        fit: BoxFit.cover,
                                      ),
                                    ),
                                SizedBox(height: 24),
                                _isProcessing
                                    ? CircularProgressIndicator(
                                      color: Color(0xFF4CAF50),
                                    )
                                    : Column(
                                      children: [
                                        Text(
                                          'Result: $_result',
                                          style: TextStyle(
                                            fontSize: 13,
                                            fontWeight: FontWeight.bold,
                                            fontStyle: FontStyle.italic,
                                            letterSpacing:
                                                1.5, // Atur jarak antar huruf

                                            fontFamily:
                                                'consolas', // Tambahkan fontFamily di sini

                                            color:
                                                _result == "Fertile"
                                                    ? Colors.green
                                                    : _result == "Infertile"
                                                    ? Colors.red
                                                    : Colors.white,
                                          ),
                                        ),
                                        SizedBox(height: 16),
                                        if (_result == "Fertile" ||
                                            _result == "Infertile")
                                          _buildProbabilityIndicator(),
                                      ],
                                    ),
                              ],
                            ),
                          ),
                        ),
                        SizedBox(height: 16),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            ElevatedButton.icon(
                              onPressed: () => _pickImage(ImageSource.gallery),
                              icon: Icon(
                                Icons.photo_library,
                                color: Colors.white,
                              ),
                              label: Text(
                                'Gallery',
                                style: TextStyle(
                                  fontSize: 11,
                                  letterSpacing: 1.5, // Atur jarak antar huruf
                                  color: Colors.white,
                                  fontStyle: FontStyle.italic,

                                  fontFamily:
                                      'consolas', // Tambahkan fontFamily di sini
                                ),
                              ),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: const Color.fromARGB(
                                  255,
                                  76,
                                  160,
                                  175,
                                ), // Warna tombol
                                padding: EdgeInsets.symmetric(
                                  horizontal: 19,
                                  vertical: 6,
                                ),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(8),
                                ),
                              ),
                            ),
                            SizedBox(width: 16),
                            ElevatedButton.icon(
                              onPressed: () => _pickImage(ImageSource.camera),
                              icon: Icon(Icons.camera_alt, color: Colors.white),
                              label: Text(
                                'Camera',
                                style: TextStyle(
                                  fontSize: 11,
                                  fontStyle: FontStyle.italic,
                                  letterSpacing: 1.5, // Atur jarak antar huruf
                                  fontFamily:
                                      'consolas', // Tambahkan fontFamily di sini

                                  color: Colors.white,
                                ),
                              ),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: const Color.fromARGB(
                                  255,
                                  245,
                                  146,
                                  80,
                                ), // Warna tombol
                                padding: EdgeInsets.symmetric(
                                  horizontal: 19,
                                  vertical: 6,
                                ),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(8),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
              )
              : Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(color: Color(0xFF4CAF50)),
                    SizedBox(height: 16),
                    Text(
                      "Loading model...",
                      style: TextStyle(
                        fontFamily:
                            "'Consolas', cursive", // Tambahkan fontFamily di sini

                        fontSize: 16,
                        fontWeight: FontWeight.w500,
                        color: Colors.white,
                      ),
                    ),
                  ],
                ),
              ),
    );
  }
}
