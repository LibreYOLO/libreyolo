import CoreML
import CoreVideo
import Foundation
import UIKit
import XCTest

final class CoreMLSmokeTests: XCTestCase {
    struct Expected: Decodable {
        let rawTopScore: Double
        let nmsTopScore: Double
    }

    func testRawCoreMLPackageCompilesAndPredictsOnPhysicalIPhone() throws {
        try requirePhysicalDevice()

        let expected = try loadExpected()
        let model = try loadCompiledModel(named: "LibreYOLOXnRaw")
        let pixelBuffer = try loadPixelBuffer(named: "parkour_yolox416")
        let output = try model.prediction(
            from: MLDictionaryFeatureProvider(
                dictionary: ["image": MLFeatureValue(pixelBuffer: pixelBuffer)]
            )
        )

        let outputName = try XCTUnwrap(
            model.modelDescription.outputDescriptionsByName.keys.first
        )
        let scores = try XCTUnwrap(
            output.featureValue(for: outputName)?.multiArrayValue
        )
        let topScore = rawYOLOXTopScore(scores)
        logComparison(
            name: "raw",
            iPhoneScore: topScore,
            macOSScore: expected.rawTopScore
        )

        XCTAssertGreaterThan(topScore, 0.25)
        XCTAssertLessThan(abs(topScore - expected.rawTopScore), 0.08)
    }

    func testEmbeddedNMSPackageCompilesAndPredictsOnPhysicalIPhone() throws {
        try requirePhysicalDevice()

        let expected = try loadExpected()
        let model = try loadCompiledModel(named: "LibreYOLOXnNMS")
        let pixelBuffer = try loadPixelBuffer(named: "parkour_yolox416")
        let output = try model.prediction(
            from: MLDictionaryFeatureProvider(
                dictionary: ["image": MLFeatureValue(pixelBuffer: pixelBuffer)]
            )
        )

        let confidence = try XCTUnwrap(
            output.featureValue(for: "confidence")?.multiArrayValue
        )
        let coordinates = try XCTUnwrap(
            output.featureValue(for: "coordinates")?.multiArrayValue
        )
        let topScore = maxValue(confidence)
        logComparison(
            name: "embedded-nms",
            iPhoneScore: topScore,
            macOSScore: expected.nmsTopScore
        )

        XCTAssertEqual(coordinates.shape.last?.intValue, 4)
        XCTAssertGreaterThan(topScore, 0.25)
        XCTAssertLessThan(abs(topScore - expected.nmsTopScore), 0.08)
    }

    private func requirePhysicalDevice() throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("CoreML iPhone smoke test requires a physical iOS device")
        #endif
    }

    private func loadExpected() throws -> Expected {
        let url = try resourceURL(named: "expected", withExtension: "json")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Expected.self, from: data)
    }

    private func loadCompiledModel(named name: String) throws -> MLModel {
        let packageURL = try resourceURL(named: name, withExtension: "mlpackage")
        let compiledURL = try MLModel.compileModel(at: packageURL)
        let config = MLModelConfiguration()
        config.computeUnits = .all
        return try MLModel(contentsOf: compiledURL, configuration: config)
    }

    private func resourceURL(named name: String, withExtension ext: String) throws -> URL {
        if let url = resourceBundle.url(
            forResource: name,
            withExtension: ext,
            subdirectory: "Resources"
        ) {
            return url
        }
        if let url = resourceBundle.url(forResource: name, withExtension: ext) {
            return url
        }
        throw XCTSkip("Missing resource \(name).\(ext); run prepare_resources.py")
    }

    private var resourceBundle: Bundle {
        #if SWIFT_PACKAGE
        return Bundle.module
        #else
        return Bundle(for: CoreMLSmokeTests.self)
        #endif
    }

    private func loadPixelBuffer(named name: String) throws -> CVPixelBuffer {
        let imageURL = try resourceURL(named: name, withExtension: "png")
        let image = try XCTUnwrap(UIImage(contentsOfFile: imageURL.path)?.cgImage)
        let width = image.width
        let height = image.height

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32ARGB,
            [
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            ] as CFDictionary,
            &pixelBuffer
        )
        XCTAssertEqual(status, kCVReturnSuccess)

        let buffer = try XCTUnwrap(pixelBuffer)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        let context = try XCTUnwrap(
            CGContext(
                data: CVPixelBufferGetBaseAddress(buffer),
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
            )
        )
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }

    private func rawYOLOXTopScore(_ output: MLMultiArray) -> Double {
        let shape = output.shape.map(\.intValue)
        precondition(shape.count == 3 && shape[2] >= 6)

        var topScore = -Double.infinity
        for boxIndex in 0..<shape[1] {
            let objectness = value(output, [0, boxIndex, 4])
            for classIndex in 5..<shape[2] {
                let score = objectness * value(output, [0, boxIndex, classIndex])
                if score > topScore {
                    topScore = score
                }
            }
        }
        return topScore
    }

    private func maxValue(_ output: MLMultiArray) -> Double {
        var topScore = -Double.infinity
        for index in 0..<output.count {
            let score = output[index].doubleValue
            if score > topScore {
                topScore = score
            }
        }
        return topScore
    }

    private func value(_ output: MLMultiArray, _ indices: [Int]) -> Double {
        output[indices.map { NSNumber(value: $0) }].doubleValue
    }

    private func logComparison(name: String, iPhoneScore: Double, macOSScore: Double) {
        let message = String(
            format: "CoreMLSmoke %@ top score: iPhone=%.12f macOS=%.12f delta=%.12f",
            name,
            iPhoneScore,
            macOSScore,
            abs(iPhoneScore - macOSScore)
        )
        print(message)
    }
}
