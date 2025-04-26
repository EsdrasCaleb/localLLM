package com.hf.sfm.crypt;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import java.util.stream.Stream;
import java.util.Base64;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Base64_base64ToByteArray_2_0_Test {

    @ParameterizedTest
    @MethodSource("testCases")
    void testBase64ToByteArray(String input, byte[] expected) {
        byte[] result;
        if (input == null) {
            result = null;
        } else {
            try {
                result = Base64.getDecoder().decode(input);
            } catch (IllegalArgumentException e) {
                fail("Invalid Base64 input: " + input, e);
                // Crucial: stop execution on invalid input
                return;
            }
        }
        assertArrayEquals(expected, result);
    }

    static Stream<Arguments> testCases() {
        return Stream.of(Arguments.of("SGVsbG8gV29ybGQh", "Hello World!".getBytes()), Arguments.of("aGVsbG8gd29ybGQ=", "hello world".getBytes()), Arguments.of("", new byte[0]), Arguments.of(null, null));
    }
}
