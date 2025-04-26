package com.hf.sfm.crypt;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import java.util.stream.Stream;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Base64_base64ToByteArray_2_4_Test {

    @Test
    void base64ToByteArray() {
        assertEquals(new byte[] { 1, 2, 3 }, Base64.base64ToByteArray("AQIDBAUGBwgJCg=="));
    }

    @ParameterizedTest
    @MethodSource("dataProvider")
    void base64ToByteArray(String s, byte[] expected) {
        assertEquals(expected, Base64.base64ToByteArray(s));
    }

    private static Stream<Arguments> dataProvider() {
        return Stream.of(Arguments.of("AQIDBAUGBwgJCg==", new byte[] { 1, 2, 3 }), Arguments.of("AQIDBAUGBwgJCg==", new byte[] { 1, 2, 3 }), Arguments.of("AQIDBAUGBwgJCg==", new byte[] { 1, 2, 3 }), Arguments.of("AQIDBAUGBwgJCg==", new byte[] { 1, 2, 3 }));
    }

    @ParameterizedTest
    @ValueSource(strings = { "AQIDBAUGBwgJCg==", "AQIDBAUGBwgJCg==", "AQIDBAUGBwgJCg==", "AQIDBAUGBwgJCg==" })
    void base64ToByteArray_ValueSource(String s) {
        assertEquals(new byte[] { 1, 2, 3 }, Base64.base64ToByteArray(s));
    }
}
