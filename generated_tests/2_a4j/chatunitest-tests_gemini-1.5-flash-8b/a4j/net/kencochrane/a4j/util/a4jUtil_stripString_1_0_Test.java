package net.kencochrane.a4j.util;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import java.util.stream.Stream;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Properties;

class a4jUtil_stripString_1_0_Test {

    private a4jUtil util = new a4jUtil();

    @ParameterizedTest
    @MethodSource("testCases")
    void testStripString(String allowedChars, String inputString, String expectedOutput) {
        assertEquals(expectedOutput, util.stripString(allowedChars, inputString));
    }

    static Stream<Arguments> testCases() {
        return // Empty string if no matching characters
        // Empty string if no matching characters
        // Empty string if no matching characters
        Stream.// Empty string if no matching characters
        of(// Empty string if no matching characters
        Arguments.of("abc", "abcdefg", "abc"), // Empty string if no matching characters
        Arguments.of("aeiou", "aeiouAEIOU", "aeiou"), // Empty string if no matching characters
        Arguments.of("123", "123456789", "123"), // Empty string if allowedChars is empty
        Arguments.of("abc", "ABCabc", "abc"), // Empty string if inputString is empty
        Arguments.of("0123456789", "a1b2c3d4e5f6g7h8i9j0", "1234567890"), Arguments.of("abc", "xyz", ""), Arguments.of("", "abc", ""), Arguments.of("abc", "", ""));
    }
}
