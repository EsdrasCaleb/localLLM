package net.kencochrane.a4j.util;

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

    @Test
    void testStripString() {
        a4jUtil a4jUtil = new a4jUtil();
        // Test with allowedChars set to all characters
        String allowedChars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        String inputString = "Hello, World!";
        String expectedOutput = "Hello, World!";
        assertEquals(expectedOutput, a4jUtil.stripString(allowedChars, inputString));
        // Test with allowedChars set to only 'a'
        allowedChars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        inputString = "Hello, World!";
        expectedOutput = "Hello, World!";
        assertEquals(expectedOutput, a4jUtil.stripString(allowedChars, inputString));
        // Test with allowedChars set to empty string
        allowedChars = "";
        inputString = "Hello, World!";
        expectedOutput = "Hello, World!";
        assertEquals(expectedOutput, a4jUtil.stripString(allowedChars, inputString));
        // Test with allowedChars set to null
        allowedChars = null;
        inputString = "Hello, World!";
        expectedOutput = "Hello, World!";
        assertEquals(expectedOutput, a4jUtil.stripString(allowedChars, inputString));
        // Test with allowedChars set to a single character
        allowedChars = "a";
        inputString = "Hello, World!";
        expectedOutput = "Hello, World!";
        assertEquals(expectedOutput, a4jUtil.stripString(allowedChars, inputString));
    }
}
