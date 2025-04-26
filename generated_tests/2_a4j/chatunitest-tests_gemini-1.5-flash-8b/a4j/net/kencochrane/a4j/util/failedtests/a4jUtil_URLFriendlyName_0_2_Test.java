package net.kencochrane.a4j.util;

import java.util.Properties;
import java.util.regex.Pattern;
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

class a4jUtil_URLFriendlyName_0_2_Test {

    private a4jUtil a4jUtil;

    private Properties mockProperties;

    @BeforeEach
    void setUp() {
        a4jUtil = new a4jUtil();
        mockProperties = Mockito.mock(Properties.class);
        // Mock the URL separator
        Mockito.when(mockProperties.getProperty("URLSeperator")).thenReturn("-");
        // Crucial for testing the stripString method, which is not shown
        Mockito.when(mockProperties.getProperty("allowedChars")).thenReturn("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-~|+");
    }

    @Test
    void testURLFriendlyNameWithSpaces() {
        String input = "Test String";
        String expected = "Test-String";
        String actual = a4jUtil.URLFriendlyName(input);
        assertEquals(expected, actual);
    }

    @Test
    void testURLFriendlyNameWithSpecialChars() {
        String input = "Test!String";
        String expected = "TestString";
        String actual = a4jUtil.URLFriendlyName(input);
        assertEquals(expected, actual);
    }

    @Test
    void testURLFriendlyNameWithEmptyInput() {
        String input = "";
        String expected = "";
        String actual = a4jUtil.URLFriendlyName(input);
        assertEquals(expected, actual);
    }

    @Test
    void testURLFriendlyNameWithNullInput() {
        String input = null;
        // Or throw an exception, depending on the expected behavior
        String expected = "";
        String actual = a4jUtil.URLFriendlyName(input);
        assertEquals(expected, actual);
    }

    @Test
    void testURLFriendlyNameWithOnlyAllowedChars() {
        String input = "allowedCharsOnly";
        String expected = "allowedCharsOnly";
        String actual = a4jUtil.URLFriendlyName(input);
        assertEquals(expected, actual);
    }

    // Add more tests to cover different scenarios, including edge cases.
    // For example, test with input containing only disallowed characters.
    // Helper method (crucial for testing the stripString method)
    private String stripString(String allowedChars, String input) {
        // Handle null input
        if (input == null)
            return "";
        return input.replaceAll("[^" + Pattern.quote(allowedChars) + "]", "");
    }
}
