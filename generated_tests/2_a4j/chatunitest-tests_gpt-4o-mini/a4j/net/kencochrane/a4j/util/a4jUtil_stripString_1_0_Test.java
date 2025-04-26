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

public class a4jUtil_stripString_1_0_Test {

    private a4jUtil util;

    @BeforeEach
    public void setUp() {
        util = new a4jUtil();
    }

    @Test
    public void testStripString_WithAllowedChars() {
        String allowedChars = "abc";
        String input = "abcdef";
        String expected = "abc";
        String actual = util.stripString(allowedChars, input);
        assertEquals(expected, actual);
    }

    @Test
    public void testStripString_WithEmptyInputString() {
        String allowedChars = "abc";
        String input = "";
        String expected = "";
        String actual = util.stripString(allowedChars, input);
        assertEquals(expected, actual);
    }

    @Test
    public void testStripString_WithNoAllowedChars() {
        String allowedChars = "";
        String input = "abcdef";
        String expected = "";
        String actual = util.stripString(allowedChars, input);
        assertEquals(expected, actual);
    }

    @Test
    public void testStripString_WithNoMatchingChars() {
        String allowedChars = "xyz";
        String input = "abcdef";
        String expected = "";
        String actual = util.stripString(allowedChars, input);
        assertEquals(expected, actual);
    }

    @Test
    public void testStripString_WithAllMatchingChars() {
        String allowedChars = "abcdef";
        String input = "abcdef";
        String expected = "abcdef";
        String actual = util.stripString(allowedChars, input);
        assertEquals(expected, actual);
    }

    @Test
    public void testStripString_WithSpecialCharacters() {
        String allowedChars = "abc!@#";
        String input = "abc123!@#";
        String expected = "abc!@#";
        String actual = util.stripString(allowedChars, input);
        assertEquals(expected, actual);
    }
}
