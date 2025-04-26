package net.kencochrane.a4j.util;

import java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Properties;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;

public class a4jUtil_stripString_1_0_Test {

    @Test
    public void testStripString_AllowedCharsEmpty() {
        a4jUtil util = new a4jUtil();
        String allowedChars = "";
        String inputString = "Hello, World!";
        String expected = "";
        String actual = util.stripString(allowedChars, inputString);
        assertEquals(expected, actual);
    }

    @Test
    public void testStripString_AllowedCharsNull() {
        a4jUtil util = new a4jUtil();
        String allowedChars = null;
        String inputString = "Hello, World!";
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> util.stripString(allowedChars, inputString));
        assertEquals("Allowed characters must be a non-empty string", exception.getMessage());
    }

    @Test
    public void testStripString_InputNull() {
        a4jUtil util = new a4jUtil();
        String allowedChars = "abc";
        String inputString = null;
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> util.stripString(allowedChars, inputString));
        assertEquals("Input string must be a non-empty string", exception.getMessage());
    }

    @Test
    public void testStripString_AllowedCharsEmptyInputString() {
        a4jUtil util = new a4jUtil();
        String allowedChars = "";
        String inputString = "";
        String expected = "";
        String actual = util.stripString(allowedChars, inputString);
        assertEquals(expected, actual);
    }

    @Test
    public void testStripString_AllowedCharsNotPresent() {
        a4jUtil util = new a4jUtil();
        String allowedChars = "def";
        String inputString = "Hello, World!";
        String expected = inputString;
        String actual = util.stripString(allowedChars, inputString);
        assertEquals(expected, actual);
    }

    @Test
    public void testStripString_AllowedCharsPresent() {
        a4jUtil util = new a4jUtil();
        String allowedChars = "abc";
        String inputString = "Hello, World!";
        String expected = "Hello, World!";
        String actual = util.stripString(allowedChars, inputString);
        assertEquals(expected, actual);
    }
}

class a4jUtil {

    public String stripString(String allowedChars, String inputString) {
        if (allowedChars == null || inputString == null) {
            throw new IllegalArgumentException("Input string must be a non-empty string");
        }
        if (allowedChars.isEmpty()) {
            return "";
        }
        StringBuilder result = new StringBuilder();
        for (char c : inputString.toCharArray()) {
            if (allowedChars.indexOf(c) != -1) {
                result.append(c);
            }
        }
        return result.toString();
    }
}
