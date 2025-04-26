package net.kencochrane.a4j.util;

import java.lang.reflect.Method;
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

    private a4jUtil a4jUtil;

    @BeforeEach
    public void setUp() {
        a4jUtil = new a4jUtil();
    }

    @Test
    public void testStripString_AllowedCharsEmpty_ReturnsEmptyString() throws Exception {
        // Arrange
        String allowedChars = "";
        String inputString = "test";
        // Act
        String result = a4jUtil.stripString(allowedChars, inputString);
        // Assert
        assertEquals("", result);
    }

    @Test
    public void testStripString_InputStringEmpty_ReturnsEmptyString() throws Exception {
        // Arrange
        String allowedChars = "abc";
        String inputString = "";
        // Act
        String result = a4jUtil.stripString(allowedChars, inputString);
        // Assert
        assertEquals("", result);
    }

    @Test
    public void testStripString_NoMatchingChars_ReturnsEmptyString() throws Exception {
        // Arrange
        String allowedChars = "xyz";
        String inputString = "abc";
        // Act
        String result = a4jUtil.stripString(allowedChars, inputString);
        // Assert
        assertEquals("", result);
    }

    @Test
    public void testStripString_AllCharsMatch_ReturnsInputString() throws Exception {
        // Arrange
        String allowedChars = "abc";
        String inputString = "abc";
        // Act
        String result = a4jUtil.stripString(allowedChars, inputString);
        // Assert
        assertEquals("abc", result);
    }

    @Test
    public void testStripString_SomeCharsMatch_ReturnsFilteredString() throws Exception {
        // Arrange
        String allowedChars = "aeiou";
        String inputString = "hello world";
        // Act
        String result = a4jUtil.stripString(allowedChars, inputString);
        // Assert
        assertEquals("eoo", result);
    }

    @Test
    public void testStripString_AllowedCharsContainSpaces_ReturnsFilteredString() throws Exception {
        // Arrange
        String allowedChars = "a b";
        String inputString = "a quick brown fox";
        // Act
        String result = a4jUtil.stripString(allowedChars, inputString);
        // Assert
        assertEquals("a  ", result);
    }

    @Test
    public void testStripString_InputStringContainsSpecialChars_ReturnsFilteredString() throws Exception {
        // Arrange
        String allowedChars = "!@#";
        String inputString = "h!e@l#l$o";
        // Act
        String result = a4jUtil.stripString(allowedChars, inputString);
        // Assert
        assertEquals("!@#", result);
    }

    @Test
    public void testStripString_AllowedCharsAndInputStringContainSameSpecialChars_ReturnsFilteredString() throws Exception {
        // Arrange
        String allowedChars = "!@#$%^&*()";
        String inputString = "!@#$%^&*()";
        // Act
        String result = a4jUtil.stripString(allowedChars, inputString);
        // Assert
        assertEquals("!@#$%^&*()", result);
    }
}
