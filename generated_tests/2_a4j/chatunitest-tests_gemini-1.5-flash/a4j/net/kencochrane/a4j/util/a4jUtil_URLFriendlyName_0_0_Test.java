package net.kencochrane.a4j.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Properties;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class a4jUtil_URLFriendlyName_0_0_Test {

    private a4jUtil util;

    private LoadProperties mockLoadProperties;

    @BeforeEach
    void setUp() {
        util = new a4jUtil();
        mockLoadProperties = mock(LoadProperties.class);
        try {
            Field field = a4jUtil.class.getDeclaredField("loadProperties");
            field.setAccessible(true);
            field.set(util, mockLoadProperties);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set LoadProperties mock: " + e.getMessage());
        }
    }

    @Test
    void testURLFriendlyName_nullInput() {
        assertNull(util.URLFriendlyName(null), "Null input should return null");
    }

    @Test
    void testURLFriendlyName_emptyInput() {
        String result = util.URLFriendlyName("");
        assertEquals("", result, "Empty input should return empty string");
    }

    @Test
    void testURLFriendlyName_onlyAllowedChars() {
        String input = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-~|+";
        String expected = input;
        // Avoid NPE
        when(mockLoadProperties.getProperties()).thenReturn(new Properties());
        String result = util.URLFriendlyName(input);
        assertEquals(expected, result, "Input with only allowed chars should remain unchanged");
    }

    @Test
    void testURLFriendlyName_withSpaces() {
        String input = "This is a test";
        String expected = "This%20is%20a%20test";
        Properties props = new Properties();
        props.setProperty("URLSeperator", "%20");
        when(mockLoadProperties.getProperties()).thenReturn(props);
        String result = util.URLFriendlyName(input);
        assertEquals(expected, result, "Spaces should be replaced with URLSeperator");
    }

    @Test
    void testURLFriendlyName_withDisallowedChars() {
        String input = "This is a test with !@#$%^&*() characters.";
        String expected = "Thisisatestwithcharacters";
        Properties props = new Properties();
        // To simplify the expected result.
        props.setProperty("URLSeperator", "");
        when(mockLoadProperties.getProperties()).thenReturn(props);
        String result = util.URLFriendlyName(input);
        assertEquals(expected, result, "Disallowed characters should be removed");
    }

    @Test
    void testURLFriendlyName_mixedInput() {
        String input = "This is a Test 123 with !@#$%^&*() and spaces.";
        String expected = "Thisisatest123withandspaces";
        Properties props = new Properties();
        // To simplify the expected result.
        props.setProperty("URLSeperator", "");
        when(mockLoadProperties.getProperties()).thenReturn(props);
        String result = util.URLFriendlyName(input);
        assertEquals(expected, result, "Mixed input should handle spaces and disallowed chars correctly");
    }

    // Dummy class for mocking
    static class LoadProperties {

        public static LoadProperties instance() {
            return new LoadProperties();
        }

        public Properties getProperties() {
            return new Properties();
        }
    }
}
