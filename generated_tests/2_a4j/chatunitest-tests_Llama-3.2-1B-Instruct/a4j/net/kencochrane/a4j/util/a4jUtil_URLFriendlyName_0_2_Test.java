package net.kencochrane.a4j.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Properties;
import java.util.regex.Pattern;
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
public class a4jUtil_URLFriendlyName_0_2_Test {

    @Mock
    private Properties props;

    @InjectMocks
    private a4jUtil focal;

    @Test
    public void testURLFriendlyName() {
        // Arrange
        String input = "Hello World!";
        String expected = "hello-world";
        // Act
        String result = focal.URLFriendlyName(input);
        // Assert
        assertEquals(expected, result);
    }

    @Test
    public void testURLFriendlyNameNoNonAlphanumeric() {
        // Arrange
        String input = "Hello@World!";
        // Act
        String result = focal.URLFriendlyName(input);
        // Assert
        assertEquals("hello-world", result);
    }

    @Test
    public void testURLFriendlyNameNoSeperator() {
        // Arrange
        String input = "Hello World";
        // Act
        String result = focal.URLFriendlyName(input);
        // Assert
        assertEquals("hello-world", result);
    }

    @Test
    public void testURLFriendlyNameEmptyString() {
        // Arrange
        String input = "";
        // Act
        String result = focal.URLFriendlyName(input);
        // Assert
        assertEquals("hello-world", result);
    }

    @Test
    public void testURLFriendlyNameSingleNonAlphanumeric() {
        // Arrange
        String input = "Hello!";
        // Act
        String result = focal.URLFriendlyName(input);
        // Assert
        assertEquals("hello", result);
    }

    @Test
    public void testURLFriendlyNameNoSpaces() {
        // Arrange
        String input = "Hello";
        // Act
        String result = focal.URLFriendlyName(input);
        // Assert
        assertEquals("hello", result);
    }

    @Test
    public void testURLFriendlyNameAllNonAlphanumeric() {
        // Arrange
        String input = "!@#$%^&*()";
        // Act
        String result = focal.URLFriendlyName(input);
        // Assert
        assertEquals("hello-world", result);
    }
}
