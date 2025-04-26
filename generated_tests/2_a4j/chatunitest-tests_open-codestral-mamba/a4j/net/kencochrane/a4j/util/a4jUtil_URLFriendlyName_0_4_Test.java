package net.kencochrane.a4j.util;

import java.util.Properties;
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

class a4jUtil_URLFriendlyName_0_4_Test {

    private a4jUtil a4jUtil;

    @BeforeEach
    void setUp() {
        this.a4jUtil = new a4jUtil();
    }

    @Test
    void testURLFriendlyName() {
        // Arrange
        String input = "Hello World";
        String expectedOutput = "Hello|World";
        // Mock the LoadProperties class and its getProperties() method
        LoadProperties mockLoadProperties = Mockito.mock(LoadProperties.class);
        Properties mockProperties = Mockito.mock(Properties.class);
        Mockito.when(mockLoadProperties.getProperties()).thenReturn(mockProperties);
        Mockito.when(mockProperties.getProperty("URLSeperator")).thenReturn("|");
        // Act
        String actualOutput = a4jUtil.URLFriendlyName(input);
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }
}
