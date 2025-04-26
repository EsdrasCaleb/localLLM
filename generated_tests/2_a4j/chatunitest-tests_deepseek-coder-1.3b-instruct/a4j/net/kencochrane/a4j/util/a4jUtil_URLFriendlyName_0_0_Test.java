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

class a4jUtil_URLFriendlyName_0_0_Test {

    @Test
    void testURLFriendlyName() {
        // Arrange
        a4jUtil a4j = new a4jUtil();
        String testName = "Test Name";
        String expectedResult = "Test-Name";
        // Act
        String result = a4j.URLFriendlyName(testName);
        // Assert
        assertEquals(expectedResult, result);
    }
}
