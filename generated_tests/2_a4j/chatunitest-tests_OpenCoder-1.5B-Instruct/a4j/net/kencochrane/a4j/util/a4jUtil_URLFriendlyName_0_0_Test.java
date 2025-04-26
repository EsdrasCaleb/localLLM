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

public class a4jUtil_URLFriendlyName_0_0_Test {

    @Test
    public void testURLFriendlyName() {
        // Arrange
        a4jUtil util = new a4jUtil();
        String name = "This is a test name";
        String expected = "this-is-a-test-name";
        // Act
        String actual = util.URLFriendlyName(name);
        // Assert
        assertEquals(expected, actual);
    }
}
