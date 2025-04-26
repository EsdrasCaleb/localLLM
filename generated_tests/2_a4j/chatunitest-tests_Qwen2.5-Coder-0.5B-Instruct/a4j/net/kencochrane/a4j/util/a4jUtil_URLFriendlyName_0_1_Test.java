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

public class a4jUtil_URLFriendlyName_0_1_Test {

    @Test
    public void testURLFriendlyName() {
        // Arrange
        String input = "Hello World!";
        String expectedOutput = "hello-world!";
        // Create an instance of a4jUtil
        a4jUtil util = Mockito.mock(a4jUtil.class);
        // Set up the behavior of the method
        when(util.URLFriendlyName(input)).thenReturn(expectedOutput);
        // Act
        String result = util.URLFriendlyName(input);
        // Verify the result
        assertEquals(expectedOutput, result);
    }
}
