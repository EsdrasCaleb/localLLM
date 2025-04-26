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

class a4jUtil_URLFriendlyName_0_1_Test {

    private a4jUtil util;

    @BeforeEach
    public void setUp() {
        util = new a4jUtil();
    }

    @Test
    public void testURLFriendlyName() {
        // Test case 1: Normal input
        assertEquals("example_name", util.URLFriendlyName("example name"));
        // Test case 2: Input with multiple spaces
        assertEquals("example_name", util.URLFriendlyName(" example name "));
        // Test case 3: Input with special characters
        assertEquals("example-name-123", util.URLFriendlyName("example!name#123"));
    }
}
